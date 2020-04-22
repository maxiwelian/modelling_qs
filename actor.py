import ray

@ray.remote(num_gpus=1)
class Network(object):
    def __init__(self, config, gpu_id):
        import tensorflow as tf
        import numpy as np
        from time import time

        if config['seed']:
            tf.random.set_seed(7)
            np.random.seed(7)

        from sampling.sampling import MetropolisHasting, RandomWalker
        from model.fermi_net import fermiNet
        from energy.energy import compute_local_energy
        from model.gradients import extract_grads, KFAC_Actor
        from pretraining.pretraining import Pretrainer
        from utils.utils import load_model, load_sample, filter_dict, tofloat
        from actor_proxy import clip

        self.n_samples = config['n_samples_actor']
        config['n_samples'] = self.n_samples
        self.r_atoms = config['r_atoms']
        self.z_atoms = config['z_atoms']
        self.model_path = config['model_path']

        ferminet_params = filter_dict(config, fermiNet)
        self.model = fermiNet(gpu_id, **ferminet_params)
        print('initialized model')

        # * - pretraining
        pretrainer_params = filter_dict(config, Pretrainer)
        self.pretrainer = Pretrainer(**pretrainer_params)

        # * - sampling
        self.sample_space = RandomWalker(tf.zeros(3),
                                         tf.eye(3) * config['sampling_init'],
                                         tf.zeros(3),
                                         tf.eye(3) * config['sampling_steps'])

        model_sampler_params = filter_dict(config, MetropolisHasting)
        self.model_sampler = MetropolisHasting(self.model, self.pretrainer, self.sample_space, **model_sampler_params)
        self.samples = self.model_sampler.initialize_samples()
        self.burn()
        self.pretrain_samples = self.model_sampler.initialize_samples()
        self.burn_pretrain()

        # * - model details
        self.n_params = np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_weights])
        self.n_layers = len(self.model.trainable_weights)
        self.layers = [w.name for w in self.model.trainable_weights]
        self.trainable_shapes = [w.shape for w in self.model.trainable_weights]

        # * - optimizers
        self.optimizer_pretrain = tf.keras.optimizers.Adam(learning_rate=0.001)
        if config['opt'] == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr0'])
            print('Using ADAM optimizer')
        elif config['opt'] == 'kfac':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=config['lr0'] / 10, decay=config['decay'])
            kfac_config = filter_dict(config, KFAC_Actor)
            self.kfac = KFAC_Actor(self.model, **kfac_config)
            print('Using kfac optimizer')
        print('n_samples per actor: ', self.n_samples)

        # store references to avoid reimport
        self._extract_grads = extract_grads
        self._tofloat = tofloat
        self._tf = tf
        self._compute_local_energy, self._clip = compute_local_energy, clip
        self._time = time
        self._load_model = load_model
        self._load_sample = load_sample

        self.iteration = config['load_iteration']
        self.acceptance = tf.constant(0.0, dtype=tf.float32)
        self.e_loc = tf.zeros(len(self.samples))
        self.amps = tf.zeros(len(self.samples))

    # gradients & energy
    def get_energy(self):
        self.samples, self.amps, self.acceptance = self.model_sampler.sample(self.samples)
        self.e_loc = self._compute_local_energy(self.r_atoms, self.samples, self.z_atoms, self.model)
        return self.e_loc

    def get_pretrain_grads(self):
        self.pretrain_samples, _, _ = self.model_sampler.sample_mixed(self.pretrain_samples)
        grads = self.pretrainer.compute_grads(self.samples, self.model)
        return grads

    def get_grads(self):
        e_loc = self.get_energy()
        e_loc_centered = self.center_energy(e_loc)
        grads = self._extract_grads(self.model,
                                    self.samples,
                                    e_loc_centered,
                                    self.n_samples)
        return grads

    def center_energy(self, e_loc):
        e_loc_clipped = self._clip(e_loc)
        e_loc_centered = e_loc_clipped - self._tf.reduce_mean(e_loc_clipped)
        return e_loc_centered

    def get_grads_and_maa_and_mss(self):
        e_loc = self.get_energy()
        e_loc_centered = self.center_energy(e_loc)
        grads, m_aa, m_ss = self.kfac.extract_grads_and_a_and_s(self.model,
                                                                self.samples,
                                                                e_loc_centered,
                                                                self.n_samples)
        return grads, m_aa, m_ss

    # samples
    def initialize_samples(self):
        self.samples = self.model_sampler.initialize_samples()

    def initialize_pretrain_samples(self):
        self.pretrain_samples = self.model_sampler.initialize_samples()

    def burn(self):
        self.samples = self.model_sampler.burn(self.samples)

    def burn_pretrain(self):
        self.pretrain_samples = self.model_sampler.burn_pretrain(self.pretrain_samples)

    def get_samples(self):
        return self.samples.numpy()

    def sample(self):
        self.samples, _, _ = self.model_sampler.sample(self.samples)

    # optimizers
    def update_weights(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def update_weights_pretrain(self, grads):
        self.optimizer_pretrain.apply_gradients(zip(grads, self.model.trainable_weights))

    def step_forward(self, updates):
        updates[-1] = self._tf.reshape(updates[-1], self.trainable_shapes[-1])
        for up, weight in zip(updates, self.model.trainable_weights):
            weight.assign_add(up)

    # network details
    def get_info(self):
        return self.amps, self.acceptance, self.samples, self.e_loc

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_model_details(self):
        details = (self.n_params, self.n_layers, self.trainable_shapes, self.layers)
        return details

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
            print('loading model at iteration ', self.iteration)
        self._load_model(self.model, path)

    def load_samples(self, path=None):
        if path is None:
            path = self.model_path[:-4] + 'pk'
        self.samples = \
            self._tf.convert_to_tensor(self._load_sample(path)[:self.n_samples, ...])