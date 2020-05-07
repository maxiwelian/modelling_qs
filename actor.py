import ray

@ray.remote(num_gpus=1)
class Network(object):
    def __init__(self, config, gpu_id):
        self.gpu_id = gpu_id

        import tensorflow as tf
        self._tf = tf
        import numpy as np
        self._np = np

        from time import time, sleep

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
        if config['system'] == 'Be':
            self.confirm_antisymmetric(config)
        print('initialized model')

        # * - pretraining
        pretrainer_params = filter_dict(config, Pretrainer)
        self.pretrainer = Pretrainer(**pretrainer_params)

        # * - sampling
        self.sample_space = RandomWalker(gpu_id, 0.0, config['sampling_steps'])

        model_sampler_params = filter_dict(config, MetropolisHasting)
        self.model_sampler = MetropolisHasting(self.model, self.pretrainer, self.sample_space, gpu_id, **model_sampler_params)
        self.samples = self.model_sampler.initialize_samples()
        # print('sample example: ', self.samples[0, 0, :])
        self.burn()
        self.pretrain_samples = self.model_sampler.initialize_samples()
        self.burn_pretrain()

        self.validation_samples = self.model_sampler.initialize_samples()

        # * - model details
        self.n_params = np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_weights])
        print('n params in network: ', self.n_params)
        self.n_layers = len(self.model.trainable_weights)
        self.layers = [w.name for w in self.model.trainable_weights]
        self.trainable_shapes = [w.shape for w in self.model.trainable_weights]

        # * - optimizers
        self.optimizer_pretrain = tf.keras.optimizers.Adam(learning_rate=0.001)
        if config['opt'] == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr0'])
            print('Using ADAM optimizer')
        elif config['opt'] == 'kfac':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=config['lr0'], decay=config['decay'])
            kfac_config = filter_dict(config, KFAC_Actor)
            self.kfac = KFAC_Actor(self.model, **kfac_config)
            print('Using kfac optimizer')
        assert self.n_samples == len(self.samples)
        print('n_samples per actor: ', self.n_samples, len(self.samples))

        # store references to avoid reimport
        self._extract_grads = extract_grads
        self._tofloat = tofloat

        self._compute_local_energy, self._clip = compute_local_energy, clip
        self._time = time
        self._load_model = load_model
        self._load_sample = load_sample

        self.iteration = config['load_iteration']
        self.acceptance = tf.constant(0.0, dtype=tf.float32)
        self.e_loc = tf.zeros(len(self.samples))
        self.amps = tf.zeros(len(self.samples))

        self.compute_validation_energy()

    # gradients & energy
    def get_energy(self):
        self.samples, self.amps, self.acceptance = self.model_sampler.sample(self.samples)
        self.e_loc = self._compute_local_energy(self.r_atoms, self.samples, self.z_atoms, self.model)
        return self.e_loc

    # gradients & energy
    def get_energy_of_current_samples(self):
        self.e_loc = self._compute_local_energy(self.r_atoms, self.samples, self.z_atoms, self.model)
        return self.e_loc


    def get_pretrain_grads(self):
        self.pretrain_samples, _, _ = self.model_sampler.sample_mixed(self.pretrain_samples)
        grads = self.pretrainer.compute_grads(self.samples, self.model)
        return grads

    def get_grads(self, e_loc_centered):
        if e_loc_centered is None:
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

    def get_grads_and_maa_and_mss(self, e_loc_centered):
        if e_loc_centered is None:
            e_loc = self.get_energy()
            e_loc_centered = self.center_energy(e_loc)

        grads, m_aa, m_ss, all_a, all_s = self.kfac.extract_grads_and_a_and_s(self.model,
                                                                              self.samples,
                                                                              e_loc_centered,
                                                                              self.n_samples)
        return grads, m_aa, m_ss, all_a, all_s

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

    def set_mxx(self, m_aa, m_ss):
        self.kfac.m_aa = m_aa
        self.kfac.m_ss = m_ss

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
        start = self.gpu_id*self.n_samples
        stop = start + self.n_samples
        self.samples = \
            self._tf.convert_to_tensor(self._load_sample(path)[start:stop, ...])

    def compute_validation_energy(self):
        for _ in range(10):
            self.validation_samples = self.model_sampler.burn(self.validation_samples)
        e_loc = self._compute_local_energy(self.r_atoms, self.validation_samples, self.z_atoms, self.model)
        return e_loc


    def get_amplitudes_of_these_samples(self, samples):
        samples = self._tf.convert_to_tensor(samples, dtype=self._tf.float32)
        amps, sign, _, _, _, _ = self.model(samples)
        return amps, sign


    def confirm_antisymmetric(self, config):
        samples = self._np.random.normal(size=(1, config['n_electrons'], 3))
        # take some amplitudes
        amps, sign = self.get_amplitudes_of_these_samples(samples)

        # swap the electrons up up / up down
        new_idxs = [1, 0, 3, 2]
        samples_upup = samples[:, new_idxs, :]
        amps_upup, sign_upup = self.get_amplitudes_of_these_samples(samples_upup)

        diff = self._tf.reduce_mean(self._tf.abs(amps - amps_upup))
        print('is the wf antsymmetric? ', diff)
        assert diff < 1e-7
        return
