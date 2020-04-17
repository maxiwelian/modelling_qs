import ray
import tensorflow as tf
from energy.energy_utils import clip


@ray.remote(num_gpus=1)
class Network(object):
    def __init__(self, config, gpu_id):
        import tensorflow as tf
        import numpy as np
        from time import time

        from sampling.sampling import MetropolisHasting, RandomWalker
        from model.fermi_net import fermiNet
        from energy.energy import compute_local_energy
        from model.gradients import extract_grads, KFAC_Actor
        from pretraining.pretraining import Pretrainer
        from utils.utils import load_model, load_sample, filter_dict, tofloat

        if config['seed']:
            tf.random.set_seed(7)
            np.random.seed(7)

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
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=config['lr0'], decay=config['decay'])
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
        e_loc_centered = center_energy(e_loc)
        grads = self._extract_grads(self.model,
                                    self.samples,
                                    e_loc_centered,
                                    self.n_samples)
        return grads

    def get_grads_and_maa_and_mss(self):
        e_loc = self.get_energy()
        e_loc_centered = center_energy(e_loc)
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
        updates[-1] = tf.reshape(updates[-1], self.trainable_shapes[-1])
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
        return (self.n_params, self.n_layers, self.trainable_shapes, self.layers)

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


# The functions below are proxies for the actor

# sampling
def burn(models, n_burns):
    for _ in range(n_burns):
        [model.burn.remote() for model in models]


def burn_pretrain(models, n_burns):
    for _ in range(n_burns):
        [model.burn_pretrain.remote() for model in models]


# pretraining
def get_pretrain_grads(models):
    grads = ray.get([model.get_pretrain_grads.remote() for model in models])
    new_grads = []
    for i in range(len(grads[0])):
        grad = tf.stack([grad[i] for grad in grads])
        grad = tf.reduce_sum(grad, axis=0)
        new_grads.append(grad)
    return new_grads


def update_weights_pretrain(models, grads):
    grad_id = ray.put(grads)
    models[0].update_weights_pretrain.remote(grad_id)
    weights_id = models[0].get_weights.remote()
    [model.set_weights.remote(weights_id) for model in models[1:]]


# training
def get_grads(models):
    grads = ray.get([model.get_grads.remote() for model in models])
    new_grads = []
    for layer_id in range(len(grads[0])):
        grads_layer = tf.reduce_mean(tf.stack([grad[layer_id] for grad in grads]), axis=0)
        new_grads.append(grads_layer)
    return new_grads


def update_weights_optimizer(models, grads):
    grad_id = ray.put(grads)
    models[0].update_weights.remote(grad_id)
    weights_id = models[0].get_weights.remote()
    [model.set_weights.remote(weights_id) for model in models[1:]]
    return


# kfac
def get_grads_and_maa_and_mss(models, layers):
    data = ray.get([model.get_grads_and_maa_and_mss.remote() for model in models])
    grads = [d[0] for d in data]
    m_aa = [d[1] for d in data]
    m_ss = [d[2] for d in data]

    mean_g = []
    mean_m_aa = {}
    mean_m_ss = {}
    for j, name in enumerate(layers):
        mean_g.append(tf.reduce_mean(tf.stack([g[j] for g in grads]), axis=0))
        maa = (tf.reduce_mean(tf.stack([ma[name] for ma in m_aa]), axis=0))
        mss = (tf.reduce_mean(tf.stack([ms[name] for ms in m_ss]), axis=0))
        # enforce symmetry (there may be numerical errors)
        maa = (tf.linalg.matrix_transpose(maa) + maa) / 2.
        mss = (tf.linalg.matrix_transpose(mss) + mss) / 2.
        mean_m_aa[name] = maa
        mean_m_ss[name] = mss

    return mean_g, mean_m_aa, mean_m_ss


def step_forward(models, updates):
    updates_id = ray.put(updates)
    models[0].step_forward.remote(updates_id)
    weights_id = models[0].get_weights.remote()
    [model.set_weights.remote(weights_id) for model in models[1:]]
    return


# energy
def get_energy(models):
    e_locs = ray.get([model.get_energy.remote() for model in models])
    e_loc = tf.concat(e_locs, axis=0)
    e_mean = tf.reduce_mean(e_loc)
    e_std = tf.math.reduce_std(e_loc)
    return e_loc, e_mean, e_std


def get_energy_and_center(models, iteration):
    e_loc, e_mean, e_std = get_energy(models)
    e_loc_clipped = clip(e_loc, iteration)
    e_loc_centered = e_loc_clipped - tf.reduce_mean(e_loc_clipped)
    return e_loc_centered, e_mean, e_loc, e_std


def center_energy(e_loc):
    e_loc_clipped = clip(e_loc)
    e_loc_centered = e_loc_clipped - tf.reduce_mean(e_loc_clipped)
    return e_loc_centered


# utils
def get_info(models):
    info = ray.get([model.get_info.remote() for model in models])
    amplitudes = [lst[0] for lst in info]
    acceptance = tf.reduce_mean([lst[1] for lst in info])
    samples = [lst[2] for lst in info]
    e_loc = tf.concat([lst[3] for lst in info], axis=0)
    return amplitudes, acceptance, samples, e_loc
