import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as layers


class hyNet(tk.Model):
    def __init__(self, n_samples, n_atoms, r_atoms, n_electrons, nf_single_in,
                 nf_hidden_single=256, n_determinants=1, std=0.01, dtype=tf.float32):

        super(hyNet, self).__init__()
        self.r_atoms = r_atoms
        self.n_electrons = n_electrons
        self.n_atoms = n_atoms
        self.n_samples = n_samples
        self.n_determinants = n_determinants

        self.single_stream1 = singleStream(nf_single_in, nf_hidden_single, dtype=dtype)
        self.single_stream2 = singleStream(nf_hidden_single, nf_hidden_single)

        self.envelope = envelopeLayer(n_electrons, n_atoms, nf_hidden_single, n_determinants, std, dtype=dtype)

        if n_determinants > 1:
            self.output_layer = tf.Variable(tf.random.normal((n_determinants,), mean=0.0,stddev=std),dtype=dtype)

    # @tf.function
    def call(self, r_electrons):
        r_atoms = tf.tile(self.r_atoms, (r_electrons.shape[0], 1, 1))

        ae_vectors = compute_ae_vectors(r_atoms, r_electrons)

        # Compute the input vectors
        single_inputs = compute_hydrogen_inputs(ae_vectors, self.n_atoms,self.n_electrons)

        # Layer 1
        single_outputs = self.single_stream1(single_inputs)
        single_outputs = self.single_stream2(single_outputs)

        # Envelopes
        psi = self.envelope(single_outputs, ae_vectors)

        psi = tf.squeeze(psi)

        if self.n_determinants > 1:
            psi = tf.reduce_sum(psi * self.output_layer, axis=1)

        sgn = tf.sign(psi)
        log_psi = tf.math.log(tf.math.abs(psi))
        return log_psi, sgn, sgn, sgn, sgn, sgn  # needed because of tech debt with main model

def compute_determinants(inputs):
    output = tf.linalg.slogdet(inputs)
    return output


class envelopeLayer(tk.Model):
    def __init__(self, n_spins, n_atoms, nf_single, n_determinants, std, dtype=tf.float32):
        super(envelopeLayer, self).__init__()
        self.W = tf.Variable(tf.random.normal((n_determinants, nf_single, n_spins), mean=0.0, stddev=std, dtype=dtype))
        self.b = tf.Variable(tf.random.normal((n_determinants, 1, n_spins), mean=0.0, stddev=std, dtype=dtype))

        self.Sigma = tf.Variable(tf.random.normal((n_determinants, n_spins, n_atoms, 3, 3), mean=0.0, stddev=std, dtype=dtype))
        self.Pi = tf.Variable(tf.random.normal((n_determinants, n_spins, n_atoms), mean=0.0, stddev=std, dtype=dtype))

    def call(self, inputs, ae_vectors):
        # inputs: (n_samples, n_electrons, nf_single)
        # ae_vectors: (n_samples, n_electrons, n_atoms, 3)
        # print(tf.shape(inputs))
        # n: n_samples, e: n_electrons, f: nf_single, i: n_electrons, k: n_determinants
        factor = tf.einsum('njf,kfi->nkij', inputs, self.W)
        factor = factor + self.b

        # k: n_determinants, i: n_electrons, m: n_atoms, n: n_samples, j: n_electrons
        # print(tf.shape(ae_vectors), tf.shape(self.Sigma))
        exponential = tf.einsum('kimvc,njmv->nkijmc', self.Sigma, ae_vectors)
        exponential = tf.exp(-tf.norm(exponential, axis=-1))

        # print(tf.shape(self.Pi), tf.shape(exponential))
        envelope_sum = tf.einsum('kim,nkijm->nkij', self.Pi, exponential)

        output = factor * envelope_sum
        return output


class singleStream(tk.Model):
    def __init__(self, in_dim, out_dim, dtype=tf.float32):
        super(singleStream, self).__init__()
        self.layer = layers.Dense(out_dim, input_shape=(in_dim,), activation='tanh', dtype=dtype)

    def call(self, inputs):
        output = self.layer(inputs)
        return output


def compute_ae_vectors(r_atoms, r_electrons):
    r_atoms = tf.expand_dims(r_atoms, 1)
    r_electrons = tf.expand_dims(r_electrons, 2)
    ae_vectors = r_electrons - r_atoms
    return ae_vectors


def compute_hydrogen_inputs(ae_vectors, n_atoms, n_electrons):
    # r_atoms: (n_atoms, 3)
    # r_electrons: (n_samples, n_electrons, 3)
    # ae_vectors: (n_samples, n_electrons, n_atoms, 3)
    ae_distances = tf.norm(ae_vectors, axis=-1, keepdims=True)
    single_inputs = tf.concat((ae_vectors, ae_distances),axis=-1)
    single_inputs = tf.reshape(single_inputs, (-1, n_electrons, 4*n_atoms))
    return single_inputs

def extract_grads(model, inp, e_loc_centered, n_samples):
    # out (n_samples,)
    # e_loc_centered (n_samples,)
    with tf.GradientTape() as tape:
        out, _, _, _, _, _ = model(inp)
        out *= e_loc_centered
    grads = tape.gradient(out, model.trainable_weights)
    return [grad / n_samples for grad in grads]

def to_prob(amp):
    return tf.exp(amp) ** 2


def r_gradients(model, samples):

    with tf.GradientTape() as g:
        g.watch(samples)
        log_phi, sign, _, _, _, _ = model(samples)

    dlogphi_dr = g.gradient(log_phi, samples)
    return dlogphi_dr

import tensorflow as tf
import math
dtype = tf.float32

_hydrogen_a0 = tf.convert_to_tensor(1, dtype=dtype)
_pi = tf.convert_to_tensor(math.pi, dtype=dtype)
_hydrogen_norm = tf.cast(1 / (tf.sqrt(_pi) * _hydrogen_a0 ** (3./2)), dtype)

@tf.function
def hydrogen_psi(r):
    r = tf.norm(r, axis=-1)
    return _hydrogen_norm * tf.exp(-r/_hydrogen_a0)

@tf.function
def hydrogen_log_psi(r):
    r = tf.norm(r, axis=-1)
    return tf.math.log(_hydrogen_norm) - r/_hydrogen_a0, 1, 1

@tf.function
def helium(r):

    exp = tf.exp(tf.reduce_mean(-tf.norm(r, axis=-1), axis=1))
    # exp = sf * tf.math.reduce_prod(exp, 1)
    return tf.squeeze(tf.math.log(exp))


if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    from sampling.sampling import MetropolisHasting, RandomWalker
    from energy.energy import compute_local_energy
    from utils.utils import print_neat


    def log_psi_wrap(psi):
        def _log_psi(x):
            return tf.math.log(psi(x))

        return _log_psi


    dtype = tf.float32

    writer = tf.summary.create_file_writer('runs/hydrogen_supervised')

    ### System
    n_training_iterations = 10
    n_samples = 512
    n_atoms = 1
    n_electrons = 1
    n_spin_up = 1

    n_spin_down = n_electrons - n_spin_up
    nf_single = 4 * n_atoms
    n_pairwise = n_electrons ** 2 - n_electrons
    nf_pairwise = 4

    r_atoms = tf.zeros((1, n_atoms, 3), dtype=dtype)
    z_atoms = tf.ones((1, n_atoms))

    r_electrons = tf.random.normal((n_samples, n_electrons, 3), dtype=dtype)

    ### Sampler
    # Inputs are two tensors:
    # (n_samples, n_electrons, n_single_features, n_dim_single_features),
    # (n_samples, n_pairwise, n_pairwise_features, n_dim_pairwise_features)
    sample_space = RandomWalker(0, 0.0, 0.02)

    n_determinants = 1
    ### Model
    model = hyNet(n_samples, n_atoms, r_atoms, n_electrons, nf_single,
                  nf_hidden_single=4, n_determinants=n_determinants, std=0.1, dtype=tf.float32)

    n_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])

    ### Optimizer
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.05,
        decay_steps=100,
        decay_rate=0.5
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    ### Supervised training
    correlation_length = 25

    system = 'Be'
    DIR = ''
    config = {}
    config['n_pretrain_batches'] = 1

    pretrainer = ''#Pretrainer(system, DIR, config['n_pretrain_batches'], n_determinants, n_electrons, n_spin_up, n_spin_down)
    atom_positions = [[0.0, 0.0, 0.0]]
    ne_atoms = [1]

    model_sampler = MetropolisHasting(model,
                                      pretrainer,
                                      sample_space,
                                      0,
                                      n_samples,
                                      n_electrons,
                                      correlation_length,
                                      10,
                                      n_atoms,
                                      atom_positions,
                                      ne_atoms,
                                      n_spin_up)

    psi_sampler = MetropolisHasting(hydrogen_log_psi,
                                     pretrainer,
                                     sample_space,
                                     0,
                                     n_samples,
                                     n_electrons,
                                     correlation_length,
                                     10,
                                     n_atoms,
                                     atom_positions,
                                     ne_atoms,
                                     n_spin_up)

    samples = tf.random.normal((n_samples, 1, 3))
    # Unsupervised training
    for _ in range(n_training_iterations):
        samples, amplitudes, acceptance = model_sampler.sample(samples)
        print('Acceptance percentage: ', acceptance * 100.)
        # Compute the gradients
        e_loc = compute_local_energy(r_atoms, samples, z_atoms, model) #r_atoms, r_electrons, z_atoms, model
        e_loc_centered = e_loc - tf.reduce_mean(e_loc)

        grads = extract_grads(model, samples, e_loc_centered, n_samples)

        # Update the model
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        print_neat('Energy: ', tf.reduce_mean(e_loc), 5)
        n_grads = sum([len(grad) for grad in grads])

        grads = sum([float(tf.reduce_sum(tf.math.abs(grad))) for grad in grads]) / n_grads
        print_neat('Mean abs update: ', tf.reduce_mean(tf.math.abs(grads)), 5)


    distr = RandomWalker(tf.zeros(3, dtype=dtype),
                            tf.eye(3, dtype=dtype) * 2.,
                            tf.zeros(3, dtype=dtype),
                            tf.eye(3, dtype=dtype) * 0.8)

    curr_sample = samples
    curr_weights = tf.ones(len(samples))

    print('starting the DMC')
    for _ in range(100):
        e_loc = compute_local_energy(r_atoms, samples, z_atoms, model)  # r_atoms, r_electrons, z_atoms, model
        e_loc_centered = e_loc - tf.reduce_mean(e_loc)
        # curr_grad = extract_grads(model, curr_sample, e_loc_centered, n_samples)
        curr_grad = r_gradients(model, samples)
        # function for L wrt r

        # intermediate
        intermediate_sample = curr_sample + curr_grad

        # next sample
        new_sample = distr.resample(intermediate_sample)
        new_log_amp, _, _, _, _, _ = model(new_sample)
        new_prob = to_prob(new_log_amp)

        e_loc, _ = compute_local_energy(r_atoms, samples, z_atoms, model)  # r_atoms, r_electrons, z_atoms, model
        e_loc_centered = e_loc - tf.reduce_mean(e_loc)
        new_sample_grad = extract_grads(model, new_sample, e_loc_centered, n_samples)



        e_new =e_loc * curr_weights

        # weights update



### Check energy hydrogen
        # n_samples_check = 4000
        # samples, amps, acceptance = psi_sampler.sample(n_samples_check, n_electrons, correlation_length)
        # print_neat('psi acceptance', acceptance*100, prec=3)
        # psi_energy = tf.reduce_mean(compute_local_energy(r_atoms, samples, z_atoms, log_psi_wrap(hydrogen_psi)))
        # print_neat('Supervised learning energy psi: ', psi_energy, prec=4)

        # samples, amps, acceptance = psi_sampler.sample(n_samples, n_electrons, correlation_length)
        # print_neat('psi acceptance', acceptance*100, prec=3)
        # psi_energy = tf.reduce_mean(compute_local_energy(r_atoms, samples, z_atoms, hydrogen_log_psi))
        # print_neat('Supervised learning energy psi: ', psi_energy, prec=4)
        #
        # log_psi = tf.reshape(hydrogen_log_psi(samples), (-1,))
        #
        # n_epochs = 500
        # printer = {}
        # for epoch in range(n_epochs):
        #
        #     with tf.GradientTape() as t:
        #         log_model = model(samples)
        #         loss = tf.keras.losses.MSE(log_psi, log_model)
        #     grads = t.gradient(loss, model.trainable_weights)
        #
        #     mean_grads = sum([float(tf.reduce_sum(tf.math.abs(grad))) for grad in grads]) / n_params
        #     optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #
        #     samples_model, amps, acceptance = model_sampler.sample(n_samples, n_electrons, correlation_length)
        #     model_energy = tf.reduce_mean(compute_local_energy(r_atoms, samples_model, z_atoms, model))
        #
        #     printer['loss'] = [loss, 3]
        #     printer['mean_grads:'] = [mean_grads, 4]
        #     printer['epoch'] = epoch
        #     printer['energy_model'] = model_energy
        #
        #     if epoch % 20 == 0:
        #         with writer.as_default():
        #             bounds = (tf.reduce_min(samples[..., 0]), tf.reduce_max(samples[..., 0]), tf.reduce_min(samples[..., 1]), tf.reduce_max(samples[..., 1]))
        #             tf.summary.image('wavefunction/psi', plot_wavefunction(hydrogen_log_psi, bounds, 100), 0)
        #             tf.summary.image('wavefunction/model', plot_wavefunction(model, bounds, 100), 0)
        #             tf.summary.flush()



