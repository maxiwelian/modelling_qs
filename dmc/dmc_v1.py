import tensorflow as tf

import tensorflow.keras as tk

import tensorflow.keras.layers as layers

import numpy as np

from numpy import pi

import math

dtype = tf.float32
_hydrogen_a0 = tf.convert_to_tensor(1, dtype=dtype)
_pi = tf.convert_to_tensor(math.pi, dtype=dtype)
_hydrogen_norm = tf.cast(1. / (tf.sqrt(_pi) * _hydrogen_a0 ** (3. / 2)), dtype)

@tf.function
def hydrogen_psi(r):
    r = tf.norm(r, axis=-1)
    return _hydrogen_norm * tf.exp(-r / _hydrogen_a0)

@tf.function
def hydrogen_log_psi(r):
    r = tf.norm(r, axis=-1)
    return tf.squeeze(tf.math.log(_hydrogen_norm) - r / _hydrogen_a0), tf.ones(r.shape[0]), 1, 1, 1

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
            self.output_layer = tf.Variable(tf.random.normal((n_determinants,), mean=0.0, stddev=std), dtype=dtype)
    # @tf.function
    def call(self, r_electrons):
        r_atoms = tf.tile(self.r_atoms, (r_electrons.shape[0], 1, 1))
        ae_vectors = compute_ae_vectors(r_atoms, r_electrons)
        # Compute the input vectors
        single_inputs = compute_hydrogen_inputs(ae_vectors, self.n_atoms, self.n_electrons)
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
        return log_psi, sgn, sgn, sgn, sgn

def compute_determinants(inputs):
    output = tf.linalg.slogdet(inputs)
    return output

class envelopeLayer(tk.Model):

    def __init__(self, n_spins, n_atoms, nf_single, n_determinants, std, dtype=tf.float32):
        super(envelopeLayer, self).__init__()
        self.W = tf.Variable(tf.random.normal((n_determinants, nf_single, n_spins), mean=0.0, stddev=std, dtype=dtype))
        self.b = tf.Variable(tf.random.normal((n_determinants, 1, n_spins), mean=0.0, stddev=std, dtype=dtype))
        self.Sigma = tf.Variable(
            tf.random.normal((n_determinants, n_spins, n_atoms, 3, 3), mean=0.0, stddev=std, dtype=dtype))
        self.Pi = tf.Variable(tf.random.normal((n_determinants, n_spins, n_atoms), mean=0.0, stddev=std, dtype=dtype))

    def call(self, inputs, ae_vectors):
        # inputs: (n_samples, n_electrons, nf_single)
        # ae_vectors: (n_samples, n_electrons, n_atoms, 3)
        # print(tf.shape(inputs)
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
    single_inputs = tf.concat((ae_vectors, ae_distances), axis=-1)
    single_inputs = tf.reshape(single_inputs, (-1, n_electrons, 4 * n_atoms))
    return single_inputs

def extract_grads(model, inp, e_loc_centered, n_samples):
    # out (n_samples,)
    # e_loc_centered (n_samples,)
    with tf.GradientTape() as tape:
        out, _, _, _, _ = model(inp)
        out *= e_loc_centered
    grads = tape.gradient(out, model.trainable_weights)
    return [grad / n_samples for grad in grads]

def to_prob(amp):
    return tf.exp(amp) ** 2

def r_gradients(model, samples):
    '''
    :param model:
    :param samples: (n_samples, n_electrons, 3) shape tensor
    :return: dlogphi_dr (n_samples, n_electrons, 3) shape tensor
    The derivatives of the output (logphi the log of the waefunction) wrt r i.e. dlogphi / dr
    '''
    with tf.GradientTape() as g:
        g.watch(samples)
        log_phi, sign, _, _, _ = model(samples)
        # dlogphi_dr = g.gradient(log_phi, samples)
        phi = sign * tf.exp(log_phi)
    dlogphi_dr = g.gradient(phi, samples)
    return dlogphi_dr

if __name__ == '__main__':
    import sys
    sys.path.append("/Users/fwudarsk/NASA/Work/ML_Chemistry/programs/f_wf/")
    import tensorflow as tf
    import numpy as np
    from sampling.sampling import MetropolisHasting, RandomWalker
    from energy.utils import compute_local_energy
    from utils import print_neat

    def log_psi_wrap(psi):
        def _log_psi(x):
            return tf.math.log(psi(x))
        return _log_psi

    dtype = tf.float32
    writer = tf.summary.create_file_writer('runs/hydrogen_supervised')
    ### System
    n_samples = 2048
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
    sample_space = RandomWalker(tf.zeros(3, dtype=dtype),
                                tf.eye(3, dtype=dtype) * 0.5,
                                tf.zeros(3, dtype=dtype),
                                tf.eye(3, dtype=dtype) * 0.8)

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

    pretrainer = ''  # Pretrainer(system, DIR, config['n_pretrain_batches'], n_determinants, n_electrons, n_spin_up, n_spin_down)

    model_sampler = MetropolisHasting(model,
                                      pretrainer,
                                      sample_space,
                                      n_samples,
                                      n_electrons,
                                      correlation_length,
                                      10,
                                      n_atoms,
                                      r_atoms,
                                      [1],
                                      n_spin_up)

    psi_sampler = MetropolisHasting(hydrogen_log_psi,
                                      pretrainer,
                                      sample_space,
                                      n_samples,
                                      n_electrons,
                                      correlation_length,
                                      10,
                                      n_atoms,
                                      r_atoms,
                                      [1],
                                      n_spin_up)

    samples = tf.random.normal((n_samples, n_electrons, 3))

    # Unsupervised training

    for _ in range(1):
        sample_initial_according = model_sampler.initialize_samples()  # argument for the bottom function
        samples, amplitudes, acceptance = model_sampler.sample(samples)
        print('Acceptance percentage: ', acceptance * 100.)

        # Compute the gradients

        e_loc = compute_local_energy(r_atoms, samples, z_atoms, model)  # r_atoms, r_electrons, z_atoms, model
        e_loc_centered = e_loc - tf.reduce_mean(e_loc)
        grads = extract_grads(model, samples, e_loc_centered, n_samples)

        # Update the model
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print_neat('Energy: ', tf.reduce_mean(e_loc), 5)

    n_grads = sum([len(grad) for grad in grads])
    grads = sum([float(tf.reduce_sum(tf.math.abs(grad))) for grad in grads]) / n_grads
    print_neat('Mean abs update: ', tf.reduce_mean(tf.math.abs(grads)), 5)
    distr = RandomWalker(tf.zeros(3, dtype=dtype),
                         tf.eye(3, dtype=dtype) * 0.5,
                         tf.zeros(3, dtype=dtype),
                         tf.eye(3, dtype=dtype) * 0.2)

    curr_sample = samples
    curr_weights = tf.ones(len(samples))


    def green_branching(R_old, R_new, tau):
        # for energy
        e_loc_old = compute_local_energy(r_atoms, R_old, z_atoms, model)
        e_loc_new = compute_local_energy(r_atoms, R_new, z_atoms, model)
        return tf.exp(-1 / 2 * (e_loc_new + e_loc_old) * tau)


    def green_diffusion(R_old, R_new, tau, model):
        # F = (grad psi)/psi  -- force
        F = force(R_new, model)
        number_of_electrons = R_old.shape[1]
        return tf.reshape((2 * pi * tau) ** (-3 * number_of_electrons / 2) * tf.exp(
            -tf.reduce_sum((R_old - R_new - tau * F / 2) ** 2, axis=-1) / (2 * tau)), (-1,))


    def green_total(R_old, R_new, tau, model):
        return green_branching(R_old, R_new, tau) * green_diffusion(R_old, R_new, tau, model)


    def weight(R_old, R_new, tau, model):
        log_phi_old, sign, _, _, _ = model(R_old)  # sample -> R_old
        phi_old = tf.exp(log_phi_old) ** 2
        log_phi_new, sign, _, _, _ = model(R_new)  # sample -> R_old
        phi_new = tf.exp(log_phi_new) ** 2
        G_new_old = green_total(R_new, R_old, tau, model)
        G_old_new = green_total(R_old, R_new, tau, model)
        return (phi_new * G_new_old) / (phi_old * G_old_new)


    def force(R_old, model):
        grad = r_gradients(model, R_old)
        log_phi, sign, _, _, _ = model(R_old)  # sample -> R_old
        # dlogphi_dr = g.gradient(log_phi, samples)
        phi = sign * tf.exp(log_phi)
        F = grad / tf.reshape(phi, (-1, 1, 1))
        return F


    def new_move(R_old, tau, electron, model):
        n_samples, _, _ = R_old.shape
        mask = tf.constant([1 if _ == electron else 0 for _ in range(n_electrons)])
        mask = tf.reshape(mask, (1, -1, 1))
        mask = tf.cast(tf.tile(mask, (n_samples, 1, 3)), dtype=tf.float32)
        random_step = tf.random.normal(R_old.shape, mean=0, stddev=tau)
        R_new = R_old + (tau * force(R_old, model) + random_step) * mask
        return R_new


    def branching_factor(R_old, R_new, tau, E_T):
        e_loc_old = compute_local_energy(r_atoms, R_old, z_atoms, model)
        e_loc_new = compute_local_energy(r_atoms, R_new, z_atoms, model)
        # e_old = tf.reduce_mean(e_loc_old)  # for energy
        # e_new = tf.reduce_mean(e_loc_new)
        return tf.exp(-tau * (1 / 2 * (e_loc_new + e_loc_old) - E_T))


    model = hydrogen_log_psi
    # specify number of samples
    n_blocks = 1000
    n_samples_target = n_samples
    n_samples = n_samples_target
    n_electrons = 1  # for hydrgen

    writer = tf.summary.create_file_writer('runs/new_sample_prob_1')

    e_loc = compute_local_energy(r_atoms, R_new, z_atoms, model)

    with writer.as_default():

        # here we should sample from the preoptimized model, i.e. that displays some features of psi - the trial wave function
        block_size = 10
        E_T = tf.reduce_mean(e_loc)  # trial energy
        tau = 0.01  # time step - what should be agood value for it?
        R_new = tf.random.normal((n_samples, n_electrons, 3))
        # R_old (n_samples, n_electrons, 3)tmp = tf.boolean_mask()
        for block in range(n_blocks):
            for iteration in range(block_size):
                for electron in range(n_electrons):
                    R_old = R_new
                    R_new = new_move(R_old, tau, electron, model)  # (n_samples, n_electrons, 3)
                    log_phi_old, sign_old, _, _, _ = model(R_old)
                    log_phi_new, sign_new, _, _, _ = model(R_new)
                    mask_sign = sign_old == sign_new  # multiply the change by this mask
                    # do not update if sign change
                    W = weight(R_old, R_new, tau, model)
                    prob_acceptance = tf.minimum(tf.ones((n_samples,)), W)
                    p = tf.random.uniform((n_samples,))
                    # accept the change. mask contains accept if true
                    mask_prob = p < prob_acceptance  # (n_samples, )
                    mask_prob = tf.math.logical_and(mask_prob, mask_sign)
                    prob_acc = tf.reduce_mean(tf.cast(mask_prob, tf.float32))
                    mask_prob = tf.reshape(mask_prob, (-1, 1, 1))
                    mask_prob = tf.tile(mask_prob, (1, n_electrons, 3))

                    # change R_old
                    R_new = tf.where(mask_prob, R_new, R_old)
                # are weights given by W or by a branching factor

                # computing branching factor
                Pb = branching_factor(R_old, R_new, tau, E_T)  # (n_samples, )
                n_copies = tf.cast(Pb + tf.random.uniform(Pb.shape), tf.int32)  # + 1  # add 1 for the original
                # n_copies = tf.ones(Pb.shape, dtype=tf.int32)     # for VMC-like
                # print(n_samples)
                max_copies = tf.reduce_min([10, max(n_copies)])
                # print(max_copies)
                mask = tf.constant([i for i in range(max_copies)])

            mask = tf.reshape(mask, (-1, 1, 1, 1))
            mask = tf.tile(mask, (1, n_samples, n_electrons, 3))
            n_copies = tf.reshape(n_copies, (1, -1, 1, 1))
            n_copies = tf.tile(n_copies, (max_copies, 1, n_electrons, 3))
            mask = mask < n_copies
            R_new = tf.reshape(R_new, (1, *R_new.shape))
            R_new = tf.tile(R_new, (max_copies, 1, 1, 1))
            R_new = tf.boolean_mask(R_new, mask)
            R_new = tf.reshape(R_new, (-1, n_electrons, 3))
            # E_T = E_T - 1/tau * tf.math.log(R_new.shape[0]/n_samples_target)

            R_new = tf.random.shuffle(R_new)
            R_new = tf.slice(R_new, [0, 0, 0], [n_samples_target, n_electrons, 3])

            e_loc = compute_local_energy(r_atoms, R_new, z_atoms, model)

            E_T = tf.reduce_mean(e_loc)



            print(E_T)

            r = tf.linalg.norm(R_new, axis=-1)
            r = tf.reduce_mean(r)
            tf.summary.scalar('dmc/energy', E_T, block)
            tf.summary.scalar('dmc/mean dist', r, block)
            tf.summary.scalar('dmc/prob_acceptance', prob_acc, block)

        e_loc = compute_local_energy(r_atoms, R_new, z_atoms, model)
        e_mean = tf.reduce_mean(e_loc)
        if block % 10 == 0:
            print("block = ", block, ", E = ", e_mean)

