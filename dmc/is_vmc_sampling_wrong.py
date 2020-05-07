from numpy import pi
import math
import tensorflow as tf
from sampling.sampling import MetropolisHasting, RandomWalker
from energy.utils import compute_local_energy

dtype = tf.float32
_hydrogen_a0 = tf.convert_to_tensor(1., dtype=dtype)
_pi = tf.convert_to_tensor(math.pi, dtype=dtype)
_hydrogen_norm = tf.cast(1. / (tf.sqrt(_pi) * _hydrogen_a0 ** (3. / 2)), dtype)


# @tf.function
def hydrogen_psi(r):
    r = tf.norm(r, axis=-1)
    return _hydrogen_norm * tf.exp(-r / _hydrogen_a0)


# @tf.function
def hydrogen_log_psi(r):
    r = tf.norm(r, axis=-1)
    return tf.squeeze(tf.math.log(_hydrogen_norm) - r / _hydrogen_a0), tf.ones(r.shape[0]), 1, 1, 1


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

def log_psi_wrap(psi):
    def _log_psi(x):
        return tf.math.log(psi(x))

    return _log_psi

### System
n_samples = 512
n_atoms = 1
n_electrons = 1
n_spin_up = 1
n_spin_down = n_electrons - n_spin_up
nf_single = 4 * n_atoms
n_pairwise = n_electrons ** 2 - n_electrons
nf_pairwise = 4
n_blocks = 1000
n_samples_target = n_samples
n_samples = n_samples_target
n_electrons = 1


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

### Supervised training
correlation_length = 25
system = 'Be'
DIR = ''
config = {}
config['n_pretrain_batches'] = 1

pretrainer = ''

vmc_sampler = MetropolisHasting(hydrogen_log_psi,
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
vmc_samples, _, _ = vmc_sampler.sample(samples)
model = hydrogen_log_psi

R_new = samples
e_loc = compute_local_energy(r_atoms, R_new, z_atoms, model)
E_T = tf.reduce_mean(e_loc)
tau = 0.01

block_size = 10

writer = tf.summary.create_file_writer('runs/11')
with writer.as_default():
    for block in range(n_blocks):

        # vmc energy
        vmc_samples, _, _ = vmc_sampler.sample(vmc_samples)
        vmc_e_loc = compute_local_energy(r_atoms, R_new, z_atoms, model)
        vmc_e_mean = tf.reduce_mean(vmc_e_loc)

        r_vmc_hist = tf.linalg.norm(vmc_samples, axis=-1)
        r_vmc = tf.reduce_mean(r_vmc_hist)
        tf.summary.scalar('vmc/energy', vmc_e_mean, block)
        tf.summary.scalar('vmc/mean dist', r_vmc, block)
        if block % 20 == 0:
            tf.summary.histogram('vmc/hist_samples', tf.reshape(r_vmc_hist, (-1,)), block, buckets=100)

        tf.summary.flush()




writer = tf.summary.create_file_writer('runs/wave')
with writer.as_default():


    # samples = tf.linspace((2500, 1, 3), minval=0., maxval=4.)
    # r = tf.linalg.norm(samples, axis=-1)
    # r = tf.linspace(0., 4., 100)
    # samples = tf.meshgrid(r, r, r)
    # print(len(samples))
    # print(samples[0].shape)
    # new_samples = []
    # for sample in samples:
    #     sample = tf.reshape(sample, (-1, ))
    #     new_samples.append(sample)
    # samples = tf.expand_dims(tf.stack(new_samples), 1)

    samples = tf.random.uniform((100000, 1, 3), minval=0., maxval=3.)
    r = tf.linalg.norm(samples, axis=-1)

    # r = tf.linspace(0., 3., 1000)
    # r = tf.reshape(r, (-1, 1))
    log_psi, _, _, _, _ = model(samples)
    psi = tf.math.exp(log_psi)
    probs = psi**2
    # for sample, prob in zip(samples, probs):
    #     r = tf.linalg.norm(sample, axis=-1)
    #     tf.summary.scalar('wavefunction', prob, r)


# import matplotlib.pyplot as plt
# plt.scatter(r, probs)
# plt.xlim((0, 4.))
# plt.show()
