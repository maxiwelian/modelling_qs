from numpy import pi
import math
import tensorflow as tf
from sampling.sampling import MetropolisHasting, RandomWalker
from energy.utils import compute_local_energy

dtype = tf.float32
_hydrogen_a0 = tf.convert_to_tensor(1.5, dtype=dtype)
_pi = tf.convert_to_tensor(math.pi, dtype=dtype)
_hydrogen_norm = tf.cast(1. / (tf.sqrt(_pi) * _hydrogen_a0 ** (3. / 2.)), dtype)


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


def green_branching(R_new, R_old, tau, E_T):
    # for energy
    e_loc_old = compute_local_energy(r_atoms, R_old, z_atoms, model)
    e_loc_new = compute_local_energy(r_atoms, R_new, z_atoms, model)
    return tf.exp(((-0.5*(e_loc_new + e_loc_old)) - E_T) * tau)


def green_diffusion(R_new, R_old, tau, model):
    # F = (grad psi)/psi  -- force
    n_electrons = R_old.shape[1]
    F = force(R_new, model)
    c = (2.*pi*tau) ** (-3.*n_electrons/2.)
    exp = tf.exp(-tf.linalg.norm((R_old-R_new - tau*F), axis=-1)**2 / (2*tau))
    return tf.squeeze(c * exp)


def green_total(R_new, R_old, tau, model, E_T, block):
    # return tf.reshape(green_branching(R_new, R_old,  tau, E_T), (-1, 1)) * green_diffusion(R_new, R_old,  tau, model)
    branch = green_branching(R_new, R_old,  tau, E_T)
    diffusion = green_diffusion(R_new, R_old,  tau, model)
    tf.summary.scalar('greens/branch', tf.reduce_mean(branch), block)
    tf.summary.scalar('greens/diffusion', tf.reduce_mean(diffusion), block)
    return  branch * diffusion


def weight(R_new, R_old, tau, model, E_T, block):
    log_phi_old, sign, _, _, _ = model(R_old)  # sample -> R_old
    # phi_old = tf.expand_dims(tf.exp(log_phi_old) ** 2, -1)
    phi_old = tf.exp(log_phi_old) ** 2
    log_phi_new, sign, _, _, _ = model(R_new)  # sample -> R_old
    # phi_new = tf.expand_dims(tf.exp(log_phi_new) ** 2, -1)
    phi_new = tf.exp(log_phi_new) ** 2
    G_new_old = green_total(R_new, R_old, tau, model, E_T, block)

    G_old_new = green_total(R_old, R_new, tau, model, E_T, block)

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
    random_step = tf.random.normal(R_old.shape, mean=0., stddev=tau**.5)
    F = force(R_old, model)
    R_new = R_old + (tau * F + random_step) * mask
    return R_new


def branching_factor(R_new, R_old, tau, E_T):
    e_loc_old = compute_local_energy(r_atoms, R_old, z_atoms, model)
    e_loc_new = compute_local_energy(r_atoms, R_new, z_atoms, model)
    # e_old = tf.reduce_mean(e_loc_old)  # for energy
    # e_new = tf.reduce_mean(e_loc_new)
    return tf.exp(-tau * (1./2.*(e_loc_new + e_loc_old) - E_T))

def log_psi_wrap(psi):
    def _log_psi(x):
        return tf.math.log(psi(x))

    return _log_psi

### System
n_samples = 2048
n_atoms = 1
n_electrons = 1
n_spin_up = 1
n_spin_down = n_electrons - n_spin_up
nf_single = 4 * n_atoms
n_pairwise = n_electrons ** 2 - n_electrons
nf_pairwise = 4
n_blocks = 10000
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

n_determinants = 1


writer = tf.summary.create_file_writer('runs/wave')
with writer.as_default():

    model = hydrogen_log_psi
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

R_new = samples
e_loc = compute_local_energy(r_atoms, R_new, z_atoms, model)
E_T = tf.reduce_mean(e_loc)
tau = 0.01

block_size = 10

branch = True
writer = tf.summary.create_file_writer('runs/dmc_test3')
with writer.as_default():
    for block in range(n_blocks):
        for iteration in range(block_size):
            for electron in range(n_electrons):
                R_old = R_new
                R_new = new_move(R_old, tau, electron, model)  # (n_samples, n_electrons, 3)
                log_phi_old, sign_old, _, _, _ = model(R_old)
                log_phi_new, sign_new, _, _, _ = model(R_new)
                # mask_sign = tf.expand_dims(sign_old == sign_new, -1)  # multiply the change by this mask
                mask_sign = sign_old == sign_new  # multiply the change by this mask
                # do not update if sign change
                W = weight(R_new, R_old, tau, model, E_T, block)

                # prob_acceptance = tf.minimum(tf.ones((R_new.shape[0], 3)), W)
                prob_acceptance = tf.minimum(tf.ones((R_new.shape[0],), dtype=tf.float32), W)

                # p = tf.random.uniform((R_new.shape[0], 3))
                p = tf.random.uniform((R_new.shape[0],))

                # accept the change. mask contains accept if true
                mask_prob = p < prob_acceptance  # (n_samples, )
                mask_prob = tf.math.logical_and(mask_prob, mask_sign)
                tmp = mask_prob.numpy()
                prob_acc = tf.reduce_mean(tf.cast(mask_prob, tf.float32))
                mask_prob = tf.reshape(mask_prob, (-1, 1, 1))

                # mask_prob = tf.tile(mask_prob, (1, n_electrons, 1))
                mask_prob = tf.tile(mask_prob, (1, n_electrons, 3))

                # change R_old
                R_new = tf.where(mask_prob, R_new, R_old)
            # are weights given by W or by a branching factor

            if branch:
                # computing branching factor
                Pb = branching_factor(R_new, R_old, tau, E_T)  # (n_samples, )
                n_copies = tf.cast(Pb + tf.random.uniform(Pb.shape), tf.int32)  # + 1  # add 1 for the original
                # n_copies = tf.ones(Pb.shape, dtype=tf.int32)     # for VMC-like
                # print(n_samples)
                max_copies = tf.reduce_min([10, max(n_copies)])
                # print(max_copies)
                mask = tf.constant([i for i in range(max_copies)])

        if branch:
            # print('copying')
            tmp = []
            for n_copy, R in zip(n_copies, R_new):
                sample = tf.tile(tf.expand_dims(R, 0), (n_copy+1, 1, 1))
                tmp.append(sample)
            R_new = tf.concat(tmp, axis=0)

            R_new = tf.random.shuffle(R_new)
            R_new = tf.slice(R_new, [0, 0, 0], [n_samples_target, n_electrons, 3])

        # dmc energy
        e_loc = compute_local_energy(r_atoms, R_new, z_atoms, model)
        E_T = tf.reduce_mean(e_loc)

        # vmc energy
        vmc_samples, _, _ = vmc_sampler.sample(vmc_samples)
        vmc_e_loc = compute_local_energy(r_atoms, vmc_samples, z_atoms, model)
        vmc_e_mean = tf.reduce_mean(vmc_e_loc)

        # logging
        r_dmc_samples = tf.linalg.norm(R_new, axis=-1)
        r_dmc = tf.reduce_mean(r_dmc_samples)
        tf.summary.scalar('dmc/energy', E_T, block)
        tf.summary.scalar('dmc/mean dist', r_dmc, block)
        tf.summary.scalar('dmc/prob_acceptance', prob_acc, block)

        r_vmc_hist = tf.linalg.norm(vmc_samples, axis=-1)
        r_vmc = tf.reduce_mean(r_vmc_hist)
        tf.summary.scalar('vmc/energy', vmc_e_mean, block)
        tf.summary.scalar('vmc/mean dist', r_vmc, block)
        tf.summary.histogram('vmc/hist_samples', tf.reshape(r_vmc_hist, (-1,)), block, buckets=100)

        if branch:
            tf.summary.scalar('dmc/sum_n_copies', tf.reduce_sum(n_copies), block)

        e_loc = compute_local_energy(r_atoms, R_new, z_atoms, model)
        e_mean = tf.reduce_mean(e_loc)

        if block % 10 == 0:
            print("block = ", block, ", E = ", e_mean)

        if block % 20 == 0:
            tf.summary.histogram('dmc/r_samples', r_dmc_samples, block, buckets=100)

