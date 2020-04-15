import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

@tf.function
def batched_cdist_l2(x1, x2):
    x1_norm = tf.reduce_sum(x1 ** 2, axis=-1, keepdims=True)
    x2_norm = tf.reduce_sum(x2 ** 2, axis=-1, keepdims=True)
    cdist = tf.sqrt(tf.linalg.matrix_transpose(x2_norm) + x1_norm - 2 * x1 @ tf.linalg.matrix_transpose(x2))
    return cdist


@tf.function
def compute_potential_energy(r_atom, r_electron, z_atom, dtype=tf.float32):
    assert len(r_atom.shape) == 3 and len(r_electron.shape) == 3 and len(z_atom.shape) == 2
    n_samples, n_electron = r_electron.shape[:2]
    n_atom = r_atom.shape[1]

    potential_energy = tf.zeros(n_samples, dtype=dtype)

    if n_electron > 1:
        e_e_dist = batched_cdist_l2(r_electron, r_electron)  # electron - electron distances
        unique_mask = np.tril(np.ones((n_electron, n_electron), dtype=np.bool), -1)
        unique_e_e = tf.boolean_mask(e_e_dist, unique_mask, axis=1)
        potential_energy += tf.reduce_sum(1./unique_e_e, -1)

    a_e_dist = batched_cdist_l2(r_atom, r_electron)  # atom - electron distances
    potential_energy -= tf.einsum('ba,bae->b', z_atom, 1./a_e_dist)

    if n_atom > 1:
        a_a_dist = batched_cdist_l2(r_atom, r_atom)
        weighted_a_a = tf.einsum('bn,bm,bnm->bnm', z_atom, z_atom, 1/a_a_dist)
        unique_a_a = tf.boolean_mask(weighted_a_a, np.tril(np.ones((n_atom, n_atom), dtype=np.bool), -1), axis=1)
        potential_energy += tf.reduce_sum(unique_a_a, -1)

    return potential_energy


@tf.function
def laplacian(model, r_electrons):

    n_electrons = r_electrons.shape[1]
    r_electrons = tf.reshape(r_electrons, (-1, n_electrons*3))
    r_s = [r_electrons[..., i] for i in range(r_electrons.shape[-1])]
    with tf.GradientTape(True) as g:
        [g.watch(r) for r in r_s]
        r_electrons = tf.stack(r_s, -1)
        r_electrons = tf.reshape(r_electrons, (-1, n_electrons, 3))
        with tf.GradientTape(True) as gg:
            gg.watch(r_electrons)
            log_phi, _, _, _, _ = model(r_electrons)
        dlogphi_dr = gg.gradient(log_phi, r_electrons)
        dlogphi_dr = tf.reshape(dlogphi_dr, (-1, n_electrons*3))
        grads = [dlogphi_dr[..., i] for i in range(dlogphi_dr.shape[-1])]
    d2logphi_dr2 = tf.stack([g.gradient(grad, r) for grad, r in zip(grads, r_s)], -1)
    return dlogphi_dr**2, d2logphi_dr2


@tf.function
def compute_local_energy(r_atoms, r_electrons, z_atoms, model):
    n_samples = r_electrons.shape[0]
    r_atoms = tf.tile(r_atoms, (n_samples, 1, 1))
    z_atoms = tf.tile(z_atoms, (n_samples, 1))

    first_order_squared, second_order = laplacian(model, r_electrons)
    potential_energy = compute_potential_energy(r_atoms, r_electrons, z_atoms)

    return -0.5 * (tf.reduce_sum(second_order, -1) + tf.reduce_sum(first_order_squared, -1)) + potential_energy

@tf.function
def clip(x_in):
    median = tfp.stats.percentile(x_in, 50.0)
    total_var = tf.reduce_mean(tf.math.abs(x_in-median))
    clip_min = median - 5*total_var
    clip_max = median + 5*total_var

    x_out = tf.clip_by_value(x_in, clip_min, clip_max)
    return tf.reshape(x_out, x_in.shape)

