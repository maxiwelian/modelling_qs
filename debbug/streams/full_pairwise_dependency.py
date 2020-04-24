
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import numpy as np

n_samples = 10
n_electrons = 4

def generate_pairwise_masks(n_electrons, n_pairwise, n_spin_up, n_spin_down, n_pairwise_features):
    ups = np.ones(n_electrons, dtype=np.bool)
    ups[n_spin_up:] = False
    downs = ~ups

    spin_up_mask = []
    spin_down_mask = []

    mask = np.zeros((n_electrons, n_electrons), dtype=np.bool)

    for electron in range(n_electrons):
        mask_up = np.copy(mask)  # each of these indicates how to mask out the pairwise terms
        mask_up[electron, :] = ups
        spin_up_mask.append(mask_up)

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        spin_down_mask.append(mask_down)

    spin_up_mask = tf.convert_to_tensor(spin_up_mask, dtype=tf.bool)
    # (n_samples, n_electrons, n_electrons, n_pairwise_features)
    spin_up_mask = tf.reshape(spin_up_mask, (1, n_electrons, n_pairwise, 1))
    spin_up_mask = tf.tile(spin_up_mask, (1, 1, 1, n_pairwise_features))

    spin_down_mask = tf.convert_to_tensor(spin_down_mask, dtype=tf.bool)
    spin_down_mask = tf.reshape(spin_down_mask, (1, n_electrons, n_pairwise, 1))
    spin_down_mask = tf.tile(spin_down_mask, (1, 1, 1, n_pairwise_features))

    return spin_up_mask, spin_down_mask

w = tf.random.normal((4, 10))

mask, _ = generate_pairwise_masks(n_electrons, n_electrons**2, 2, 2, 4)

re = tf.random.normal((n_samples, n_electrons, 3))

# grads working
with tf.GradientTape() as g:
    g.watch(re)
    re1 = tf.expand_dims(re, 2)
    re2 = tf.transpose(re1, perm=(0, 2, 1, 3))
    ee_vec = re1 - re2
    ee_vec = tf.reshape(ee_vec, (-1, n_electrons ** 2, 3))
    ee_dist = tf.norm(ee_vec, keepdims=True, axis=-1)

    pairwise = tf.concat((ee_vec, ee_dist), -1)

    y = pairwise @ w

z = g.gradient(y, pairwise)

print(z)

with tf.GradientTape() as g:
    g.watch(re)
    re1 = tf.expand_dims(re, 2)
    re2 = tf.transpose(re1, perm=(0, 2, 1, 3))
    ee_vec = re1 - re2
    ee_vec = tf.reshape(ee_vec, (-1, n_electrons ** 2, 3))
    ee_dist = tf.norm(ee_vec, keepdims=True, axis=-1)

    pairwise = tf.concat((ee_vec, ee_dist), -1)

    # ops
    sum_pairwise = tf.tile(tf.expand_dims(pairwise, 1), (1, n_electrons, 1, 1))
    replace = tf.zeros_like(sum_pairwise)
    # up
    sum_pairwise_up = tf.where(mask, sum_pairwise, replace)
    e_masked = tf.reduce_sum(sum_pairwise_up, 2) / 2

    y = e_masked @ w

z = g.gradient(y, e_masked)

print(z)
