import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np

def generate_pairwise_masks(n_electrons, n_pairwise, n_spin_up, n_spin_down, n_pairwise_features):
    ups = np.ones(n_electrons, dtype=np.bool)
    ups[n_spin_up:] = False
    downs = ~ups

    spin_up_mask = []
    spin_down_mask = []
    mask = np.zeros((n_electrons, n_electrons), dtype=np.bool)

    for electron in range(n_electrons):
        mask_up = np.copy(mask)
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

def compare_tensors(t1, t2):
    return tf.reduce_mean(tf.abs(tf.cast(t1, tf.float32) - tf.cast(t2, tf.float32)))

n_samples = 100
n_electrons = 4
nf = 20
n_pairwise = n_electrons**2
n_spin_up = 2

pairwise_spin_up_mask, _ = generate_pairwise_masks(n_electrons, n_pairwise, 2, 2, nf)
sum_pairwise = tf.random.normal((n_samples, n_electrons**2, 4))
replace = tf.zeros_like(sum_pairwise)
sum_pairwise_up = tf.where(pairwise_spin_up_mask, sum_pairwise, replace)
sum_pairwise_up = tf.reduce_sum(sum_pairwise_up, 2) / n_spin_up