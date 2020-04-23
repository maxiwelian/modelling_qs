
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf


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

# sum over all the pairs interacting with that electron
n_samples = 100
n_electrons = 4
n_spin_up = 2
n_spin_down = 2
n_pairwise_features = 20
n_pairwise = n_electrons**2

# q1 how are the streams arranged
r_electrons = tf.random.normal((n_samples, n_electrons, 3))
re1 = tf.expand_dims(r_electrons, 2)
re2 = tf.transpose(re1, perm=(0, 2, 1, 3))

# re1 (n, ne, 1, 3)
# re2 (n, 1, ne, 3)

ee_vectors = re1 - re2
# ee (n, ne, ne`, 3)
# [0, 0] 0 - 0
# [0, 1] 0 - 1

ee_distances = tf.norm(ee_vectors, axis=-1, keepdims=True)

n_single_features = 4
tmp1 = tf.ones((1, n_spin_up, n_single_features), dtype=tf.bool)
tmp2 = tf.zeros((1, n_spin_down, n_single_features), dtype=tf.bool)
spin_up_mask = tf.concat((tmp1, tmp2), 1)
spin_down_mask = ~spin_up_mask

pairwise_spin_up_mask, pairwise_spin_down_mask = generate_pairwise_masks(n_electrons, n_pairwise, n_spin_up, n_spin_down, n_pairwise_features)

single = r_electrons
pairwise = tf.random.normal((n_samples, n_pairwise, n_pairwise_features))

# single (n_samples, n_electrons, n_single_features)
# pairwise (n_samples, n_electrons, n_pairwise_features)
spin_up_mask = tf.tile(spin_up_mask, (n_samples, 1, 1))
spin_down_mask = tf.tile(spin_down_mask, (n_samples, 1, 1))

# --- Single summations
replace = tf.zeros_like(single)
# up
sum_spin_up = tf.where(spin_up_mask, single, replace)
sum_spin_up = tf.reduce_sum(sum_spin_up, 1, keepdims=True) / n_spin_up
sum_spin_up = tf.tile(sum_spin_up, (1, n_electrons, 1))
# down
sum_spin_down = tf.where(spin_down_mask, single, replace)
sum_spin_down = tf.reduce_sum(sum_spin_down, 1, keepdims=True) / n_spin_down
sum_spin_down = tf.tile(sum_spin_down, (1, n_electrons, 1))

# --- Pairwise summations
sum_pairwise = tf.tile(tf.expand_dims(pairwise, 1), (1, n_electrons, 1, 1))
replace = tf.zeros_like(sum_pairwise)
# up
sum_pairwise_up = tf.where(pairwise_spin_up_mask, sum_pairwise, replace)
sum_pairwise_up = tf.reduce_sum(sum_pairwise_up, 2) / (n_spin_up - 1)
# down
sum_pairwise_down = tf.where(pairwise_spin_down_mask, sum_pairwise, replace)
sum_pairwise_down = tf.reduce_sum(sum_pairwise_down, 2) / (n_spin_down - 1)

features = tf.concat((single, sum_spin_up, sum_spin_down, sum_pairwise_up, sum_pairwise_down), 2)