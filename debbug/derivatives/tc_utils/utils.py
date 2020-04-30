import numpy as np
import tensorflow as tf
import torch as tc

def tc_generate_pairwise_masks_full(n_electrons, n_pairwise, n_spin_up, n_spin_down, n_pairwise_features):
    ups = np.ones(n_electrons, dtype=np.bool)
    ups[n_spin_up:] = False
    downs = ~ups

    spin_up_mask = []
    spin_down_mask = []
    mask = np.zeros((n_electrons, n_electrons), dtype=np.bool)

    for electron in range(n_electrons):
        mask_up = np.copy(mask)
        mask_up[electron, :] = ups
        spin_up_mask.append(mask_up.reshape(-1))

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        spin_down_mask.append(mask_down.reshape(-1))

    spin_up_mask = tc.tensor(spin_up_mask, dtype=tc.bool)
    # (n_samples, n_electrons, n_electrons, n_pairwise_features)
    spin_up_mask = spin_up_mask.view((1, n_electrons, n_pairwise, 1))
    spin_up_mask = spin_up_mask.repeat((1, 1, 1, n_pairwise_features))

    spin_down_mask = tc.tensor(spin_down_mask, dtype=tc.bool)
    spin_down_mask = spin_down_mask.view((1, n_electrons, n_pairwise, 1))
    spin_down_mask = spin_down_mask.repeat((1, 1, 1, n_pairwise_features))

    return spin_up_mask, spin_down_mask

def tc_generate_pairwise_masks(n_electrons, n_pairwise, n_spin_up, n_spin_down, n_pairwise_features):
    eye_mask = ~np.eye(n_electrons, dtype=np.bool)
    ups = np.ones(n_electrons, dtype=np.bool)
    ups[n_spin_up:] = False
    downs = ~ups

    spin_up_mask = []
    spin_down_mask = []
    mask = np.zeros((n_electrons, n_electrons), dtype=np.bool)

    for electron in range(n_electrons):
        mask_up = np.copy(mask)
        mask_up[electron, :] = ups
        mask_up = mask_up[eye_mask]
        spin_up_mask.append(mask_up.reshape(-1))

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        mask_down = mask_down[eye_mask]
        spin_down_mask.append(mask_down.reshape(-1))

    spin_up_mask = tf.convert_to_tensor(spin_up_mask, dtype=tf.bool)
    # (n_samples, n_electrons, n_electrons, n_pairwise_features)
    spin_up_mask = tf.reshape(spin_up_mask, (1, n_electrons, n_pairwise, 1))
    spin_up_mask = tf.tile(spin_up_mask, (1, 1, 1, n_pairwise_features))

    spin_down_mask = tf.convert_to_tensor(spin_down_mask, dtype=tf.bool)
    spin_down_mask = tf.reshape(spin_down_mask, (1, n_electrons, n_pairwise, 1))
    spin_down_mask = tf.tile(spin_down_mask, (1, 1, 1, n_pairwise_features))

    return spin_up_mask, spin_down_mask

