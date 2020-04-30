
import tensorflow as tf
import torch as tc
import numpy as np
import os
import tensorflow.keras as tk

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from torch.autograd import grad as tcgrad
# DTYPE = tf.float64

# @tf.function



# model architecture
full_pairwise = True
n_atoms = 1
n_electrons = 4
n_spin_up = 2
n_spin_down = n_electrons - n_spin_up
n_pairwise = n_electrons**2
if not full_pairwise:
    n_pairwise -= n_electrons

nf_single_in = 4 * n_atoms
nf_hidden_single = 128
nf_pairwise_in = 4
nf_hidden_pairwise = 16
nf_intermediate_single = 3*nf_hidden_single + 2*nf_hidden_pairwise

n_determinants = 8
