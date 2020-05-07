
import sys
import tensorflow as tf
import numpy as np
from energy.utils import compute_local_energy
from utils import print_neat
import math

dtype = tf.float32
_hydrogen_a0 = tf.convert_to_tensor(1., dtype=dtype)
_pi = tf.convert_to_tensor(math.pi, dtype=dtype)
_hydrogen_norm = tf.cast(1. / (tf.sqrt(_pi) * _hydrogen_a0 ** (3. / 2.)), dtype)

# @tf.function
def hydrogen_psi(r):
    r = tf.linalg.norm(r, axis=-1)
    return _hydrogen_norm * tf.exp(-_hydrogen_a0 * r)


# @tf.function
def hydrogen_log_psi(r):
    r = tf.linalg.norm(r, axis=-1)
    return tf.squeeze(tf.math.log(_hydrogen_norm) - _hydrogen_a0 * r), tf.ones(r.shape[0]), 1, 1, 1

tf.random.set_seed(123)

n_samples = 2

r = tf.random.normal((n_samples, 1, 3))
r_norm = tf.linalg.norm(r, axis=-1)





print('hydrogen a0', _hydrogen_a0)
print('r')
print(r)
print('1/r')
print(1./r)


# r = tf.linalg.norm(r, axis=-1)
# print('r norm', r)

r = [tf.reshape(r[:,0,i], (-1, 1, 1)) for i in range(3)]

with tf.GradientTape(True) as g:
    [g.watch(_) for _ in r]
    print(len(r))
    print(r[0].shape)
    rtmp = tf.concat(r, axis=-1)
    print(rtmp.shape)
    log_psi, _, _, _, _ = hydrogen_log_psi(rtmp)

grad_log = [g.gradient(log_psi, _) for _ in r]
print(grad_log)

with tf.GradientTape() as g:
    g.watch(r)
    log_psi, _, _, _, _ = hydrogen_log_psi(r)
    psi = tf.math.exp(log_psi)
grad_log = g.gradient(psi, r)

print(grad_log.shape)
psi = tf.reshape(psi, (-1, 1))
print('psi shape', psi.shape)
print(grad_log / psi)


with tf.GradientTape() as g:
    g.watch(r)
    psi = hydrogen_psi(r)

grad_psi = g.gradient(psi, r)
grad_psi = grad_psi / tf.reshape(psi, (-1, 1, 1))
#

# print(r / _hydrogen_norm)
# print(grad_psi)
#
# print(1./psi)
#
# print(_hydrogen_norm/r)
#
# print(r / _hydrogen_norm)
#
# print(r / hydrogen_psi(r))






# def r_gradients(model, samples):
#     '''
#     :param model:
#     :param samples: (n_samples, n_electrons, 3) shape tensor
#     :return: dlogphi_dr (n_samples, n_electrons, 3) shape tensor
#     The derivatives of the output (logphi the log of the waefunction) wrt r i.e. dlogphi / dr
#     '''
#     with tf.GradientTape() as g:
#         g.watch(samples)
#         log_phi, sign, _, _, _ = model(samples)
#         # dlogphi_dr = g.gradient(log_phi, samples)
#         phi = sign * tf.exp(log_phi)
#     dlogphi_dr = g.gradient(phi, samples)
#     return dlogphi_dr