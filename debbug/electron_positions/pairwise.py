
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''


import tensorflow as tf
from time import time



n_electrons = 4
n_samples = 100
r_electrons = tf.random.normal((n_samples, n_electrons, 3))

n = 1000
t0 = time()
for _ in range(n):
    r_electrons_1 = tf.expand_dims(r_electrons, 2)
    r_electrons_2 = tf.tile(tf.expand_dims(r_electrons, 2), (1, 1, n_electrons, 1))
    ee_vectors = r_electrons_1 - tf.transpose(r_electrons_2, perm=(0, 2, 1, 3))  # * -1.
print(time() - t0)

t0 = time()
for _ in range(1000):
    re1 = tf.expand_dims(r_electrons, 1)
    re2 = tf.transpose(re1, perm=(0, 2, 1, 3))
    ee_vec = - re1 + re2
print(time() - t0)

print(tf.reduce_mean(tf.abs(ee_vec - ee_vectors)))

t0 = time()
for _ in range(n):
    mask1 = tf.eye(n_electrons, dtype=tf.bool)
    mask1 = ~tf.tile(tf.expand_dims(tf.expand_dims(mask1, 0), 3), (n_samples, 1, 1, 3))
print(time() - t0)

t0 = time()
for _ in range(n):
    mask2 = ~tf.transpose(tf.eye(n_electrons, batch_shape=(n_samples, 3), dtype=tf.bool), perm=(0, 2, 3, 1))
print(time() - t0)

print(tf.reduce_mean(tf.abs(tf.cast(mask1, tf.float32) - tf.cast(mask2, tf.float32))))


