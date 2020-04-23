import tensorflow as tf
import tensorflow_probability as tfp
from time import time

n_samples = 4096
n_electrons = 4
n = 100
mu = tf.zeros(3)
sig = 0.02
sigma = tf.eye(3) * sig
sig = tf.sqrt(sig)

prev_sample = tf.random.uniform((n_samples, n_electrons, 3))
step_gaussian = tfp.distributions.MultivariateNormalFullCovariance(mu, sigma)

t0 = time()
for _ in range(n):
    x = step_gaussian.sample(prev_sample.shape[:-1], dtype=tf.float32)
print(time() - t0)

t0 = time()
for _ in range(n):
    x = tf.random.normal(prev_sample.shape, stddev=sig)
print(time() - t0)