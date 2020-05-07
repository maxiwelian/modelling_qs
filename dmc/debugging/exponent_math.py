


import tensorflow as tf

x1 = -2.
x2 = -4.3
tau = 0.01

z1 = tf.exp(-1 / 2 * (x1 + x2) * tau)
z2 = tf.exp(-(1 / 2) * (x1 + x2) * tau)

print(z1, z2)