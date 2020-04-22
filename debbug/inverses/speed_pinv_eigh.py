import tensorflow as tf
from time import time

n_samples = 1
n_dim = 700
damping = 10.
n = 10

x = tf.random.uniform((n_samples, n_dim, n_dim))
x = tf.linalg.matrix_transpose(x) + x

t0 = time()
for _ in range(n):
    vals, vecs = tf.linalg.eigh(x + tf.eye(n_dim) * damping)
    # with tf.device('/cpu:0'):
    #     vals, vecs = tf.linalg.eigh(x + tf.eye(n_dim) * damping)

    vals = tf.eye(vals.shape[-1], batch_shape=(n_samples,)) / tf.expand_dims(vals, -1)
    inv1 = vecs @ vals @ tf.linalg.matrix_transpose(vecs)
print(time() - t0)

t0 = time()
for _ in range(n):
    inv2 = tf.linalg.inv(x + tf.eye(n_dim) * damping)
print(time() - t0)

# print(1./vals)





def compare(t1, t2):
    return tf.reduce_mean(tf.abs(t1 - t2))


print(compare(inv1, inv2))
