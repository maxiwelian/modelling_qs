import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.random.set_seed(1)

def compare(t1, t2):
    return tf.reduce_mean(tf.abs(t1 - t2))

@tf.custom_gradient
def safe_norm_grad(x, norm):
    # x : (n, ne**2, 3)
    # norm : (n, ne**2, 1)
    print(x.shape, norm.shape)
    # tf.debugging.check_numerics(x, 'x')
    # tf.debugging.check_numerics(norm, 'norm')
    g = x / norm
    # tf.debugging.check_numerics(g, 'g')
    g = tf.where(tf.math.is_nan(g), tf.zeros_like(g), g)
    cache = (x, norm)

    def grad_grad(dy):
        x, norm = cache
        x = tf.expand_dims(x, -1)  # (n, ne**2, 3, 1)
        xx = x * tf.transpose(x, perm=(0, 1, 3, 2))  # cross terms
        inv_norm = tf.tile(1. / norm, (1, 1, 3))  # (n, ne**2, 3) inf where the ee terms are same e
        norm_diag = tf.linalg.diag(inv_norm) # (n, ne**2, 3, 3) # diagonal where the basis vector is the same
        gg = norm_diag - xx / tf.expand_dims(norm, -1)**3
        gg = tf.reduce_sum(gg, axis=-1)
        gg = tf.where(tf.math.is_nan(gg), tf.zeros_like(gg), gg)
        return dy*gg, None

    return g, grad_grad

@tf.custom_gradient
def safe_norm(x):
    norm = tf.norm(x, keepdims=True, axis=-1)
    def grad(dy):
        g = safe_norm_grad(x, norm)
        return dy*g
    return norm, grad

n_samples = 1000
n_electrons = 8
x = tf.random.normal((n_samples, n_electrons, 3), stddev=2., mean=2.)
# z = tf.random.normal(x.shape)

x = tf.expand_dims(x, 2)
# z = tf.expand_dims(z, 1)

x = x - tf.transpose(x, perm=(0, 2, 1, 3))
# x = x - z

x = tf.reshape(x, (-1, n_electrons**2, 3))
# x = tf.zeros((n_samples, n_electrons, 3))

with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y1 = tf.linalg.norm(x, keepdims=True, axis=-1)
    g1 = gg.gradient(y1, x)
g2 = g.gradient(g1, x)
# print(g1, g2)

with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y11 = safe_norm(x)
    g11 = gg.gradient(y11, x)
g22 = g.gradient(g11, x)
# print(g11, g22)
# print(g22)

# g1 = tf.where(tf.math.is_nan(g1), tf.zeros_like(g1), g1)
# g2 = tf.where(tf.math.is_nan(g2), tf.zeros_like(g2), g2)

# for a, b in zip(g2, g22):
#     print(a[1], b[1])

print(compare(y1, y11))
print(compare(g1, g11))
print(compare(g2, g22))