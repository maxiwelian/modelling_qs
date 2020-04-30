import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf


def compare_tensors(t1, t2):
    try:
        return (tf.reduce_mean(tf.abs(t1 - t2)))
    except:
        return (t1, t2)


def rs(x):
    return tf.reduce_sum(x, axis=-1, keepdims=True)


def gc(x1, x2):
    u = tf.reduce_sum(x1 * x2, axis=-1, keepdims=True)
    sech = 1. / tf.math.cosh(u)
    sech2 = sech ** 2
    dx1 = sech2 * x2
    dx2 = sech2 * x1
    return dx1, dx2


def grad_grad(dx1dash, dx2dash, *cache):
    x1un, x2un = cache
    u = tf.reduce_sum(x1un * x2un, axis=-1, keepdims=True)
    tanh = tf.nn.tanh(u)
    sech = 1. / tf.math.cosh(u)
    sech2 = sech ** 2
    dsech2 = -2. * sech2 * tanh

    # rs reduce sum over all variables that the derivative is over i.e. the backpropagated errors
    d2x1 = dsech2 * x2un * rs(dx1dash * x2un)
    dx1dx2 = dsech2 * x2un * rs(dx2dash * x1un) + dx2dash * sech2

    d2x2 = dsech2 * x1un * rs(dx2dash * x1un)
    dx2dx1 = dsech2 * x1un * rs(dx1dash * x2un) + dx1dash * sech2

    ddx1 = d2x1 + dx1dx2
    ddx2 = d2x2 + dx2dx1
    return ddx1, ddx2


@tf.custom_gradient
def grad_custom(x1un, x2un):
    dx1, dx2 = gc(x1un, x2un)
    cache = (x1un, x2un)
    return (dx1, dx2), lambda dx1dash, dx2dash: grad_grad(dx1dash, dx2dash, *cache)


@tf.custom_gradient
def custom(x1, x2):
    u = tf.reduce_sum(x1 * x2, axis=-1, keepdims=True)
    z = tf.nn.tanh(u)

    def grad(dy):
        dx1p, dx2p = grad_custom(x1, x2)
        return dy * dx1p, dy * dx2p
    return z, grad

n_dim = 2
x0 = tf.random.normal((n_dim,))
x1 = tf.random.normal((n_dim,))

x00 = tf.Variable(x0)
x01 = tf.Variable(x1)

def compute_derivatives(variables, i, j):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            y = variables[0] * variables[1]
            q = tf.reduce_sum(y, axis=-1)
            z = tf.nn.tanh(q)
        grad1 = gg.gradient(z, variables[i])
    try:
        grad_grad1 = g.gradient(grad1, variables[j])

    except AttributeError:
        grad_grad1 = None
    return grad1, grad_grad1

print('with no intermediate variable ')

def compute_derivatives_custom(variables, i, j):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = custom(variables[0], variables[1])
        grad1 = gg.gradient(z, variables[i])
    try:
        grad_grad1 = g.gradient(grad1, variables[j])
    except AttributeError:
        grad_grad1 = None
    return grad1, grad_grad1

n_vars = 2
for i in range(n_vars):
    for j in range(n_vars):
        print('vars: ', i, j)
        variables = [x00, x01]
        g1, gg1 = compute_derivatives(variables, i, j)

        variables = [x00, x01]
        g2, gg2 = compute_derivatives_custom(variables, i, j)
        print('first', compare_tensors(g1, g2))
        # print(g1, g2)
        print('second', compare_tensors(gg1, gg2))
        # print(gg1, gg2)
        # print('\n')

print('with intermediate variable')

def compute_derivatives(variables, i, j):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            y = variables[0] * variables[1]
            q = tf.reduce_sum(y * variables[2], axis=-1)
            z = tf.nn.tanh(q)
        grad1 = gg.gradient(z, variables[i])
    try:
        grad_grad1 = g.gradient(grad1, variables[j])

    except AttributeError:
        grad_grad1 = None
    return grad1, grad_grad1


def compute_derivatives_custom(variables, i, j):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            y = variables[0] * variables[1]
            z = custom(y, variables[2])
        grad1 = gg.gradient(z, variables[i])
    try:
        grad_grad1 = g.gradient(grad1, variables[j])
    except AttributeError:
        grad_grad1 = None
    return grad1, grad_grad1


n_dim = 2
x0 = tf.random.normal((n_dim,))
x1 = tf.random.normal((n_dim,))
x2 = tf.random.normal((n_dim,))

x00 = tf.Variable(x0)
x01 = tf.Variable(x1)
x02 = tf.Variable(x2)

n_vars = 3
for i in range(n_vars):
    for j in range(n_vars):
        print('vars: ', i, j)
        variables = [x00, x01, x02]
        g1, gg1 = compute_derivatives(variables, i, j)

        variables = [x00, x01, x02]
        g2, gg2 = compute_derivatives_custom(variables, i, j)
        print('first', compare_tensors(g1, g2))
        # print(g1, g2)
        print('second', compare_tensors(gg1, gg2))
        print(gg1 / gg2)
        # print(gg1, gg2)
        # print('\n')


