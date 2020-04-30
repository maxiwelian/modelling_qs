

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import tensorflow.keras as tk
import numpy as np

tf.keras.backend.set_floatx('float64')

def compare_tf_tensors(tflow1, tflow2, shape=True, rtol=1e-05, atol=1e-08): #
    if shape:
        print(tflow1.shape, tflow2.shape)
    tflow1 = tflow1.numpy()
    tflow2 = tflow2.numpy()
    return print(np.mean(np.abs(tflow1 - tflow2)))

def get_first_order(model, samples, n):
    with tf.GradientTape() as g:
        z = model(samples)
    grad = g.gradient(z, model.vars[n])
    return grad

def tflaplacian(model, r_electrons):
    n_electrons = r_electrons.shape[1]
    r_electrons = tf.reshape(r_electrons, (-1, n_electrons*3))
    r_s = [r_electrons[..., i] for i in range(r_electrons.shape[-1])]
    with tf.GradientTape(True) as g:
        [g.watch(r) for r in r_s]
        r_electrons = tf.stack(r_s, -1)
        r_electrons = tf.reshape(r_electrons, (-1, n_electrons, 3))
        with tf.GradientTape(True) as gg:
            gg.watch(r_electrons)
            log_phi = model(r_electrons)
        dlogphi_dr = gg.gradient(log_phi, r_electrons)
        dlogphi_dr = tf.reshape(dlogphi_dr, (-1, n_electrons*3))
        grads = [dlogphi_dr[..., i] for i in range(dlogphi_dr.shape[-1])]
    d2logphi_dr2 = tf.stack([g.gradient(grad, r) for grad, r in zip(grads, r_s)], -1)
    return dlogphi_dr**2, d2logphi_dr2

class Model1(tk.Model):
    def __init__(self):
        super(Model1, self).__init__()
        self.w0 = tf.Variable(w0)
        self.w1 = tf.Variable(w1)
        self.w2 = tf.Variable(w2)
        self.w3 = tf.Variable(w3)

    def __call__(self, x):
        self.y0 = x @ self.w0
        self.y1 = self.y0 @ self.w1
        self.y2 = self.y0 @ self.w2
        self.y3 = self.y1 * self.y2

        z = tf.tanh(tf.reduce_sum(self.y3, axis=-1))

        self.vars = [self.y0, self.y1, self.y2, self.y3, self.w0, self.w1, self.w2]
        return z

class Model2(tk.Model):
    def __init__(self):
        super(Model2, self).__init__()
        self.w0 = tf.Variable(w0)
        self.w1 = tf.Variable(w1)
        self.w2 = tf.Variable(w2)
        self.w3 = tf.Variable(w3)

    def __call__(self, x):
        self.y0 = x @ self.w0
        self.y1 = self.y0 @ self.w1
        self.y2 = self.y0 @ self.w2
        self.y3 = self.y1 * self.y2

        z = custom(self.y3)

        self.vars = [self.y0, self.y1, self.y2, self.y3, self.w0, self.w1, self.w2]
        return z

def rs(x):
    return tf.reduce_sum(x, axis=-1, keepdims=True)

def expand(x, shape):
    s1 = len(x.shape)
    for i in range(len(shape) - s1):
        x = tf.expand_dims(x, -1)
    return x

# @tf.custom_gradient
# def op_with_fused_backprop(x):
#   y, x_grad = fused_op(x)
#   def first_order_gradient(dy):
#     @tf.custom_gradient
#     def first_order_custom(unused_x):
#
#       def second_order_and_transpose(ddy):
#
#         # the second order of x and the gradient wrt to dy
#         return second_order_for_x(...), gradient_wrt_dy(...)
#
#       # for some reason the gradient of x followed by the second order gradient method
#       return x_grad, second_order_and_transpose
#
#
#     # dy times the output of your grad function
#     return dy * first_order_custom(x)
#
#
#   # output and the first order grad
#   return y, first_order_gradient

def get_second_order_nm(model, samples, n, m):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.vars[m])
    grad_grad = g.gradient(grad, model.vars[n])
    return grad_grad

def fused_op(w):
    # fused op
    z = tf.tanh(tf.reduce_sum(w, axis=-1, keepdims=True))
    u = tf.reduce_sum(w, axis=-1, keepdims=True)
    sech = 1. / tf.math.cosh(u)
    dw = sech ** 2
    return z, dw

def gc(w):
    u = tf.reduce_sum(w, axis=-1, keepdims=True)
    sech = 1. / tf.math.cosh(u)
    dw = sech ** 2
    return dw

def grad_grad(ddash, *cache):
    w = cache
    u = tf.reduce_sum(w, axis=-1, keepdims=True)
    sech = 1. / tf.math.cosh(u)
    sech2 = sech**2
    tanh = tf.tanh(u)
    dsech2 = -2. * sech2 * tanh
    out = ddash * dsech2
    return out

@tf.custom_gradient
def grad_custom(w):
    dw = gc(w)
    cache = (w,)
    return dw, lambda ddash: grad_grad(ddash, *cache)

@tf.custom_gradient
def custom(w):
    z = tf.tanh(tf.reduce_sum(w, axis=-1, keepdims=True))
    def grad(dy):
        dw = grad_custom(w)
        dy = expand(dy, dw.shape)
        return dy*grad_custom(w)
    return tf.squeeze(z), grad

n_dim = 3
n_samples = 10

w0 = tf.random.normal((n_dim, n_dim), dtype=tf.float64)
w1 = tf.random.normal((n_dim, n_dim), dtype=tf.float64)
w2 = tf.random.normal((n_dim, n_dim), dtype=tf.float64)
w3 = tf.random.normal((1, n_dim), dtype=tf.float64)

samples = tf.random.normal((n_samples, n_dim), dtype=tf.float64)

model1 = Model1()
model2 = Model2()
o1 = model1(samples)
o2 = model2(samples)
print('output')
compare_tf_tensors(o1, o2)
n_vars = len(model1.vars)
for i in range(n_vars):
    g1 = get_first_order(model1, samples, i)
    g2 = get_first_order(model2, samples, i)
    compare_tf_tensors(g1, g2)


for i in range(n_vars):
    for j in range(n_vars):
        print(i, j)
        gg1 = get_second_order_nm(model1, samples, i, j)
        gg2 = get_second_order_nm(model2, samples, i, j)
        # print(gg1, gg2)
        compare_tf_tensors(gg1, gg2)
# compare = 1
# for i in range(len(model1.vars)):
#     print(i)
#     gg1 = get_second_order_nm(model1, samples, compare, i)
#     gg2 = get_second_order_nm(model2, samples, compare, i)
#
#     compare_tf_tensors(gg1, gg2)

