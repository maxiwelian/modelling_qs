

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

def get_second_order_inputs(model, samples):
    r_s = [samples[..., i] for i in range(samples.shape[-1])]
    with tf.GradientTape(True) as g:
        [g.watch(r) for r in r_s]
        samples = tf.stack(r_s, -1)
        samples = tf.reshape(samples, (-1, n_dim))
        with tf.GradientTape(True) as gg:
            gg.watch(samples)
            log_phi = model(samples)
        dlogphi_dr = gg.gradient(log_phi, samples)
        gs = tf.reshape(dlogphi_dr, (-1, n_dim))
        grads = [gs[..., i] for i in range(gs.shape[-1])]
    ggs = tf.stack([g.gradient(grad, r) for grad, r in zip(grads, r_s)], -1)
    return gs, ggs

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
        return z

def rs(x):
    return tf.reduce_sum(x, axis=-1, keepdims=True)

def expand(x, shape):
    s1 = len(x.shape)
    for i in range(len(shape) - s1):
        x = tf.expand_dims(x, -1)
    return x


def fused_op(w):
    # fused op
    z = tf.tanh(tf.reduce_sum(w, axis=-1, keepdims=True))
    u = tf.reduce_sum(w, axis=-1, keepdims=True)
    sech = 1. / tf.math.cosh(u)
    dw = sech ** 2
    return z, dw

@tf.custom_gradient
def custom(w):
    # z, dw = fused_op(w)
    z = tf.tanh(tf.reduce_sum(w, axis=-1, keepdims=True))
    def grad(dy):
        @tf.custom_gradient
        def grad_custom(w):
            u = tf.reduce_sum(w, axis=-1, keepdims=True)
            sech = 1. / tf.math.cosh(u)
            dw = sech ** 2

            def grad_grad(ddash):
                # u = tf.reduce_sum(w, axis=-1, keepdims=True)
                # sech = 1. / tf.math.cosh(u)
                sech2 = sech**2
                tanh = tf.tanh(u)
                dsech2 = -2. * sech2 * tanh
                out = ddash * dsech2
                return out

            return dw, grad_grad

        dw = grad_custom(w)
        dy = expand(dy, dw.shape)

        return dy*dw

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

g1, gg1 = get_second_order_inputs(model1, samples)
g2, gg2 = get_second_order_inputs(model2, samples)
compare_tf_tensors(g1, g2)
compare_tf_tensors(gg1, gg2)

#
# o1 = model1(samples)
# o2 = model2(samples)
# print('output')
# compare_tf_tensors(o1, o2)
# n_vars = len(model1.vars)
# for i in range(n_vars):
#     g1 = get_first_order(model1, samples, i)
#     g2 = get_first_order(model2, samples, i)
#     compare_tf_tensors(g1, g2)
#
#
#
#
# def get_second_order_nm(model, samples, n, m):
#     with tf.GradientTape(True) as g:
#         with tf.GradientTape(True) as gg:
#             z = model(samples)
#         grad = gg.gradient(z, model.vars[m])
#     grad_grad = g.gradient(grad, model.vars[n])
#     return grad_grad
# for i in range(n_vars):
#     for j in range(n_vars):
#         print(i, j)
#         gg1 = get_second_order_nm(model1, samples, i, j)
#         gg2 = get_second_order_nm(model2, samples, i, j)
#         # print(gg1, gg2)
#         compare_tf_tensors(gg1, gg2)
# compare = 1
# for i in range(len(model1.vars)):
#     print(i)
#     gg1 = get_second_order_nm(model1, samples, compare, i)
#     gg2 = get_second_order_nm(model2, samples, compare, i)
#
#     compare_tf_tensors(gg1, gg2)

