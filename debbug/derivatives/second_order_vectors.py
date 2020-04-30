
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import tensorflow.keras as tk
import numpy as np

tf.keras.backend.set_floatx('float64')

def get_second_order_nm(model, samples, n, m):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.vars[m])
    grad_grad = g.gradient(grad, model.vars[n])
    return grad_grad


def get_second_order_ww(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.w3)
    grad_grad = g.gradient(grad, model.w3)
    return grad_grad

def get_second_order_aw(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.w3)
    grad_grad = g.gradient(grad, model.y1)
    return grad_grad

def get_second_order_bw(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.w3)
    grad_grad = g.gradient(grad, model.y2)
    return grad_grad

def get_second_order_wa(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.y1)
    grad_grad = g.gradient(grad, model.w3)
    return grad_grad

def get_second_order_aa(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.y1)
    grad_grad = g.gradient(grad, model.y1)
    return grad_grad

def get_second_order_ba(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.y1)
    grad_grad = g.gradient(grad, model.y2)
    return grad_grad

def get_second_order_wb(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.y2)
    grad_grad = g.gradient(grad, model.w3)
    return grad_grad

def get_second_order_ab(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.y2)
    grad_grad = g.gradient(grad, model.y1)
    return grad_grad

def get_second_order_bb(model, samples):
    with tf.GradientTape(True) as g:
        with tf.GradientTape(True) as gg:
            z = model(samples)
        grad = gg.gradient(z, model.y2)
    grad_grad = g.gradient(grad, model.y2)
    return grad_grad

def compare_tf_tensors(tflow1, tflow2, shape=True, rtol=1e-05, atol=1e-08): #
    if shape:
        print(tflow1.shape, tflow2.shape)
    tflow1 = tflow1.numpy()
    tflow2 = tflow2.numpy()
    return print(np.mean(np.abs(tflow1 - tflow2)))

def get_first_order(model, samples):
    with tf.GradientTape() as g:
        z = model(samples)
    grad = g.gradient(z, model.w3)
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

        z = tf.tanh(tf.reduce_sum(self.w3 * self.y1 * self.y2, axis=-1))

        self.vars = [self.y0, self.y1, self.y2, self.w0, self.w1, self.w2, self.w3]
        return tf.squeeze(z)


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

        z = custom(self.w3, self.y1, self.y2)

        self.vars = [self.y0, self.y1, self.y2, self.w0, self.w1, self.w2, self.w3]
        return z

def rs(x):
    return tf.reduce_sum(x, axis=-1, keepdims=True)
    # return x



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
#       def second_order_and_transpose(ddy):
#         return second_order_for_x(...), gradient_wrt_dy(...)
#       return x_grad, second_order_and_transpose
#     return dy * first_order_custom(x)
#   return y, first_order_gradien

@tf.custom_gradient
def grad_custom(w, a, b):
    u = tf.reduce_sum(w * a * b, axis=-1, keepdims=True)
    sech = 1. / tf.math.cosh(u)
    sech2 = sech**2
    dw = sech2 * a * b
    da = sech2 * w * b
    db = sech2 * w * a

    def grad_grad(dwdash, dadash, dbdash):
        tanh = tf.tanh(u)
        dsech2 = -2.*sech2*tanh
        d2w = dsech2 * a*b * rs(a*b)
        d2a = dsech2 * w*b * tf.reduce_sum(w*b, axis=-1, keepdims=True)
        d2b = dsech2 * w*a * tf.reduce_sum(w*a, axis=-1, keepdims=True)

        #         i \neq j                    i = j
        # dwda = dsech2 * rs(w * b) * a * b + sech2 * b
        # dwdb = dsech2 * rs(w * a) * a * b + sech2 * a
        #
        # dadw = dsech2 * rs(a * b) * w * b + sech2 * b
        # dadb = dsech2 * rs(a * w) * w * b + sech2 * w
        #
        # dbdw = dsech2 * rs(b * a) * w * a + sech2 * a
        # dbda = dsech2 * rs(b * w) * w * a + sech2 * w
        # other
        # dwda = dsech2 * w * b * rs(a * b) + sech2 * b
        # dwdb = dsech2 * w * a * rs(a * b) + sech2 * a
        # dadw = dsech2 * a * b * rs(w * b) + sech2 * b
        # dadb = dsech2 * a * w * rs(w * b) + sech2 * w
        # dbdw = dsech2 * b * a * rs(w * a) + sech2 * a
        # dbda = dsech2 * b * w * rs(w * a) + sech2 * w
        # other other
        dwda = dsech2 * rs(w * b) * a * b + rs(sech2 * b)
        dwdb = dsech2 * rs(w * a) * a * b + rs(sech2 * a)

        dadw = dsech2 * rs(a * b) * w * b + rs(sech2 * b)
        dadb = dsech2 * rs(a * w) * w * b + rs(sech2 * w)

        dbdw = dsech2 * rs(b * a) * w * a + rs(sech2 * a)
        dbda = dsech2 * rs(b * w) * w * a + rs(sech2 * w)

        ddw = dwdash*d2w+dadash*dwda+dbdash*dwdb
        dda = dadash*d2a+dwdash*dadw+dbdash*dadb
        ddb = dbdash*d2b+dwdash*dbdw+dadash*dbda
        return tf.reduce_sum(ddw, axis=0, keepdims=True), dda, ddb

    return (dw, da, db), grad_grad

@tf.custom_gradient
def custom(w, a, b):
    z = tf.tanh(tf.reduce_sum(w * a * b, axis=-1, keepdims=True))
    def grad(dy):
        # dy = tf.expand_dims(dy, -1)
        dw, da, db = grad_custom(w, a, b)
        dy = expand(dy, dw.shape)
        return tf.reduce_sum(dy*dw, axis=0, keepdims=True), dy*da, dy*db

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

g1 = get_first_order(model1, samples)
g2 = get_first_order(model2, samples)
print('first order')
compare_tf_tensors(g1, g2)

compare = 0
gg1 = get_second_order_nm(model1, samples, compare, compare)
gg2 = get_second_order_nm(model2, samples, compare, compare)
print('test')

compare_tf_tensors(gg1, gg2)






# g1, gg1 = get_second_order_inputs(model1, samples)
# g2, gg2 = get_second_order_inputs(model2, samples)





#
# def get_second_order_inputs(model, samples):
#     r_s = [samples[..., i] for i in range(samples.shape[-1])]
#     with tf.GradientTape(True) as g:
#         [g.watch(r) for r in r_s]
#         samples = tf.stack(r_s, -1)
#         samples = tf.reshape(samples, (-1, n_dim))
#         with tf.GradientTape(True) as gg:
#             gg.watch(samples)
#             log_phi = model(samples)
#         dlogphi_dr = gg.gradient(log_phi, samples)
#         gs = tf.reshape(dlogphi_dr, (-1, n_dim))
#         grads = [gs[..., i] for i in range(gs.shape[-1])]
#     ggs = tf.stack([g.gradient(grad, r) for grad, r in zip(grads, r_s)], -1)
#     return gs, ggs
# # full matrix does not work
# @tf.custom_gradient
# def grad_custom(w, a, b):
#     u = tf.reduce_sum(w * a * b, axis=-1, keepdims=True)
#     sech = 1. / tf.math.cosh(u)
#     sech2 = sech ** 2
#     dw = sech2 * a * b
#     da = sech2 * w * b
#     db = sech2 * w * a
#
#     def grad_grad(dwdash, dadash, dbdash):
#         sech = 1. / tf.math.cosh(u)
#         sech2 = sech ** 2
#         sech2 = tf.expand_dims(sech2, -1)
#         dwdash = tf.expand_dims(dwdash, -1)
#         tanh = tf.expand_dims(tf.tanh(u), -1)
#         dsech2 = -2.*sech2*tanh
#         d2w = dsech2 * tf.expand_dims(a*b, -1) * tf.expand_dims(a*b, -2)
#         d2a = dsech2 * w*b * w*b
#         d2b = dsech2 * w*a * w*a
#
#         # i == j
#         dwda = dsech2 * tf.expand_dims(w * b, -2) * tf.expand_dims(a * b, -1) + sech2 * tf.expand_dims(b, -1)
#         dwdb = dsech2 * tf.expand_dims(w * a, -2) * tf.expand_dims(a * b, -1) + sech2 * tf.expand_dims(a, -1)
#
#         dadw = dsech2 * rs(a * b) * w * b + sech2 * b
#         dadb = dsech2 * rs(a * w) * w * b + sech2 * w
#
#         dbdw = dsech2 * rs(b * a) * w * a + sech2 * a
#         dbda = dsech2 * rs(b * w) * w * a + sech2 * w
#
#         return tf.reduce_sum(dwdash*d2w+dadash*dwda+dbdash*dwdb, axis=0, keepdims=True), \
#                (dadash*d2a+dwdash*dadw+dbdash*dadb), \
#                (dbdash*d2b+dwdash*dbdw+dadash*dbda)
#         #
#         # return tf.reduce_sum(dwdash * (d2w + dwda + dwdb), axis=0, keepdims=True), \
#         #        (dadash * (d2a + dadw + dadb)), \
#         #        (dbdash * (d2b + dbdw + dbda))
#
#     return (dw, da, db), grad_grad