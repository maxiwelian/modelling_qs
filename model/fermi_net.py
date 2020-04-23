import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
from utils.utils import tofloat


def initializer(in_dim, weight_shape, out_dim, _):
    minval = tf.maximum(-1., -(6/(in_dim+out_dim))**0.5)
    maxval = tf.minimum(1., (6/(in_dim+out_dim))**0.5)
    weights = tf.random.uniform(weight_shape, minval=minval, maxval=maxval)
    return weights

def env_initializer(in_dim, weight_shape, out_dim, env_init):
    weights = tf.random.uniform(weight_shape, minval=-env_init, maxval=env_init)
    return weights

class fermiNet(tk.Model):
    """
    fermi net, baby
    """
    def __init__(self,
                 gpu_id,
                 r_atoms,
                 n_electrons,
                 n_atoms,
                 n_spin_up,
                 n_spin_down,
                 nf_hidden_single,
                 nf_hidden_pairwise,
                 n_determinants,
                 env_init):

        super(fermiNet, self).__init__()

        # --- initializations
        nf_single_in = 4 * n_atoms
        n_pairwise = n_electrons ** 2 - n_electrons
        nf_pairwise_in = 4

        nf_single_in_mixed = 3 * nf_single_in + 2 * nf_pairwise_in

        nf_single_intermediate_in = nf_hidden_single * 3 + nf_hidden_pairwise * 2
        nf_pairwise_intermediate_in = nf_hidden_pairwise

        # --- use internally
        self.n_electrons = n_electrons
        self.n_atoms = n_atoms
        self.n_spin_up = n_spin_up
        self.n_spin_down = n_electrons - n_spin_up
        n_spins = n_spin_up + n_spin_down
        self.n_determinants = n_determinants
        self.r_atoms = r_atoms
        self.n_pairwise = n_pairwise

        # --- model
        self.input_mixer = Mixer(n_electrons, nf_single_in, n_pairwise, nf_pairwise_in, n_spin_up, n_spin_down)

        self.single_stream_in = Stream(nf_single_in_mixed, nf_hidden_single, n_spins, gpu_id, 0)
        self.pairwise_stream_in = Stream(nf_pairwise_in, nf_hidden_pairwise, n_pairwise, gpu_id, 0)
        self.mixer_in = Mixer(n_electrons, nf_hidden_single, n_pairwise, nf_hidden_pairwise, n_spin_up, n_spin_down)

        self.s1 = Stream(nf_single_intermediate_in, nf_hidden_single, n_spins, gpu_id, 1)
        self.p1 = Stream(nf_pairwise_intermediate_in, nf_hidden_pairwise, n_pairwise, gpu_id, 1)
        self.m1 = Mixer(n_electrons, nf_hidden_single, n_pairwise, nf_hidden_pairwise, n_spin_up, n_spin_down)

        self.s2 = Stream(nf_single_intermediate_in, nf_hidden_single, n_spins, gpu_id, 2)
        self.p2 = Stream(nf_pairwise_intermediate_in, nf_hidden_pairwise, n_pairwise, gpu_id, 2)
        self.m2 = Mixer(n_electrons, nf_hidden_single, n_pairwise, nf_hidden_pairwise, n_spin_up, n_spin_down)

        self.final_single_stream = Stream(nf_single_intermediate_in, nf_hidden_single, n_spins, gpu_id, 3)

        self.envelopes = \
            envelopesLayer(n_spin_up, n_spin_down, n_atoms, nf_hidden_single, n_determinants, env_init, gpu_id)

        # self.output_layer = tf.Variable(initializer(n_determinants, (1,n_determinants,1,1), 1, _))
        # self.output_layer = tf.Variable(tf.ones((1, n_determinants, 1, 1))/n_determinants, name='w_1')
        self.output_layer = tf.Variable(env_initializer(16, (1, n_determinants, 1, 1), 1, env_init / n_determinants),
                                        name='wf_1')
        # self.epoch = 1

    # @tf.function  # phase = 0, 1, 2 // test, supervised, unsupervised
    def call(self, r_electrons):
        n_samples = r_electrons.shape[0]
        r_atoms = tf.tile(self.r_atoms, (n_samples, 1, 1))

        # --- computing inputs
        # single_inputs: (b, n_electrons, 4), pairwise_inputs: (, n_pairwise, 4)
        ae_vectors = compute_ae_vectors(r_atoms, r_electrons)
        single, pairwise = compute_inputs(r_electrons, n_samples, ae_vectors, self.n_atoms, self.n_electrons)
        single_mix = self.input_mixer(single, pairwise, n_samples, self.n_electrons)

        # --- input layer
        single, a_in_s, s_in_s = self.single_stream_in(single_mix, n_samples, self.n_electrons)
        pairwise, a_in_p, s_in_p = self.pairwise_stream_in(pairwise, n_samples, self.n_pairwise)
        single_mix = self.mixer_in(single, pairwise, n_samples, self.n_electrons)

        # --- intermediate layers
        tmp, a_1_s, s_1_s = self.s1(single_mix, n_samples, self.n_electrons)
        single += tmp
        tmp, a_1_p, s_1_p = self.p1(pairwise, n_samples, self.n_pairwise)
        pairwise += tmp
        single_mix = self.m1(single, pairwise, n_samples, self.n_electrons)

        tmp, a_2_s, s_2_s = self.s2(single_mix, n_samples, self.n_electrons)
        single += tmp
        tmp, a_2_p, s_2_p = self.p2(pairwise, n_samples, self.n_pairwise)
        pairwise += tmp
        single_mix = self.m2(single, pairwise, n_samples, self.n_electrons)

        # --- final layer
        tmp, a_f, s_f = self.final_single_stream(single_mix, n_samples, self.n_electrons)
        single += tmp

        # --- envelopes
        spin_up_determinants, spin_down_determinants, a_up, s_up, a_down, s_down = \
            self.envelopes(single, ae_vectors, n_samples)

        # --- logabsdet
        log_psi, sign, a, s = log_abs_sum_det(spin_up_determinants, spin_down_determinants, self.output_layer)

        # yep # 7, 8, 9, 10, 11, 12 #
        activation = (a_in_s, a_in_p, a_1_s, a_1_p, a_2_s, a_2_p, a_f,
                      a_up[0], a_up[1], a_up[2], a_down[0], a_down[1], a_down[2], a)
        sensitivity = (s_in_s, s_in_p, s_1_s, s_1_p, s_2_s, s_2_p, s_f,
                       s_up[0], s_up[1], s_up[2], s_down[0], s_down[1], s_down[2], s)

        return tf.squeeze(log_psi), activation, sensitivity, spin_up_determinants, spin_down_determinants



class envelopesLayer(tk.Model):
    def __init__(self, n_spin_up, n_spin_down, n_atoms, nf_single, n_determinants, env_init, gpu_id):
        super(envelopesLayer, self).__init__()
        # --- variables
        self.n_spin_up = n_spin_up
        self.n_spin_down = n_spin_down
        self.n_k = n_determinants
        self.n_atoms = n_atoms

        # --- envelopes
        self.spin_up_envelope = envelopeLayer(n_spin_up, n_atoms, nf_single, n_determinants, env_init, gpu_id, name='up')
        self.spin_down_envelope = envelopeLayer(n_spin_down, n_atoms, nf_single, n_determinants, env_init, gpu_id, name='down')

    # @tf.function
    def call(self, inputs, ae_vectors, n_samples):
        # --- arrange inputs
        spin_up_ae_vectors, spin_down_ae_vectors = tf.split(ae_vectors, [self.n_spin_up, self.n_spin_down], axis=1)
        spin_up_inputs, spin_down_inputs = tf.split(inputs, [self.n_spin_up, self.n_spin_down], axis=1)

        # --- envelopes
        spin_up_output, a_up, s_up = self.spin_up_envelope(spin_up_inputs, spin_up_ae_vectors,
                                               n_samples, self.n_spin_up, self.n_k, self.n_atoms)
        spin_down_output, a_down, s_down = self.spin_down_envelope(spin_down_inputs, spin_down_ae_vectors,
                                                   n_samples, self.n_spin_down, self.n_k, self.n_atoms)

        return spin_up_output, spin_down_output, a_up, s_up, a_down, s_down


class envelopeLayer(tk.Model):
    def __init__(self, n_spins, n_atoms, nf_single, n_determinants, env_init, gpu_id, name=''):
        super(envelopeLayer, self).__init__()
        # k: n_determinants, i: n_electrons, f: n_features
        w = initializer(nf_single, (n_determinants, n_spins, nf_single, 1), 1, None)
        b = tf.zeros((n_determinants, n_spins, 1, 1))
        w = tf.concat((w, b), axis=2)

        self.w = tf.Variable(w, name='env_%s_w_%i' % (name, n_spins))

        self.Sigma = tf.Variable(env_initializer(3, (n_determinants, n_spins, n_atoms, 3, 3), 3, env_init),
                                 name='env_%s_sigma_%i' % (name, n_spins))

        self.Pi = tf.Variable(env_initializer(n_atoms, (n_determinants, n_spins, n_atoms, 1), 1, env_init),
                              name='env_%s_pi_%i' % (name, n_spins))

    # @tf.function
    def call(self, inputs, ae_vectors, n_samples, n_spins, n_k, n_atoms):
        # inputs: (n_samples, n_electrons, nf_single)
        # ae_vectors: (n_samples, n_electrons, n_atoms, 3)
        # n: n_samples, e: n_electrons, f: nf_single, i: n_electrons, k: n_determinants

        # env_w 'njf,kifs->nkjis'
        inputs_w_bias = tf.concat((inputs, tf.ones((n_samples, n_spins, 1))), axis=-1)
        factor = tf.einsum('njf,kifs->njkis', inputs_w_bias, self.w)

        # k: n_determinants, i: n_electrons, m: n_atoms, n: n_samples, j: n_electrons
        # env_sigma 'njmv,kimvc->nkjimc'
        exponent = tf.einsum('njmv,kimvc->njkimc', ae_vectors, self.Sigma)
        exponential = tf.exp(-tf.norm(exponent, axis=-1))

        # env_pi 'njkim,kims->nkjis'
        exp = tf.einsum('njkim,kims->njkis', exponential, self.Pi)

        output = factor * exp
        output = tf.transpose(output, perm=(0, 2, 3, 1, 4))  # ij ordering doesn't matter / slight numerical diff

        return tf.squeeze(output, -1), (inputs, ae_vectors, exponential), (factor, exponent, exp)


class Stream(tk.Model):
    """
    single / pairwise electron streams
    """
    def __init__(self, in_dim, out_dim, n_spins, gpu_id, node):
        super(Stream, self).__init__()
        # --- variables
        # lim = tf.math.sqrt(6 / (in_dim + out_dim))
        # w = tf.concat((tf.random.uniform((in_dim, out_dim), minval=-lim, maxval=lim), tf.zeros((1, out_dim))), axis=0)
        w = initializer(in_dim, (in_dim, out_dim), out_dim, None)
        b = tf.zeros((1, out_dim))
        # b = tf.random.normal((1, out_dim), stddev=std, dtype=dtype)
        w = tf.concat((w, b), axis=0)
        self.w = tf.Variable(w, name='stream%i_%i' % (node, n_spins))

    def call(self, inputs, n_samples, n_streams):
        inputs_w_bias = tf.concat((inputs, tf.ones((n_samples, n_streams, 1))), axis=-1)
        out1 = inputs_w_bias @ self.w
        out2 = tf.nn.tanh(out1)
        return out2, inputs, out1


def compute_ae_vectors(r_atoms, r_electrons):
    # ae_vectors (n_samples, n_electrons, n_atoms, 3)
    r_atoms = tf.expand_dims(r_atoms, 1)
    r_electrons = tf.expand_dims(r_electrons, 2)
    ae_vectors = r_electrons - r_atoms
    return ae_vectors


# @tf.function
def compute_inputs(r_electrons, n_samples, ae_vectors, n_atoms, n_electrons):
    # r_atoms: (n_atoms, 3)
    # r_electrons: (n_samples, n_electrons, 3)
    # ae_vectors: (n_samples, n_electrons, n_atoms, 3)
    ae_distances = tf.norm(ae_vectors, axis=-1, keepdims=True)
    single_inputs = tf.concat((ae_vectors, ae_distances), axis=-1)
    single_inputs = tf.reshape(single_inputs, (-1, n_electrons, 4*n_atoms))

    re1 = tf.expand_dims(r_electrons, 2)
    re2 = tf.transpose(re1, perm=(0, 2, 1, 3))
    ee_vectors = re1 - re2

    mask = tf.eye(n_electrons, dtype=tf.bool)
    mask = ~tf.tile(tf.expand_dims(tf.expand_dims(mask, 0), 3), (n_samples, 1, 1, 3))

    ee_vectors = tf.boolean_mask(ee_vectors, mask)
    ee_vectors = tf.reshape(ee_vectors, (-1, n_electrons**2 - n_electrons, 3))
    ee_distances = tf.norm(ee_vectors, axis=-1, keepdims=True)
    pairwise_inputs = tf.concat((ee_vectors, ee_distances), axis=-1)

    return single_inputs, pairwise_inputs


class Mixer(tk.Model):
    """
    Mixes stream outputs to input into single streams
    """
    def __init__(self, n_electrons, n_single_features, n_pairwise, n_pairwise_features, n_spin_up, n_spin_down):
        super(Mixer, self).__init__()

        self.n_spin_up = tofloat(n_spin_up)
        self.n_spin_down = tofloat(n_spin_down)

        tmp1 = tf.ones((1, n_spin_up, n_single_features), dtype=tf.bool)
        tmp2 = tf.zeros((1, n_spin_down, n_single_features), dtype=tf.bool)
        self.spin_up_mask = tf.concat((tmp1, tmp2), 1)
        self.spin_down_mask = ~self.spin_up_mask

        self.pairwise_spin_up_mask, self.pairwise_spin_down_mask = \
            generate_pairwise_masks(n_electrons, n_pairwise, n_spin_up, n_spin_down, n_pairwise_features)

    # @tf.function
    def call(self, single, pairwise, n_samples, n_electrons):
        # single (n_samples, n_electrons, n_single_features)
        # pairwise (n_samples, n_electrons, n_pairwise_features)
        spin_up_mask = tf.tile(self.spin_up_mask, (n_samples, 1, 1))
        spin_down_mask = tf.tile(self.spin_down_mask, (n_samples, 1, 1))

        # --- Single summations
        replace = tf.zeros_like(single)
        # up
        sum_spin_up = tf.where(spin_up_mask, single, replace)
        sum_spin_up = tf.reduce_sum(sum_spin_up, 1, keepdims=True) / self.n_spin_up
        sum_spin_up = tf.tile(sum_spin_up, (1, n_electrons, 1))
        # down
        sum_spin_down = tf.where(spin_down_mask, single, replace)
        sum_spin_down = tf.reduce_sum(sum_spin_down, 1, keepdims=True) / self.n_spin_down
        sum_spin_down = tf.tile(sum_spin_down, (1, n_electrons, 1))

        # --- Pairwise summations
        sum_pairwise = tf.tile(tf.expand_dims(pairwise, 1), (1, n_electrons, 1, 1))
        replace = tf.zeros_like(sum_pairwise)
        # up
        sum_pairwise_up = tf.where(self.pairwise_spin_up_mask, sum_pairwise, replace)
        sum_pairwise_up = tf.reduce_sum(sum_pairwise_up, 2) / (self.n_spin_up - 1)
        # down
        sum_pairwise_down = tf.where(self.pairwise_spin_down_mask, sum_pairwise, replace)
        sum_pairwise_down = tf.reduce_sum(sum_pairwise_down, 2) / (self.n_spin_down - 1)

        features = tf.concat((single, sum_spin_up, sum_spin_down, sum_pairwise_up, sum_pairwise_down), 2)
        return features


def slogdet_keepdim(tensor):
    sign, tensor_out = tf.linalg.slogdet(tensor)
    tensor_out = tf.reshape(tensor_out, (*tensor_out.shape, 1, 1))
    sign = tf.reshape(sign, (*sign.shape, 1, 1))
    return sign, tensor_out


def generate_gamma(s):
    n_egvs = s.shape[2]
    gamma = [tf.reduce_prod(s[:, :, :i], axis=-1) * tf.reduce_prod(s[:, :, i+1:], axis=-1) for i in range(n_egvs-1)]
    gamma.append(tf.reduce_prod(s[:, :, :-1], axis=-1))
    gamma = tf.stack(gamma, axis=2)
    gamma = tf.expand_dims(gamma, axis=2)
    return gamma


def first_derivative_det(A):
    with tf.device("/cpu:0"):  # this is incredible stupid /// its actually not
        s, u, v = tf.linalg.svd(A, full_matrices=False)
    # s, u, v = tf.linalg.svd(A, full_matrices=False)
    v_t = tf.linalg.matrix_transpose(v)
    gamma = generate_gamma(s)
    sign = (tf.linalg.det(u) * tf.linalg.det(v))[..., None, None]
    out = sign * ((u * gamma) @ v_t)
    return out, (s, u, v_t, sign)


def generate_p(s):
    """
    :param s:
    :return:
    """
    n_samples, n_k, n_dim = s.shape
    new_shape = (1, 1, 1, n_dim, n_dim)
    s = s[..., None, None]
    s = tf.tile(s, new_shape)
    mask = np.ones(s.shape, dtype=np.bool)
    for i in range(n_dim):
        for j in range(n_dim):
            mask[..., i, i, j] = False
            mask[..., j, i, j] = False
    mask = tf.convert_to_tensor(mask)
    s = tf.where(mask, s, tf.ones_like(s, dtype=s.dtype))
    s_prod = tf.reduce_prod(s, axis=-3)
    s_prod = tf.linalg.set_diag(s_prod, tf.zeros((s_prod.shape[:-1]), dtype=s.dtype))
    return s_prod


def second_derivative_det(A, C_dash, *A_cache):
    """

    :param A:
    :param C_dash:
    :param A_cache:
    :return:
    """
    # This function computes the second order derivative of detA wrt to A
    # A matrix
    # C_bar backward sensitivity
    # A_cache cached values returned by grad_det(A)
    s, u, v_t, sign = A_cache  # decompose the cache

    M = v_t @ tf.linalg.matrix_transpose(C_dash) @ u

    p = generate_p(s)

    sgn = tf.math.sign(sign)

    m_jj = tf.linalg.diag_part(M)
    xi = -M * p

    xi_diag = p @ tf.expand_dims(m_jj, -1)
    xi = tf.linalg.set_diag(xi, tf.squeeze(xi_diag, -1))
    return sgn * u @ xi @ v_t


def k_sum(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)


def matrix_sum(x):
    return tf.reduce_sum(x, axis=[-2, -1], keepdims=True)


@tf.function
def _log_abs_sum_det_fwd(a, b, w):
    a = tf.stop_gradient(a)
    b = tf.stop_gradient(b)
    w = tf.stop_gradient(w)
    """

    :param a:
    :param b:
    :param w:
    :return:
    """
    # Take the slogdet of all k determinants
    sign_a, logdet_a = slogdet_keepdim(a)
    sign_b, logdet_b = slogdet_keepdim(b)

    x = logdet_a + logdet_b
    xmax = tf.math.reduce_max(x, axis=1, keepdims=True)

    unshifted_exp = sign_a * sign_b * tf.exp(x)
    unshifted_exp_w = w * unshifted_exp
    sign_unshifted_sum = tf.math.sign(tf.reduce_sum(unshifted_exp_w, axis=1, keepdims=True))

    exponent = x - xmax
    shifted_exp = sign_a * sign_b * tf.exp(exponent)

    u = w * shifted_exp
    u_sum = tf.reduce_sum(u, axis=1, keepdims=True)
    sign_shifted_sum = tf.math.sign(u_sum)
    log_psi = tf.math.log(tf.math.abs(u_sum)) + xmax

    # Both of these derivations appear to be valid
    # activations = shifted_exp
    # sensitivities = sign_unshifted_sum * tf.exp(-log_psi)
    # dw = sign_unshifted_sum * sign_a * sign_b * tf.exp(x-log_psi)
    #
    # return log_psi, sign_shifted_sum, activations, sensitivities, \
    #           (a, b, w, unshifted_exp, sign_unshifted_sum, dw, sign_a, logdet_a, sign_b, logdet_b, log_psi)

    sensitivities = tf.exp(xmax - log_psi) * sign_shifted_sum
    dw = sign_unshifted_sum * sign_a * sign_b * tf.exp(x - log_psi)

    return log_psi, sign_shifted_sum, shifted_exp, sensitivities, \
           (a, b, w, unshifted_exp, sign_unshifted_sum, dw, sign_a, logdet_a, sign_b, logdet_b, log_psi)

# @tf.function
def _log_abs_sum_det_first_order(*fwd_cache):
    """

    :param fwd_cache:
    :return:
    """
    a, b, w, unshifted_exp, sign_unshifted_sum, dw, sign_a, logdet_a, sign_b, logdet_b, log_psi = fwd_cache

    ddeta, ddeta_cache = first_derivative_det(a)
    ddetb, ddetb_cache = first_derivative_det(b)

    dfddeta = w * sign_unshifted_sum * sign_b * tf.exp(logdet_b - log_psi)
    dfddetb = w * sign_unshifted_sum * sign_a * tf.exp(logdet_a - log_psi)

    da = dfddeta * ddeta
    db = dfddetb * ddetb

    return (da, db, dw), (sign_unshifted_sum, ddeta, ddeta_cache, ddetb, ddetb_cache, dfddeta, dfddetb, da, db, dw)


# @tf.function
def _log_abs_sum_det_second_order(a_dash, b_dash, w_dash, *cache):
    """

    :param a_dash:
    :param b_dash:
    :param w_dash:
    :param cache:
    :return:
    """
    a, b, w, unshifted_exp, sign_unshifted_sum, _, sign_a, logdet_a, sign_b, logdet_b, log_psi, \
    sign_u, ddeta, ddeta_cache, ddetb, ddetb_cache, dfddeta, dfddetb, da, db, dw = cache

    dfddeta_w = dfddeta / w
    dfddetb_w = dfddetb / w

    ddeta_sum = matrix_sum(a_dash * ddeta)
    da_sum = matrix_sum(da * a_dash)
    ddetb_sum = matrix_sum(b_dash * ddetb)
    db_sum = matrix_sum(db * b_dash)
    a_sum = k_sum(dfddeta * ddeta_sum)
    b_sum = k_sum(dfddetb * ddetb_sum)

    # Compute second deriviate of f wrt to w
    d2w = w_dash * -dw * k_sum(dw)

    # compute deriviate of df/da wrt to w
    dadw = -dw * k_sum(da_sum)
    dadw += dfddeta_w * ddeta_sum  # i=j

    # compute derivative of df/db wrt to w
    dbdw = -dw * k_sum(db_sum)
    dbdw += dfddetb_w * ddetb_sum  # i=j

    # Compute second derivative of f wrt to a
    d2a = -da * a_sum
    d2a += dfddeta * second_derivative_det(a, a_dash, *ddeta_cache)  # i=j
    # Compute derivative of df/db wrt to a
    dbda = -da * b_sum
    dbda += ddeta * sign_u * tf.exp(-log_psi) * w * ddetb_sum  # i=j
    # Compute derivative of df/dw wrt to a
    dwda = w_dash * -da * k_sum(dw)
    dwda += w_dash * da / w  # i=j

    # Compute second derivative of f wrt to b
    d2b = -db * b_sum
    d2b += dfddetb * second_derivative_det(b, b_dash, *ddetb_cache)  # i=j
    # Compute derivative of df/da wrt to b
    dadb = -db * a_sum
    dadb += ddetb * sign_u * tf.exp(-log_psi) * w * ddeta_sum  # i=j
    # Compute derivative of df/dw wrt to b
    dwdb = w_dash * -db * k_sum(dw)
    dwdb += w_dash * db / w  # i=j

    return (d2a + dbda + dwda), (d2b + dadb + dwdb), \
           (d2w + dadw + dbdw), \
           None, None, None, None, None, None, None, None, None, None, None, None

# @tf.autograph.experimental.do_not_convert
# @tf.function
@tf.custom_gradient
def first_order_gradient(a_unused, b_unused, w_unused, *fwd_cache):
    """

    :param a_unused:
    :param b_unused:
    :param w_unused:
    :param fwd_cache:
    :return:
    """
    (da, db, dw), first_order_cache = _log_abs_sum_det_first_order(*fwd_cache)
    return (da, db, dw), \
           lambda a_dash, b_dash, w_dash: _log_abs_sum_det_second_order(
               a_dash, b_dash, w_dash, *fwd_cache, *first_order_cache)


# @tf.autograph.experimental.do_not_convert
# @tf.function
@tf.custom_gradient
def log_abs_sum_det(a, b, w):
    """

    :param a:
    :param b:
    :param w:
    :return:
    """
    log_psi, sign, act, sens, fwd_cache = _log_abs_sum_det_fwd(a, b, w)

    def _first_order_grad(dy, dsg, _, __):
        da, db, dw = first_order_gradient(a, b, w, *fwd_cache)

        # print('dy', dy)
        return dy * da, dy * db, tf.reduce_sum(dy * dw, axis=0, keepdims=True)
    # tf.reduce_sum(dy * dw, axis=0, keepdims=True)

    return (log_psi, sign, act, sens), _first_order_grad


def generate_pairwise_masks(n_electrons, n_pairwise, n_spin_up, n_spin_down, n_pairwise_features):
    eye_mask = ~np.eye(n_electrons, dtype=np.bool)
    ups = np.ones(n_electrons, dtype=np.bool)
    ups[n_spin_up:] = False
    downs = ~ups

    spin_up_mask = []
    spin_down_mask = []
    mask = np.zeros((n_electrons, n_electrons), dtype=np.bool)

    for electron in range(n_electrons):
        mask_up = np.copy(mask)
        mask_up[electron, :] = ups
        mask_up = mask_up[eye_mask]
        spin_up_mask.append(mask_up)

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        mask_down = mask_down[eye_mask]
        spin_down_mask.append(mask_down)

    spin_up_mask = tf.convert_to_tensor(spin_up_mask, dtype=tf.bool)
    # (n_samples, n_electrons, n_electrons, n_pairwise_features)
    spin_up_mask = tf.reshape(spin_up_mask, (1, n_electrons, n_pairwise, 1))
    spin_up_mask = tf.tile(spin_up_mask, (1, 1, 1, n_pairwise_features))

    spin_down_mask = tf.convert_to_tensor(spin_down_mask, dtype=tf.bool)
    spin_down_mask = tf.reshape(spin_down_mask, (1, n_electrons, n_pairwise, 1))
    spin_down_mask = tf.tile(spin_down_mask, (1, 1, 1, n_pairwise_features))

    return spin_up_mask, spin_down_mask
