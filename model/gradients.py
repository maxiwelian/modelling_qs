import tensorflow as tf


def extract_grads(model, inp, e_loc_centered, n_samples):
    with tf.GradientTape() as tape:
        out, _, _, _, _ = model(inp)
        loss = out * e_loc_centered
    grads = tape.gradient(loss, model.trainable_weights)
    return [grad / n_samples for grad in grads]


class KFAC_Actor():
    def __init__(self,
                 model,
                 conv_approx):

        self.layers = [w.name for w in model.trainable_weights]

        self.n_spins = 4
        self.cov_moving_weight = 0.95
        self.cov_weight = 0.05
        self.cov_normalize = self.cov_moving_weight + self.cov_weight

        if conv_approx == 'mg':
            self.absorb_j = lambda x, n: tf.reshape(x, (-1, *x.shape[2:])) if (float(n[-3]) > 1.) else x
            self.compute_normalization = lambda n_samples, conv_factor: n_samples * conv_factor

        elif conv_approx == 'ba':
            self.absorb_j = lambda x, n: tf.reduce_mean(x, axis=1) if (float(n[-3]) > 1.) else x
            self.compute_normalization = lambda n_samples, conv_factor: n_samples

        self.m_aa = {}
        self.m_ss = {}

        m_aa_shapes, m_ss_shapes = self.extract_m_xx_shapes(model)

        for shape_a, shape_s, name in zip(m_aa_shapes, m_ss_shapes, self.layers):
            self.m_aa[name] = tf.ones(shape_a)
            self.m_ss[name] = tf.ones(shape_s)

        self.should_center = True  # this is true by default as it is correct, though we can change

    def extract_grads_and_a_and_s(self, model, inp, e_loc_centered, n_samples):
        with tf.GradientTape(True) as tape:
            out, activations, pre_activations, _, _ = model(inp)
            loss = out * e_loc_centered

        grads = tape.gradient(loss, model.trainable_weights)
        grads = [grad / n_samples for grad in grads]

        n_s, n_a = grads[-1].shape[:2]
        grads[-1] = tf.reshape(grads[-1], (n_a, n_s))

        s_w = pre_activations[-1]
        pre_activations = [pa for pa in pre_activations[:-1]]
        sensitivities = tape.gradient(out, pre_activations)
        sensitivities.append(s_w)

        for a, s, g, name in zip(activations, sensitivities, grads, self.layers):
            conv_factor = float(name[-3])

            if self.should_center:  # d pfau said 'centering didnt have that much of an effect'
                a = self.center(a)
                s = self.center(s)

            a = self.absorb_j(a, name)  # couple different conv approx methods
            s = self.absorb_j(s, name)

            a = self.append_bias_if_needed(a, name)  # after the centering

            a = self.expand_dim_a(a, name)  # align the dimensions of the moving averages
            s = self.expand_dim_s(s, name)

            normalize = self.compute_normalization(n_samples, conv_factor)  # this is dependent on the conv approx

            aa = self.outer_product_and_sum(a) / normalize
            ss = self.outer_product_and_sum(s) / normalize

            # assert len(aa.shape[:-2]) == len(ss.shape[:-2]) == len(g.shape[:-2])

            self.update_m_aa_and_m_ss(aa, ss, name)

        return grads, self.m_aa, self.m_ss

    def update_m_aa_and_m_ss(self, aa, ss, name):
        # m_xx = (cov_moving_weight * m_xx + cov_weight * xx)  / normalization
        self.m_aa[name] *= self.cov_moving_weight / self.cov_normalize  # multiply
        self.m_aa[name] += (self.cov_weight * aa) / self.cov_normalize  # add

        self.m_ss[name] *= self.cov_moving_weight / self.cov_normalize
        self.m_ss[name] += (self.cov_weight * ss) / self.cov_normalize
        return

    @staticmethod
    def center(x):
        return x - tf.reduce_mean(x, axis=0)

    # stream 'njf, fs -> njs' + append bias
    # env_w 'njf,kifs->njkis' + append bias
    # env_sigma 'njmv,kimvc->njkimc'
    # env_pi 'njkim,kims->njkis'
    # w 'na, nas -> ns'

    @staticmethod
    def outer_product_and_sum(x):
        outer_product = tf.linalg.matmul(tf.expand_dims(x, -1), tf.expand_dims(x, -2))
        return tf.reduce_sum(outer_product, axis=0)

    @staticmethod
    def append_bias_if_needed(a, name):
        if 'stream' in name or '_w_' in name:  # for streams or final linear transform in env
            bias = tf.ones((*a.shape[:-1], 1))
            return tf.concat((a, bias), axis=-1)
        return a

    @staticmethod
    def expand_dim_s(s, name):
        if 'w_' in name:
            return tf.expand_dims(tf.squeeze(s), -1)
        return s

    @staticmethod
    def expand_dim_a(a, name):
        if 'env' in name:
            if '_w_' in name:
                return tf.expand_dims(tf.expand_dims(a, 1), 1)
            elif 'sigma' in name:
                return tf.expand_dims(tf.expand_dims(a, 1), 1)
        if 'w_' in name:
            return tf.squeeze(a)
        return a  # stream, pi, w

    def extract_m_xx_shapes(self, model):
        _, activations, pre_activations, _, _ = model(tf.random.uniform((1, self.n_spins, 3)))

        a_shapes, s_shapes = [], []
        for a, s, name in zip(activations, pre_activations, self.layers):
            a = self.absorb_j(a, name)
            s = self.absorb_j(s, name)

            a = self.append_bias_if_needed(a, name)

            a = self.expand_dim_a(a, name)
            s = self.expand_dim_s(s, name)

            aa = self.outer_product_and_sum(a)
            ss = self.outer_product_and_sum(s)

            a_shapes.append(aa.shape)
            s_shapes.append(ss.shape)
        return a_shapes, s_shapes

# stream 'njf, fs -> njs' + append bias
# env_w 'njf,kifs->nkjis' + append bias
# env_sigma 'njmv,kimvc->nkjimc'
# env_pi 'nkjim,kims->nkjis'
# w 'na, nas -> ns'
class KFAC():
    def __init__(self,
                 layers,
                 lr0,
                 decay,
                 initial_damping,
                 norm_constraint,
                 damping_method):

        self.lr0 = lr0
        self.decay = decay

        self.layers = layers
        self.damping = initial_damping
        self.norm_constraint = norm_constraint

        if damping_method == 'tikhonov':
            self.damp = lambda x, y, z: (x, y)  # do nothing
            self.decomp_damping = lambda conv_factor: self.damping / conv_factor

        elif damping_method == 'ft':  # factored tikhonov
            self.damp = self.ft_damp
            self.decomp_damping = lambda conv_factor: 0.  # add zero in the inversion as damping is done before

    def compute_updates(self, grads, m_aa, m_ss, iteration):
        lr = self.compute_lr(iteration)

        nat_grads = []
        for g, layer in zip(grads, self.layers):
            conv_factor = float(layer[-3])
            maa = m_aa[layer]
            mss = m_ss[layer]

            maa, mss = self.damp(maa, mss, conv_factor)

            vals_a, vecs_a, vals_s, vecs_s = self.compute_eig_decomp(maa, mss)

            decomp_damping = self.decomp_damping(conv_factor)  # calls depending on the damping method

            # T F + \lambda I = T (F + \lambda I / (T))
            ng = self.compute_nat_grads(vals_a, vecs_a, vals_s, vecs_s, g, decomp_damping) / conv_factor

            nat_grads.append(ng)

        eta = self.compute_norm_constraint(nat_grads, grads, lr)

        return [-1. * eta * lr * ng for ng in nat_grads]

    def compute_norm_constraint(self, nat_grads, grads, lr):
        sq_fisher_norm = 0
        for ng, g in zip(nat_grads, grads):
            sq_fisher_norm += tf.reduce_sum(ng * g)

        eta = tf.minimum(1., tf.sqrt(self.norm_constraint / (lr**2 * sq_fisher_norm)))
        return eta

    def compute_eig_decomp(self, maa, mss):
        # get the eigenvalues and eigenvectors of a symmetric positive matrix

        # enforce symmetry (there may be numerical errors)
        maa = (tf.linalg.matrix_transpose(maa) + maa) / 2.
        mss = (tf.linalg.matrix_transpose(mss) + mss) / 2.

        with tf.device("/cpu:0"):
            vals_a, vecs_a = tf.linalg.eigh(maa)
            vals_s, vecs_s = tf.linalg.eigh(mss)

        # zero negative eigenvalues. eigh outputs VALUES then VECTORS
        vals_a = tf.maximum(vals_a, tf.zeros_like(vals_a))
        vals_s = tf.maximum(vals_s, tf.zeros_like(vals_s))

        return vals_a, vecs_a, vals_s, vecs_s

    def compute_nat_grads(self, vals_a, vecs_a, vals_s, vecs_s, grad, normalized_damping):
        # apply tikhonov damping here using the expression from appendix B https://arxiv.org/pdf/1503.05671.pdf
        # don't include the fisher information matrix scaling in the damping (is absorbed into lr)
        # 4 T F + \lambda I = 4 T (F + \lambda I / (4 * T))

        v1 = tf.linalg.matmul(vecs_a, grad, transpose_a=True) @ vecs_s
        divisor = tf.expand_dims(vals_s, -2) * tf.expand_dims(vals_a, -1)
        v2 = v1 / (divisor + normalized_damping)
        v3 = vecs_a @ tf.linalg.matmul(v2, vecs_s, transpose_b=True)
        return v3

    def compute_lr(self, iteration):
        return self.lr0 / (1 + self.decay * iteration)

    def ft_damp(self, m_aa, m_ss, conv_factor):  # factored tikhonov damping
        dim_a = m_aa.shape[-1]
        dim_s = m_ss.shape[-1]
        batch_shape = list((1 for _ in m_aa.shape[:-2]))  # needs to be cast as list or disappears in tf.eye

        tr_a = tf.linalg.trace(m_aa)
        tr_s = tf.linalg.trace(m_ss)

        pi = tf.expand_dims(tf.expand_dims((tr_a * dim_s) / (tr_s * dim_a), -1), -1)

        eye_a = tf.eye(dim_a, batch_shape=batch_shape)
        eye_s = tf.eye(dim_s, batch_shape=batch_shape)

        m_aa += eye_a * tf.sqrt(pi * self.damping / conv_factor)
        m_ss += eye_s * tf.sqrt(self.damping / (pi * conv_factor))

        return m_aa, m_ss













