import tensorflow as tf

def expand(tensor, shape):
    return tf.reshape(tensor, (-1, *[1 for _ in shape[1:]]))

def extract_grads(model, inp, e_loc_centered, n_samples):
    with tf.GradientTape() as tape:
        out, _, _, _, _, _ = model(inp)
        loss = out * e_loc_centered
    grads = tape.gradient(loss, model.trainable_weights)
    return [grad / n_samples for grad in grads]

def extract_cv(name):
    return float(name.split('_')[-1].split(':')[0])

class KFAC_Actor():
    def __init__(self,
                 model,
                 n_spins,
                 conv_approx,
                 cov_weight,
                 should_center):

        self.layers = [w.name for w in model.trainable_weights]

        self.n_spins = n_spins
        self.cov_moving_weight = 0.95
        self.cov_weight = cov_weight
        self.cov_normalize = self.cov_moving_weight + self.cov_weight

        if conv_approx == 'mg':
            self.absorb_j = lambda x, cv: tf.reshape(x, (-1, *x.shape[2:])) if cv > 1. else x
            self.compute_normalization = lambda n_samples, conv_factor: float(n_samples * conv_factor)

        elif conv_approx == 'ba':
            self.absorb_j = lambda x, cv: tf.reduce_mean(x, axis=1) if cv > 1. else x
            self.compute_normalization = lambda n_samples, conv_factor: float(n_samples)

        self.m_aa = {}
        self.m_ss = {}

        self.m_aa_shapes, self.m_ss_shapes = self.extract_m_xx_shapes(model)

        for shape_a, shape_s, name in zip(self.m_aa_shapes, self.m_ss_shapes, self.layers):
            self.m_aa[name] = tf.ones(shape_a)
            self.m_ss[name] = tf.ones(shape_s)

        self.should_center = should_center  # this is true by default as it is correct, though we can change
        self.iteration = 0

    def extract_grads_and_a_and_s(self, model, inp, e_loc_centered, n_samples):

        with tf.GradientTape(True) as tape:
            out, _, activations, pre_activations, _, _ = model(inp)
            loss = out * e_loc_centered
            s_w = pre_activations[-1]
            pre_activations = [pa for pa in pre_activations[:-1]]

        grads = tape.gradient(loss, model.trainable_weights)
        grads = [grad / float(n_samples) for grad in grads]
        n_s, n_a = grads[-1].shape[:2]
        grads[-1] = tf.reshape(grads[-1], (n_a, n_s))

        sensitivities = tape.gradient(out, pre_activations)
        sensitivities.append(s_w)

        # loss_sensitivities = tape.gradient(loss, pre_activations)
        # loss_sensitivities.append(s_w * expand(e_loc_centered, s_w.shape))
        # for a, s, g, ls, name in zip(activations, sensitivities, grads, loss_sensitivities, self.layers):
        #     als = a
        #     conv_factor = float(name[-3])
        #     self.compute_comparison_gradient(als, ls, name, n_samples, g, conv_factor)

        for a, s, g, name in zip(activations, sensitivities, grads, self.layers):
            conv_factor = extract_cv(name)

            #print('xs')
            #print(a.shape)
            #print(s.shape)

            if self.should_center:  # d pfau said 'centering didnt have that much of an effect'
                s = self.center(s, conv_factor)
                a = self.center(a, conv_factor)

            a = self.absorb_j(a, conv_factor)  # couple different conv approx methods
            s = self.absorb_j(s, conv_factor)

            a = self.append_bias_if_needed(a, name)  # after the centering

            a = self.expand_dim_a(a, name)  # align the dimensions of the moving averages
            s = self.expand_dim_s(s, name)

            normalize = self.compute_normalization(n_samples, conv_factor)  # this is dependent on the conv approx

            aa = self.outer_product_and_sum(a, name) / normalize
            ss = self.outer_product_and_sum(s, name) / normalize

            #print('xxs')
            #print(aa.shape)
            #print(ss.shape)

            self.update_m_aa_and_m_ss(aa, ss, name, self.iteration)

            #print('outs')
            # print(self.m_aa[name].shape, tf.reduce_mean(tf.abs(self.m_aa[name].numpy())))
            # print(self.m_ss[name].shape, tf.reduce_mean(tf.abs(self.m_ss[name].numpy())))
            #print(g.shape)

        self.iteration += 1
        return grads, self.m_aa, self.m_ss

    def compute_comparison_gradient(self, als, ls, name, n_samples, g, conv_factor):
        als = self.absorb_j(als, name)
        ls = self.absorb_j(ls, name)
        als = self.append_bias_if_needed(als, name)
        als = self.expand_dim_a(als, name)
        ls = self.expand_dim_s(ls, name)
        outer_product = tf.linalg.matmul(tf.expand_dims(als, -1), tf.expand_dims(ls, -2))
        outer_product = tf.reduce_sum(outer_product, axis=0)
        g_new = outer_product / n_samples
        print(g.shape, g_new.shape)
        assert g.shape == g_new.shape
        validate = tf.reduce_mean(tf.abs(g - g_new))
        print(validate)

    # m_xx = (cov_moving_weight * m_xx + cov_weight * xx)  / normalization
    def update_m_aa_and_m_ss(self, aa, ss, name, iteration):
        cov_moving_weight = tf.minimum(1. - (1 / (1 + iteration)), self.cov_moving_weight)
        cov_weight = self.cov_weight
        self.cov_normalize = cov_weight + cov_moving_weight
        #print(cov_moving_weight, 'cov_moving_weight')
        # cov_moving_weight = self.cov_moving_weight
        # cov_weight = self.cov_weight
        # tensorflow and or ray has a weird thing about inplace operations??
        self.m_aa[name] *= cov_moving_weight / self.cov_normalize  # multiply
        self.m_aa[name] += (cov_weight * aa) / self.cov_normalize  # add

        self.m_ss[name] *= cov_moving_weight / self.cov_normalize
        self.m_ss[name] += (cov_weight * ss) / self.cov_normalize
        return

    @staticmethod
    def center(x, cv):
        if cv > 1.:
            xc = tf.reduce_mean(x, axis=[0, 1], keepdims=True)
        else:
            xc = tf.reduce_mean(x, axis=0, keepdims=True)
        return x - tf.reduce_mean(xc, axis=0, keepdims=True)

    # stream 'njf, fs -> njs' + append bias
    # env_w 'njf,kifs->njkis' + append bias
    # env_sigma 'njmv,kimvc->njkimc'
    # env_pi 'njkim,kims->njkis'
    # w 'na, nas -> ns'

    @staticmethod
    def outer_product_and_sum(x, name):
        if 'stream' in name:
            return tf.linalg.matmul(x, x, transpose_a=True)
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
        if 'wf' in name:
            return tf.expand_dims(tf.squeeze(s), -1)
        return s

    @staticmethod
    def expand_dim_a(a, name):
        if 'env' in name:
            if '_w_' in name:
                return tf.expand_dims(tf.expand_dims(a, 1), 1)
            elif 'sigma' in name:
                return tf.expand_dims(tf.expand_dims(a, 1), 1)
        if 'wf' in name:
            return tf.squeeze(a)
        return a  # stream, pi

    def extract_m_xx_shapes(self, model):
        _, _, activations, pre_activations, _, _ = model(tf.random.uniform((1, self.n_spins, 3)))

        a_shapes, s_shapes = [], []
        for a, s, name in zip(activations, pre_activations, self.layers):
            conv_factor = extract_cv(name)

            a = self.absorb_j(a, conv_factor)
            s = self.absorb_j(s, conv_factor)

            a = self.append_bias_if_needed(a, name)

            a = self.expand_dim_a(a, name)
            s = self.expand_dim_s(s, name)

            aa = self.outer_product_and_sum(a, name)
            ss = self.outer_product_and_sum(s, name)

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
                 damping_method,
                 conv_approx,
                 ft_method):

        self.lr0 = lr0
        self.lr = lr0
        self.decay = decay

        self.layers = layers
        self.damping = initial_damping
        self.norm_constraint = norm_constraint

        if conv_approx == 'ba':
            self.compute_conv_factor = lambda x: x**2
        else:
            self.compute_conv_factor = lambda x: x

        if damping_method == 'tikhonov':
            self.ops = Tikhonov()
        elif damping_method == 'ft':
            self.ops = FactoredTikhonov(ft_method)

        self.iteration = 0

    def compute_updates(self, grads, m_aa, m_ss, iteration):
        self.lr = self.compute_lr(iteration)
        # print('lr:', self.lr)
        nat_grads = []
        for g, name in zip(grads, self.layers):
            conv_factor = self.compute_conv_factor(extract_cv(name))
            maa = m_aa[name]
            mss = m_ss[name]

            damping = self.damping / (1 + self.decay * 100 * iteration)
            ng = self.ops.compute_nat_grads(maa, mss, g, conv_factor, self.damping, name, iteration)

            nat_grads.append(ng)

        eta = self.compute_norm_constraint(nat_grads, grads)

        self.iteration += 1
        return [eta * ng for ng in nat_grads]

    def compute_norm_constraint(self, nat_grads, grads):
        sq_fisher_norm = 0
        for ng, g in zip(nat_grads, grads):
            sq_fisher_norm += tf.reduce_sum(ng * g)
        self.eta = tf.minimum(1., tf.sqrt(self.norm_constraint / (self.lr**2 * sq_fisher_norm)))
        tf.summary.scalar('kfac/sq_fisher_norm', sq_fisher_norm, self.iteration)
        tf.summary.scalar('kfac/eta', self.eta, self.iteration)
        return self.eta

    def compute_lr(self, iteration):
        return self.lr0 / (1 + self.decay * iteration)



class Tikhonov():
    def __init__(self):
        print('Tikhonov damping')

    def compute_nat_grads(self,  maa, mss, g, conv_factor, damping, layer, iteration):
        vals_a, vecs_a, vals_s, vecs_s = self.compute_eig_decomp(maa, mss)
        #print('eigs')
        #print(vals_a.shape, vecs_a.shape, g.shape, vals_s.shape, vecs_s.shape)

        v1 = tf.linalg.matmul(vecs_a, g / conv_factor, transpose_a=True) @ vecs_s
        divisor = tf.expand_dims(vals_s, -2) * tf.expand_dims(vals_a, -1)
        v2 = v1 / (divisor + damping / conv_factor)
        ng = vecs_a @ tf.linalg.matmul(v2, vecs_s, transpose_b=True)

        return ng

    def compute_eig_decomp(self, maa, mss):
        # get the eigenvalues and eigenvectors of a symmetric positive matrix
        with tf.device("/cpu:0"):
            vals_a, vecs_a = tf.linalg.eigh(maa)
            vals_s, vecs_s = tf.linalg.eigh(mss)

        # zero negative eigenvalues. eigh outputs VALUES then VECTORS
        # print('zero')
        # print(vals_a.shape, vecs_a.shape)
        vals_a = tf.maximum(vals_a, tf.zeros_like(vals_a))
        vals_s = tf.maximum(vals_s, tf.zeros_like(vals_s))

        return vals_a, vecs_a, vals_s, vecs_s


class FactoredTikhonov():
    def __init__(self, ft_method):
        print('Factored Tikhonov damping')
        self.ft_method = ft_method

    def compute_eig_decomp(self, maa, mss):
        # get the eigenvalues and eigenvectors of a symmetric positive matrix
        with tf.device("/cpu:0"):
            vals_a, vecs_a = tf.linalg.eigh(maa)
            vals_s, vecs_s = tf.linalg.eigh(mss)

        # zero negative eigenvalues. eigh outputs VALUES then VECTORS
        vals_a = tf.maximum(vals_a, tf.zeros_like(vals_a))
        vals_s = tf.maximum(vals_s, tf.zeros_like(vals_s))

        return vals_a, vecs_a, vals_s, vecs_s

    def compute_nat_grads(self,  maa, mss, g, conv_factor, damping, layer, iteration):

        maa, mss = self.damp(maa, mss, conv_factor, damping, layer, iteration)

        vals_a, vecs_a, vals_s, vecs_s = self.compute_eig_decomp(maa, mss)

        v1 = tf.linalg.matmul(vecs_a, g / conv_factor, transpose_a=True) @ vecs_s
        divisor = tf.expand_dims(vals_s, -2) * tf.expand_dims(vals_a, -1)
        v2 = v1 / divisor
        ng = vecs_a @ tf.linalg.matmul(v2, vecs_s, transpose_b=True)

        return ng

    def damp(self, m_aa, m_ss, conv_factor, damping, name, iteration):  # factored tikhonov damping
        dim_a = m_aa.shape[-1]
        dim_s = m_ss.shape[-1]
        batch_shape = list((1 for _ in m_aa.shape[:-2]))  # needs to be cast as list or disappears in tf.eye

        if self.ft_method == 'alternate':
            if 'stream' not in name and '_w_' not in name:
                pi = tf.expand_dims(tf.expand_dims(tf.ones(batch_shape), -1), -1)
            else:
                tr_a = self.get_tr_norm(m_aa)
                tr_s = self.get_tr_norm(m_ss)
                pi = tf.expand_dims(tf.expand_dims((tr_a * dim_s) / (tr_s * dim_a), -1), -1)
        elif self.ft_method == 'ones_pi':
            pi = tf.expand_dims(tf.expand_dims(tf.ones(batch_shape), -1), -1)
        else:
            tr_a = self.get_tr_norm(m_aa)
            tr_s = self.get_tr_norm(m_ss)
            pi = tf.expand_dims(tf.expand_dims((tr_a * dim_s) / (tr_s * dim_a), -1), -1)


        # tf.summary.scalar('damping/pi_%s' % name, tf.reduce_mean(pi), iteration)
        # tf.debugging.check_numerics(pi, 'pi')

        eye_a = tf.eye(dim_a, batch_shape=batch_shape)
        eye_s = tf.eye(dim_s, batch_shape=batch_shape)

        eps = 1e-8
        m_aa_damping = tf.sqrt(pi * damping / conv_factor)
        # m_aa_damping = tf.maximum(eps * tf.ones_like(m_aa_damping), m_aa_damping)

        m_ss_damping = tf.sqrt(damping / (pi * conv_factor))
        # m_ss_damping = tf.maximum(eps * tf.ones_like(m_ss_damping), m_ss_damping)

        # print('damping')
        # print(m_aa.shape, eye_a.shape, m_aa_damping.shape)
        # print(m_ss.shape, eye_s.shape, m_ss_damping.shape)

        m_aa += eye_a * m_aa_damping
        m_ss += eye_s * m_ss_damping
        return m_aa, m_ss

    @staticmethod
    def get_tr_norm(m_xx):
        trace = tf.linalg.trace(m_xx)
        return tf.maximum(1e-10 * tf.ones_like(trace), trace)















