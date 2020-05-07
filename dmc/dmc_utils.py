import tensorflow as tf
import tensorflow_probability as tfp
from dmc.hydrogen_demo import extract_grads
from energy.utils import compute_local_energy

class SampleDistribution:
    def __init__(self):
        pass

    def sample(self, shape):
        raise NotImplementedError()

    def resample(self, prev_sample):
        raise NotImplementedError()

class MVGaussian(tfp.distributions.MultivariateNormalFullCovariance, SampleDistribution):
    def __init__(self, mu, sigma):
        super(MVGaussian, self).__init__(mu, sigma)

    def resample(self, prev_sample, dtype=tf.float32):
        return self.sample(prev_sample.shape[:-1], dtype=dtype)

class RandomWalker(tfp.distributions.MultivariateNormalFullCovariance, SampleDistribution):
    def __init__(self, mu, sigma, step_mu=None, step_sigma=None):
        super(RandomWalker, self).__init__(mu, sigma)
        if step_mu is None:
            step_mu = mu
        if step_sigma is None:
            step_sigma = sigma
        self.step_gaussian = tfp.distributions.MultivariateNormalFullCovariance(step_mu, step_sigma)

    @tf.function
    def resample(self, prev_sample, dtype=tf.float32):
        return prev_sample + self.step_gaussian.sample(prev_sample.shape[:-1], dtype=dtype)


class run_DMC:
    def __init__(self, model,
                 pretrainer,
                 n_samples: int,
                 n_electrons: int,
                 correlation_length: int,
                 burn_batches: int,
                 distr: SampleDistribution,
                 dtype=tf.float32):

        super(run_DMC, self).__init__()
        self.correlation_length = correlation_length
        self.model = model
        self.distr = distr
        self.alpha_distr = tfp.distributions.Uniform(tf.cast(0, dtype), tf.cast(1, dtype))
        self.burn_batches = burn_batches
        self.pretrainer = pretrainer

        curr_sample = []
        while len(curr_sample) < n_samples:
            sample = self.distr.sample((1, n_electrons))
            amps, _, _ = model(sample)
            p = self.to_prob(amps)
            if p > 0.0:
                curr_sample.append(sample)
        self.curr_sample = tf.concat(curr_sample, axis=0)
        self.curr_log_amp, _, _ = self.model(self.curr_sample)
        self.curr_prob = self.to_prob(self.curr_log_amp)

    def sample(self, model, samples, r_atoms, z_atoms, n_samples):

        curr_sample = self.curr_sample
        curr_log_amp = self.curr_log_amp
        curr_prob = self.curr_prob

        acceptance_total = 0
        for _ in range(self.correlation_length):
            e_loc, _ = compute_local_energy(r_atoms, samples, z_atoms, model)  # r_atoms, r_electrons, z_atoms, model
            e_loc_centered = e_loc - tf.reduce_mean(e_loc)
            curr_grad = extract_grads(self.model, curr_sample, e_loc_centered, n_samples)

            # intermediate
            intermediate_sample = curr_sample + curr_grad

            # next sample
            new_sample = self.distr.resample(intermediate_sample)
            new_log_amp, _, _ = self.model(new_sample)
            new_prob = self.to_prob(new_log_amp)

            e_loc, _ = compute_local_energy(r_atoms, samples, z_atoms, model)  # r_atoms, r_electrons, z_atoms, model
            e_loc_centered = e_loc - tf.reduce_mean(e_loc)
            new_sample_grad = extract_grads(model, new_sample, e_loc_centered, n_samples)



            # update sample
            alpha = new_prob / curr_prob
            tf.debugging.check_numerics(alpha, 'houston, we have a problem')

            mask = alpha > self.alpha_distr.sample(alpha.shape)
            stacked_mask = tf.tile(tf.reshape(mask, (-1, 1, 1)), (1, *new_sample.shape[1:]))

            curr_sample = tf.where(stacked_mask, new_sample, curr_sample)
            curr_log_amp = tf.where(mask, new_log_amp, curr_log_amp)
            curr_prob = tf.where(mask, new_prob, curr_prob)

            acceptance_total += tf.reduce_mean(tf.cast(mask, dtype))

        self.curr_sample = curr_sample
        self.curr_log_amp = curr_log_amp
        self.curr_prob = curr_prob

        return curr_sample, curr_log_amp, acceptance_total / self.correlation_length

    @staticmethod
    @tf.function
    def to_prob(amp):
        return tf.exp(amp) ** 2

    # @tf.function # tries to compile burn_batches * correlation length loops
    def burn(self):
        for _ in range(self.burn_batches):
            self.sample()
