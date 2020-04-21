import tensorflow as tf
import tensorflow_probability as tfp
from utils.utils import tofloat
import numpy as np

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
    def resample(self, prev_sample):
        return prev_sample + self.step_gaussian.sample(prev_sample.shape[:-1], dtype=tf.float32)


class MetropolisHasting:
    def __init__(self,
                 model,
                 pretrainer,
                 distr: SampleDistribution,
                 n_samples: int,
                 n_electrons: int,
                 correlation_length: int,
                 burn_batches: int,
                 n_atoms: int,
                 atom_positions: list,
                 ne_atoms: list,
                 n_spin_up: int):

        # super(MetropolisHasting, self).__init__()
        self.n_samples = n_samples
        self.n_electrons = n_electrons
        self.correlation_length = correlation_length
        self.model = model
        self.distr = distr
        self.alpha_distr = tfp.distributions.Uniform(tf.cast(0, tf.float32), tf.cast(1, tf.float32))
        self.burn_batches = burn_batches
        self.pretrainer = pretrainer
        self.n_atoms = n_atoms
        self.atom_positions = atom_positions
        self.ne_atoms = ne_atoms
        self.n_spin_up = n_spin_up

    def initialize_samples(self):
        ups = []
        downs = []
        for ne_atom, atom_position in zip(self.ne_atoms, self.atom_positions):
            for e in range(ne_atom):
                if e % 2 == 0:
                    curr_sample_up = np.random.normal(loc=atom_position, scale=0.5, size=(self.n_samples, 1, 3))
                    ups.append(curr_sample_up)
                else:
                    curr_sample_down = np.random.normal(loc=atom_position, scale=0.5, size=(self.n_samples, 1, 3))
                    downs.append(curr_sample_down)
        ups = np.concatenate(ups, axis=1)
        downs = np.concatenate(downs, axis=1)
        curr_sample = np.concatenate([ups, downs], axis=1)
        curr_sample = tf.convert_to_tensor(curr_sample, dtype=tf.float32)
        return curr_sample

    @tf.function
    def sample(self, curr_sample):

        curr_log_amp, _, _, _, _ = self.model(curr_sample)
        curr_prob = MetropolisHasting.to_prob(curr_log_amp)

        acceptance_total = 0.
        for _ in tf.range(self.correlation_length):

            # next sample
            new_sample = self.distr.resample(curr_sample)
            new_log_amp, _, _, _, _ = self.model(new_sample)
            new_prob = MetropolisHasting.to_prob(new_log_amp)

            # update sample
            alpha = new_prob / curr_prob

            mask = alpha > self.alpha_distr.sample(alpha.shape)
            stacked_mask = tf.tile(tf.reshape(mask, (-1, 1, 1)), (1, *new_sample.shape[1:]))

            curr_sample = tf.where(stacked_mask, new_sample, curr_sample)
            curr_log_amp = tf.where(mask, new_log_amp, curr_log_amp)
            curr_prob = tf.where(mask, new_prob, curr_prob)

            acceptance_total += tf.reduce_mean(tf.cast(mask, tf.float32))

        return curr_sample, curr_log_amp, acceptance_total / tofloat(self.correlation_length)

    @tf.function
    def sample_mixed_dist(self, curr_sample):

        # curr_sample = self.curr_sample
        curr_log_amp, _, _, _, _ = self.model(curr_sample)
        curr_prob = MetropolisHasting.to_prob(curr_log_amp)
        hf_prob = self.pretrainer.compute_det_probability(curr_sample)  # call numpy to take out of graph
        curr_prob = 0.5*(curr_prob + hf_prob)

        acceptance_total = 0.
        for _ in range(self.correlation_length):
            # next sample
            new_sample = self.distr.resample(curr_sample)
            new_log_amp, _, _, _, _ = self.model(new_sample)
            hf_prob = self.pretrainer.compute_det_probability(new_sample)  # call numpy to take out of graph
            new_prob = 0.5*(MetropolisHasting.to_prob(new_log_amp) + hf_prob)

            # update sample
            alpha = new_prob / curr_prob
            # tf.debugging.check_numerics(alpha, 'houston, we have a problem')

            mask = alpha > self.alpha_distr.sample(alpha.shape)
            stacked_mask = tf.tile(tf.reshape(mask, (-1, 1, 1)), (1, *new_sample.shape[1:]))

            curr_sample = tf.where(stacked_mask, new_sample, curr_sample)
            curr_log_amp = tf.where(mask, new_log_amp, curr_log_amp)
            curr_prob = tf.where(mask, new_prob, curr_prob)

            acceptance_total += tf.reduce_mean(tf.cast(mask, tf.float32))

        return curr_sample, curr_log_amp, acceptance_total / tofloat(self.correlation_length)

    @tf.function
    def sample_mixed(self, curr_sample, floor=1e-10):

        sams = tf.split(curr_sample, [self.n_samples//2, self.n_samples//2])
        curr_sample_model, curr_sample_hf = tf.squeeze(sams[0]), tf.squeeze(sams[1])

        # curr_sample = self.curr_sample
        curr_log_amp, _, _, _, _ = self.model(curr_sample_model)
        curr_prob_model = MetropolisHasting.to_prob(curr_log_amp)
        curr_prob_hf = self.pretrainer.compute_orbital_probability(curr_sample_hf)  # call numpy to take out of graph

        acceptance_total = 0.
        for _ in range(self.correlation_length):
            # --- next sample
            new_sample_model = self.distr.resample(curr_sample_model)
            new_log_amp, _, _, _, _ = self.model(new_sample_model)
            new_prob_model = MetropolisHasting.to_prob(new_log_amp)

            new_sample_hf = self.distr.resample(curr_sample_hf)
            new_prob_hf = self.pretrainer.compute_orbital_probability(new_sample_hf)  # call numpy to take out of graph

            # --- update sample
            alpha_model = new_prob_model / (curr_prob_model+floor)
            alpha_hf = new_prob_hf / (curr_prob_hf+floor)

            tf.debugging.check_numerics(alpha_model, 'houston, we have a problem1')
            tf.debugging.check_numerics(alpha_hf, 'houston, we have a problem2')

            # --- generate masks
            mask_model = alpha_model > self.alpha_distr.sample(alpha_model.shape)
            stacked_mask_model = tf.tile(tf.reshape(mask_model, (-1, 1, 1)), (1, *new_sample_model.shape[1:]))

            mask_hf = alpha_hf > self.alpha_distr.sample(alpha_model.shape)  # can't use alpha_hf.shape here because it comes from pyscf
            stacked_mask_hf = tf.tile(tf.reshape(mask_hf, (-1, 1, 1)), (1, *new_sample_hf.shape[1:]))

            # --- replace samples
            curr_sample_model = tf.where(stacked_mask_model, new_sample_model, curr_sample_model)
            curr_log_amp = tf.where(mask_model, new_log_amp, curr_log_amp)
            curr_prob_model = tf.where(mask_model, new_prob_model, curr_prob_model)

            curr_sample_hf = tf.where(stacked_mask_hf, new_sample_hf, curr_sample_hf)
            curr_prob_hf = tf.where(mask_hf, new_prob_hf, curr_prob_hf)

            acceptance_total += tf.reduce_mean(tf.cast(mask_model, tf.float32))
            acceptance_total += tf.reduce_mean(tf.cast(mask_hf, tf.float32))

        curr_sample = tf.concat([curr_sample_model, curr_sample_hf], axis=0)

        return curr_sample, curr_log_amp, acceptance_total / tofloat(self.correlation_length)

    @staticmethod
    @tf.function
    def to_prob(amp):
        return tf.exp(amp) ** 2

    # @tf.function # tries to compile burn_batches * correlation length loops
    def burn(self, samples):
        for _ in range(self.burn_batches):
            samples, _, _ = self.sample(samples)
        return samples

    def burn_pretrain(self, samples):
        for _ in range(self.burn_batches):
            samples, _, _ = self.sample_mixed(samples)
        return samples
