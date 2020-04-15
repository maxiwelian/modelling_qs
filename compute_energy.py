import tensorflow as tf

from utils import save_pk, compute_rolling_mean, compute_rolling_std
from actor import burn, get_energy

def compute_energy(models, config):
    burn(models, 10)
    e_locs = []
    e_overall_mean = 0
    e_overall_std = 0
    for batch in range(config['n_batch_compute_energy']):
        e_loc, e_mean, e_std = get_energy(models)

        e_locs.append(e_mean)
        burn(models, 1)
        rolling_mean = compute_rolling_mean(e_overall_mean, batch, e_mean)
        rolling_std = compute_rolling_std(e_overall_std, e_overall_mean, batch, e_std, e_mean)
        e_overall_mean = tf.reduce_mean(e_locs)

        print('overall: ', e_overall_mean.numpy(),
              'rolling_mean: ', rolling_mean,
              'n_samples: ', config['n_samples'] * (batch+1),
              'e_std: ', e_std.numpy(),
              'rolling_std: ', rolling_std)

        print('rolling mean: ', rolling_mean)
    config['e_mean_final'] = e_overall_mean.numpy()
    config['e_std_final'] = e_std.numpy()
    save_pk(config['config_path'], config)
    return