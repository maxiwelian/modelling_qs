import tensorflow as tf
import pickle as pk
import numpy as np
import inspect
import os
from time import time
import re
from tensorflow.python.client import device_lib
import shutil
from actor_proxy import get_info
import ray

def count_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpus)

def print_neat(str, tensor, prec=2):
    print(str, round(float(tensor), prec))


def print_iteration(printer, epoch):
    for key, val in printer.items():
        try:
            print_neat(key, val[0], val[1])
        except:
            pass
    print('\n')


def log_iteration(printer, epoch):
    for key, val in printer.items():
        try:
            tf.summary.scalar(key, tf.reshape(val[0], []), epoch)
        except:
            pass

    tf.summary.flush()
    print('\n')


def print_profile(timer):
    total = 0
    for key, val in timer.items():
        total += val
    for key, val in timer.items():
        print('percentage in %s = %.2f' % (key, val / total))


def load_models(models, path):
    models[0].load_model.remote(path=path)
    weights_id = models[0].get_weights.remote()
    [model.set_weights.remote(weights_id) for model in models[1:]]


def load_model(model, path):
    with open(path, 'rb') as f:
        weights = pk.load(f)
    lst_weights = []
    for i in range(len(weights)):
        lst_weights.append(tf.convert_to_tensor(weights[i]))
    model.set_weights(lst_weights)
    print('model loaded!!')
    return


def save_model(model_weights, path):
    weights = {}
    for idx, w in enumerate(model_weights):
        weights[idx] = w
    with open(path, 'wb') as f:
        pk.dump(weights, f)
    print('model saved!!')
    return


def save_samples(samples, path):
    if type(samples) == list:
        samples = [sample.numpy() for sample in samples]
        samples = np.concatenate(samples, axis=0)
    else:
        samples = samples.numpy()

    with open(path, 'wb') as f:
        pk.dump(samples, f)
    print('samples saved!!')


def load_samples(models, path):
    [model.load_samples.remote(path=path) for model in models]


def load_sample(path):
    with open(path, 'rb') as f:
        samples = pk.load(f)
    return samples


def filter_dict(dictionary, fn):
    new_dict = {k: v for k, v in dictionary.items() if
     k in [p.name for p in inspect.signature(fn).parameters.values()]}
    return new_dict


def tofloat(x):
    return tf.cast(x, tf.float32)


def mean_list_of_dicts(list_dict):
    n_mean = len(list_dict)
    new_dict = {}
    for i in range(n_mean):
        new_dict = {key: value / n_mean for key, value in list_dict[i].items}
    return new_dict

def print_config(config):
    for key, value in config.items():
        print(key, '       ', value)


def save_kfac_state(kfac, epoch, n_layers, kfac_directory):
    # count the current saved epochs
    saved_epochs = os.listdir(kfac_directory)
    n_saved = len(saved_epochs)

    # delete the minimum if there are too many
    if n_saved > 3:
        min_epoch = min([int(name[1:]) for name in saved_epochs])
        path = kfac_directory + '/e{}'.format(min_epoch)
        shutil.rmtree(path)

    # create and save the new epoch
    path = kfac_directory + '/e{}'.format(epoch)
    if not os.path.exists(path):
        os.makedirs(path)

    for layer_id in range(n_layers):
        kfac[layer_id].save(epoch)


def find_latest_model(directory):
    files = os.listdir(directory)

    epochs = [0]  # in case it exists but doesn't contain anything
    for file in files:
        numbers = [int(float(s)) for s in re.findall(r'-?\d+\.?\d*', file)]
        if len(numbers) > 0:
            epoch = max(numbers)
            # print(epoch)
            epochs.append(epoch)
    # print(epochs)
    # print(max(epochs))
    if max(epochs) < 1000:
        return 0
    return max(epochs)

def save_pk(path, x):
    with open(path, 'wb') as f:
        pk.dump(x, f)

def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x

def compute_rolling_mean(current_mean, batch, new_mean):
    batch_plus_one = batch + 1.
    rolling_mean = (batch / batch_plus_one) * current_mean + (1. / batch_plus_one) * new_mean
    return rolling_mean.numpy()

def compute_rolling_std(current_std, current_mean, batch, new_std, new_mean):
    batch_plus_one = batch+1.
    current_var = current_std**2
    new_var = new_std**2
    lc = (batch / batch_plus_one)
    rc = (1. / batch_plus_one)
    rolling_var = lc * current_var + rc * new_var + (batch / batch_plus_one**2)*(current_mean - new_mean)**2
    return tf.sqrt(rolling_var).numpy()


def log(models, updates, log_config, iteration, e_locs):
    printer = {'iteration/iteration': [iteration, 2]}
    amplitudes, acceptance, samples, e_loc = get_info(models)
    sam_mean = tf.reduce_mean(tf.abs(samples))
    tf.summary.scalar('samples/mean', sam_mean, iteration)
    # energy
    e_loc_mean_shifted = tf.abs(tf.reduce_mean(e_loc) - log_config['e_min'])
    e_loc_mean = tf.reduce_mean(e_loc)
    # e_loc_mean = tf.reduce_mean(e_loc)
    e_loc_std = tf.math.reduce_std(e_loc)
    e_locs.append(e_loc_mean)

    if iteration < 10:
        median = e_locs[iteration]
    else:
        last_iterations = iteration // 10
        median = np.median(e_locs[-last_iterations:])

    printer['energy/median_last10'] = [median, 8]
    printer['energy/e_loc_mean'] = [e_loc_mean_shifted, 7]
    printer['energy/e_loc_mean_negative'] = [e_loc_mean, 7]
    printer['energy/e_loc_std'] = [e_loc_std, 4]

    # samples
    samples_mean = tf.reduce_mean(amplitudes)
    samples_std = tf.math.reduce_std(amplitudes)
    printer['samples/acceptance'] = [acceptance * 100, 3]
    printer['samples/variance_samples'] = [samples_std, 5]
    printer['samples/mean_samples'] = [samples_mean, 4]

    # grads
    mean_grads = sum([tf.reduce_sum(tf.math.abs(grad)) for grad in updates]) / log_config['n_params']
    max_grad = max([tf.reduce_max(tf.math.abs(grad)) for grad in updates])
    printer['grads/mean_grads'] = [mean_grads, 8]
    printer['grads/max_grad'] = [max_grad, 5]

    # params
    params = ray.get(models[0].get_weights.remote())
    mean_params = sum([tf.reduce_sum(tf.abs(weight)) for weight in params]) / log_config['n_params']
    max_params = [tf.reduce_max(tf.abs(weight)) for weight in params]
    max_param = max(max_params)
    printer['params/mean_params'] = [mean_params, 3]
    printer['params/max_param'] = [max_param, 3]

    # printer
    total_time = time()
    printer['time/average_time'] = [(total_time - log_config['t0']) / (iteration+1), 2]
    print_iteration(printer, iteration)

    for layer_id, grad in enumerate(updates):
        mean = tf.reduce_mean(tf.abs(grad))
        maximum = tf.math.reduce_max(tf.abs(grad))
        printer['grads/layer_%i_mean' % layer_id] = [mean, 5]
        printer['grads/layer_%i_max' % layer_id] = [maximum, 5]

    log_iteration(printer, iteration)
    tf.summary.flush()

    # weights = ray.get(models[0].get_weights.remote())
    # for name, w in zip(layers, weights):
    #     ma = tf.reduce_mean(m_aa[name])
    #     ms = tf.reduce_mean(m_ss[name])
    #
    #     tf.summary.scalar('m_xx/m_aa_%s' % name, ma, iteration)
    #     tf.summary.scalar('m_xx/m_ss_%s' % name, ms, iteration)
    #
    #     w = tf.reduce_mean(w)
    #     tf.summary.scalar('weights/%s' % name, w, iteration)


    return

