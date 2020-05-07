

def save_pretrain_model_and_samples(models, config, iteration):
    save_model(ray.get(models[0].get_weights.remote()), config['pretrained_model'].format(iteration + 1))
    samples = tf.concat(ray.get([model.get_samples.remote() for model in models]), axis=0)
    save_samples(samples, config['pretrained_samples'].format(iteration + 1))


def save_model_and_samples(models, config, iteration):
    save_model(ray.get(models[0].get_weights.remote()), config['model_path'].format(iteration))
    samples = tf.concat(ray.get([model.get_samples.remote() for model in models]), axis=0)
    save_samples(samples, config['samples_path'].format(iteration))
    return


def compute_energy(models, config):
    load_models(models, config['load_model_path'])
    load_samples(models, config['load_samples_path'])
    e_locs, new_mean, new_std = get_energy_of_current_samples(models)
    print('initial mean ', new_mean)

    e_locs = []
    e_overall_mean = 0
    e_overall_std = 0

    burn(models, config['n_burn_in'])

    means = []
    for i in range(config['n_iterations']):
        _, e_mean, e_std = get_energy(models)
        amplitudes, acceptance, samples, e_loc = get_info(models)

        e_locs.append(e_mean)
        rolling_mean = compute_rolling_mean(e_overall_mean, i, e_mean)
        rolling_std = compute_rolling_std(e_overall_std, e_overall_mean, i, e_std, e_mean)

        e_overall_mean = tf.reduce_mean(e_locs)
        total_samples = config['n_samples'] * (i + 1)
        print('overall: ', e_overall_mean.numpy(),
              'rolling_mean: ', rolling_mean,
              'n_samples: ', total_samples,
              'e_std: ', e_std.numpy(),
              'rolling_std: ', rolling_std,
              'sem: ', rolling_std / np.sqrt(total_samples),
              'acceptance: ', acceptance)

        means.append(e_overall_mean)
    with open(os.path.join(config['load_directory'], 'means.pk'), 'wb') as f:
        pk.dump(means, f)


def main(config):

    n_iterations = config['n_iterations']
    n_samples = config['n_samples_total']
    n_gpus = config['n_gpus']

    # create the actors on the available gpus
    models = [Network.remote(config, _) for _ in range(n_gpus)]

    # compute the energy if thats whats up
    if config['compute_energy']:
        compute_energy(models, config)
        exit()

    n_params, n_layers, trainable_shapes, layers = ray.get(models[0].get_model_details.remote())

    # initialize the kfac class
    kfac_args = filter_dict(config, KFAC)
    kfac = KFAC(layers, **kfac_args)

    log_config = {'n_params': n_params,
                  'e_min': config['e_min']}

    writer = tf.summary.create_file_writer(config['experiment'])
    with writer.as_default():

        if config['load_iteration'] > 0:  # load the model
            load_models(models, config['load_model_path'])
            load_samples(models, config['load_samples_path'])

        else:  # pretrain the model
            if os.path.exists(config['pretrained_model']) and os.path.exists(config['pretrained_samples'])\
                    and not config['pretrain']:
                print('Loading pretrained model...')  # load the pretrained model
                load_models(models, config['pretrained_model'])

            else:  # pretrain
                burn_pretrain(models, config['n_burn_in'])
                burn(models, config['n_burn_in'])

                iteration = 0
                for iteration in range(config['n_pretrain_iterations']):
                    grads = get_pretrain_grads(models)
                    update_weights_pretrain(models, grads)
                    [model.sample.remote() for model in models]

                    print('pretrain iteration %i...' % iteration)

                save_pretrain_model_and_samples(models, config, iteration)

            load_samples(models, config['pretrained_samples'])

            print('burning')
            burn(models, config['n_burn_in'])  # burn in

        print('optimizing')
        e_locs = []
        for iteration in range(n_iterations):
            if iteration < 2:
                log_config['t0'] = time()

            if config['opt'] == 'adam':
                updates = get_grads(models)
                update_weights_optimizer(models, updates)

            else:
                grads, m_aa, m_ss, all_a, all_s = get_grads_and_maa_and_mss(models, layers)
                updates = kfac.compute_updates(grads, m_aa, m_ss, all_a, all_s, iteration)

                tf.summary.scalar('kfac/lr', kfac.lr, iteration)
                updates_lr = [-1. * kfac.lr * up for up in updates]

                step_forward(models, updates_lr)

            if iteration % config['log_every'] == 0:
                log(models, updates, log_config, iteration, e_locs)

            if iteration % config['save_every'] == 0:
                save_model_and_samples(models, config, iteration)
                _ = ray.get([model.compute_validation_energy.remote() for model in models])
                tf.summary.scalar('energy/validation_energy', tf.reduce_mean(_), iteration)
            # break

    return


if __name__ == '__main__':
    import logging
    import ray
    ray.init()
    import os
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # tensorflow does not try to fill device
    DIR = os.path.dirname(os.path.realpath(__file__))

    import argparse
    from time import time, sleep

    import tensorflow as tf

    from actor import Network
    from actor_proxy import get_pretrain_grads, update_weights_pretrain  # pretraining
    from actor_proxy import get_grads, update_weights_optimizer  # adam
    from actor_proxy import get_grads_and_maa_and_mss, step_forward  # kfac
    from actor_proxy import burn, burn_pretrain  # sampling
    from actor_proxy import get_energy, get_energy_of_current_samples  # energy

    from model.gradients import KFAC

    from utils.utils import *
    from systems import systems

    parser = argparse.ArgumentParser()

    # hardware
    parser.add_argument('-gpu', '--n_gpus', default=1, type=int)
    parser.add_argument('--seed', help='', action='store_true')
    parser.add_argument('--compute_energy', help='', action='store_true')

    # pretraining
    parser.add_argument('-pi', '--n_pretrain_iterations', default=1000, type=int)
    parser.add_argument('--pretrain', help='force a pretrain', action='store_true')

    # training
    parser.add_argument('-i', '--n_iterations', default=10000, type=int)
    parser.add_argument('-o', '--opt', default='kfac', type=str)
    parser.add_argument('-lr', '--lr0', default=0.0001, type=float)
    parser.add_argument('-d', '--decay', default=0.0001, type=float)

    # kfac
    parser.add_argument('-dm', '--damping_method', help='ft or tikhonov', default='ft', type=str)
    parser.add_argument('-id', '--initial_damping', default=0.001, type=float)
    parser.add_argument('-dd', '--damping_decay', default=0.001, type=float)
    parser.add_argument('-nc', '--norm_constraint', default=0.001, type=float)
    parser.add_argument('-cw', '--cov_weight', default=1., type=float)
    parser.add_argument('-cmw', '--cov_moving_weight', default=0.95, type=float)
    parser.add_argument('-ca', '--conv_approx', default='ba', type=str)
    parser.add_argument('--use_exact_envs', help='use the exact fisher block for envs', action='store_true')
    parser.add_argument('--should_center', help='center activations and sensitivities', action='store_true')
    parser.add_argument('-ftm', '--ft_method', default='original', type=str)

    # sampling
    parser.add_argument('-si', '--sampling_init', default=1., type=float)
    parser.add_argument('-ss', '--sampling_steps', default=0.02**0.5, type=float)
    parser.add_argument('-bi', '--n_burn_in', default=10, type=int)
    parser.add_argument('-cl', '--correlation_length', default=10, type=int)
    parser.add_argument('-bb', '--burn_batches', default=10, type=int)  # number of batches in a burn

    # model
    parser.add_argument('-S', '--system', default='Be', type=str)
    parser.add_argument('--half_model', help='use only half a model', action='store_true')
    parser.add_argument('-n', '--n_samples', default=4096, type=int)
    parser.add_argument('-s', '--nf_hidden_single', default=256, type=int)
    parser.add_argument('-p', '--nf_hidden_pairwise', default=32, type=int)
    parser.add_argument('-k', '--n_determinants', default=16, type=int)
    parser.add_argument('-ei', '--env_init', default=1., type=float)
    parser.add_argument('--mix_final', help='', action='store_true')
    parser.add_argument('--full_pairwise', help='', action='store_true')
    parser.add_argument('--mix_input', help='', action='store_true')
    parser.add_argument('--test', help='', action='store_true')

    # paths
    parser.add_argument('-l', '--load_iteration', default=0, type=int)
    parser.add_argument('-exp_dir', '--exp_dir', default='', type=str)
    parser.add_argument('-exp', '--exp', default='', type=str)
    parser.add_argument('--local', help='is local', action='store_true')
    parser.add_argument('-ld', '--load_directory', default='')

    # logging
    parser.add_argument('-le', '--log_every', default=1, type=int)
    parser.add_argument('-se', '--save_every', default=5000, type=int)

    args = parser.parse_args()
    args.seed = True
    args.n_iterations += 1
    args.should_center = True
    args.mix_input = True
    args.full_pairwise = True

    if args.test:
        args.local = True
        args.half_model = True
        args.full_pairwise = True
        args.mix_input = True

    # python main.py -gpu 4 -o kfac -exp the1 -pi 2000 -bi 100 -i 100000 -dm ft -exp_dir the1 -ca ba
    # python main.py -gpu 1 -o kfac -exp first_run -pi 400 -bi 10 -i 1000 -dm ft -exp_dir ft_new_inv --local --half_model

    model = ''
    if args.half_model:
        args.n_samples = 1024
        args.nf_hidden_single = 128
        args.nf_hidden_pairwise = 32
        args.n_determinants = 16
        model = '_hm'

    fp = ''
    if args.full_pairwise:
        fp = '_fp'

    mf = ''
    if args.mix_final:
        mf = '_mf'

    mi = ''
    if args.mix_input:
        mi = '_mi'

    # generate the paths
    if args.local:
        save_directory = DIR
    else:
        save_directory = '/nobackup/amwilso4/modelling_qs/'

    # name the experiment
    system_directory = os.path.join(save_directory, 'experiments', args.system)
    exp_dir = os.path.join(system_directory, args.exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    n_exps = len(os.listdir(exp_dir))
    experiment = '%s_%s_%s_%.4f_%i' % (args.exp, args.system, args.opt, args.lr0, n_exps)
    if args.opt == 'kfac':
        experiment += '_%s_%s%s%s%s%s' % (args.damping_method, args.conv_approx, model, fp, mf, mi)  # dmeth, conv_apprx etc

    path_experiment = os.path.join(exp_dir, experiment)

    load_model_path = os.path.join(args.load_directory, 'i{}.ckpt').format(args.load_iteration)
    load_samples_path = os.path.join(args.load_directory, 'i{}.pk').format(args.load_iteration)
    model_path = os.path.join(path_experiment, 'i{}.ckpt')
    samples_path = os.path.join(path_experiment, 'i{}.pk')

    pretrain_path = os.path.join(DIR, 'pretraining/data/%s/%s_data.p' % (args.system, args.system))

    # pretrain paths
    pretrain_dir = os.path.join(system_directory, 'pretrained')
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)

    pretrain_tag = (args.n_pretrain_iterations, args.nf_hidden_single, args.nf_hidden_pairwise, args.n_determinants,
                    fp, mf, mi)
    name = 'pm_%i_%i_%i_%i%s%s%s.pk' % pretrain_tag
    path_pretrained_model = os.path.join(pretrain_dir, name)
    name = 'ps_%i_%i_%i_%i%s%s%s.pk' % pretrain_tag
    path_pretrained_samples = os.path.join(pretrain_dir, name)

    # merge the dictionaries
    args_config = vars(args)
    system_config = systems[args.system]

    r_atoms = []
    for position in systems[args.system]['atom_positions']:
        position = tf.constant(position)
        r_atoms.append(tf.ones((1, 1, 3), dtype=tf.float32) * position)
    r_atoms = tf.concat(r_atoms, axis=1)

    z_atoms = []
    for ne_atom in systems[args.system]['ne_atoms']:
        z_atom = tf.ones((1, 1), dtype=tf.float32) * ne_atom
        z_atoms.append(z_atom)
    z_atoms = tf.concat(z_atoms, axis=1)

    config = {'n_samples_total': args.n_samples,
              'n_samples_actor': args.n_samples // args.n_gpus,

              # paths
              'pretrained_model': path_pretrained_model,
              'pretrained_samples': path_pretrained_samples,
              'pretrain_path': pretrain_path,

              'experiment': path_experiment,
              'model_path': model_path,
              'samples_path': samples_path,
              'load_model_path': load_model_path,
              'load_samples_path': load_samples_path,

              # values
              'r_atoms': r_atoms,
              'z_atoms': z_atoms,
              'n_spins': systems[args.system]['n_electrons']
    }

    config = dict(args_config, **config)
    config = dict(config, **system_config)

    if not config['compute_energy']:
        if not os.path.exists(path_experiment):
            os.makedirs(path_experiment)

        with open(os.path.join(path_experiment, 'config.pk'), 'wb') as f:
            pk.dump(config, f)

    main(config)



