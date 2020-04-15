
def main(config):
    if config['seed']:
        tf.random.set_seed(127)

    n_iterations = config['n_iterations']
    n_samples = config['n_samples_total']
    n_gpus = config['n_gpus']

    # create the actors on the available gpus
    models = [Network.remote(config) for _ in range(n_gpus)]
    n_params, n_layers, trainable_shapes, layers = ray.get(models[0].get_model_details.remote())

    # initialize the kfac class
    kfac_args = filter_dict(config, KFAC)
    kfac = KFAC(layers, **kfac_args)

    log_config = {'n_params': n_params,
                  'e_min': config['e_min']}

    writer = tf.summary.create_file_writer(config['experiment'])
    with writer.as_default():
        if config['load_iteration'] > 0: # load the model
            load_models(models, config['saved_model'])

        else:  # pretrain the model
            if os.path.exists(config['pretrained_model']) and os.path.exists(config['pretrained_samples']):
                print('Loading pretrained model...')  # load the pretrained model
                load_models(models, config['pretrained_model'])
                load_samples(models, config['pretrained_samples'])

            else:  # pretrain
                for iteration in range(config['n_pretrain_iterations']):
                    new_grads = get_pretrain_grads(models)
                    update_weights_pretrain(models, new_grads)

        # training
        burn(models, config['n_burn_in'])  # burn in

        for iteration in range(n_iterations):
            printer = {}

            if config['opt'] == 'adam':
                updates = get_grads(models)
                update_weights_optimizer(models, updates)

            else:
                grads, m_aa, m_ss = get_grads_and_maa_and_mss(models)

                updates = kfac.compute_updates(grads, m_aa, m_ss, iteration)

            if iteration % config['log_every']:
                log(printer, updates, log_config, iteration)
    return

if __name__ == '__main__':
    import ray
    ray.init()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # tensorflow does not try to fill device
    DIR = os.path.dirname(os.path.realpath(__file__))

    import argparse
    import tensorflow as tf

    from actor import Network
    from actor import get_pretrain_grads, update_weights_pretrain  # pretraining
    from actor import get_grads, update_weights_optimizer  # adam
    from actor import get_grads_and_maa_and_mss  # kfac
    from actor import burn  # sampling
    from model.gradients import KFAC_Actor, KFAC

    from utils import *
    from systems import systems

    parser = argparse.ArgumentParser()

    # hardware
    parser.add_argument('-gpu', '--n_gpus', default=1, type=int)
    parser.add_argument('--seed', help='', action='store_true')

    # pretraining
    parser.add_argument('-pi', '--n_pretrain_iterations', default=1000, type=int)

    # training
    parser.add_argument('-i', '--n_iterations', default=10000, type=int)
    parser.add_argument('-o', '--opt', default='adam', type=str)
    parser.add_argument('-lr', '--lr0', default=0.0001, type=float)
    parser.add_argument('-d', '--decay', default=0.0001, type=float)

    # kfac
    parser.add_argument('-dm', '--damping_method', default='ft', type=str)
    parser.add_argument('-id', '--initial_damping', default=0.001, type=float)
    parser.add_argument('-nc', '--norm_constraint', default=0.001, type=float)
    parser.add_argument('-ca', '--conv_approx', default='mg', type=str)

    # sampling
    parser.add_argument('-si', '--sampling_init', default=1., type=float)
    parser.add_argument('-ss', '--sampling_step', default=0.0004, type=float)
    parser.add_argument('-bi', '--burn_in', default=10, type=int)

    # model
    parser.add_argument('-S', '--system', default='Be', type=str)
    parser.add_argument('--half_model', help='use only half a model', action='store_true')
    parser.add_argument('-n', '--n_samples', default=4096, type=int)
    parser.add_argument('-s', '--nf_hidden_single', default=256, type=int)
    parser.add_argument('-p', '--nf_hidden_pairwise', default=32, type=int)
    parser.add_argument('-k', '--n_determinants', default=16, type=int)

    # paths
    parser.add_argument('-l', '--load_iteration', default=0, type=int)
    parser.add_argument('-exp_dir', '--exp_dir', default='', type=str)
    parser.add_argument('-exp', '--exp', default='', type=str)

    args = parser.parse_args()

    if args.half_model:
        args.n_samples = 1024
        args.nf_hidden_single = 128
        args.nf_hidden_pairwise = 32
        args.n_determinants = 16

    # generate the paths
    system_dir = os.path.join(DIR, 'experiments', args.system)

    exp_dir = os.path.join(system_dir, args.exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    n_exps = len(os.listdir(exp_dir))

    name = '%s_%s_%.4f_%i' % (args.system, args.opt, args.lr0, n_exps)
    if args.opt == 'kfac':
        name += '_%s_%s' % (args.dmeth, args.conv_approx)  # dmeth, conv_apprx etc

    path_experiment = os.path.join(exp_dir, name)

    pretrain_tag = (args.nf_hidden_single, args.nf_hidden_pairwise, args.n_determinants)
    name = 'pm_%i_%i_%i.pk' % pretrain_tag
    path_pretrained_model = os.path.join(system_dir, 'pretrained', name)
    name = 'ps_%i_%i_%i.pk' % pretrain_tag
    path_pretrained_samples = os.path.join(system_dir, 'pretrained', name)

    model_path = os.path.join(path_experiment, 'i%i.ckpt' % args.load_iteration)

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

    config = {'n_samples_total':args.n_samples,
              'n_samples_actor':args.n_samples // args.n_gpus,

              # paths
              'pretrained_model': path_pretrained_model,
              'pretrained_samples':path_pretrained_samples,
              'experiment': path_experiment,
              'model_path': model_path,

              # values
              'r_atoms': r_atoms,
              'z_atoms': z_atoms
    }

    config = dict(args_config, **config)
    config = dict(config, **system_config)

    main(config)



