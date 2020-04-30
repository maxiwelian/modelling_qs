import os

iterations = 1000
system = 'Be'

base_script = 'python main.py -gpu 8 --mix_input --full_pairwise -i 2500 -S %s -exp_dir kfac3004_baselines ' % system
# should center
cov_weights = (0.05, 1.)
ftms = ('original', 'alternate', 'ones_pi')
dampings = (0.1, 0.01, 0.001)
dms = ('ft', 'tikhonov')

for id in dampings:
    for cw in cov_weights:
        for dm in dms:
            if dm == 'tikhonov':
                os.system(base_script + '-cw %f -id %f -dm %s' % (cw, id, dm))
            else:
                for ftm in ftms:
                    os.system(base_script + '-cw %f -id %f -dm %s -ftm %s' % (cw, id, dm, ftm))


