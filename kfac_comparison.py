import os

iterations = 1000
system = 'Be'

base_script = 'python main.py -gpu 8 --mix_input --full_pairwise -i 2500 -S %s -exp_dir kfac3004_baselines ' % system
# should center
# should_centers = (True, False)
cas = ('mg', 'ba')
cov_weights = (0.05, 1.)
# pi_ones = (True, False)
dms = ('ft', 'tikhonov')

for dm in dms:
    for ca in cas:
        for cw in cov_weights:
            os.system(base_script + '-dm %s -ca %s -cs %f' % (dm, ca, cw))
