import os

# BEFORE UPLOAD CHANGE PRETRAIN CODE

systems = ['Be', 'C', 'Ne']

for system in systems:
    os.system('python main.py -S %s -o adam -exp_dir adam_sampling -exp pretrain --pretrain' % system)

    for bi in [1, 10, 100]:

        os.system('python main.py -S %s -o adam -i 1000 -bi %i -exp_dir adam_sampling -exp %ibi_' % (system, bi, bi))