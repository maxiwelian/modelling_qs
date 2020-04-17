import os

for ei in [1., 0.1, 0.01, 0.001]:
    os.system('python main.py -o adam -exp_dir env_init -exp %.3fei --pretrain' % ei)