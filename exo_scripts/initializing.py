import os

for ei in [10., 1., 0.1, 0.01]:
    os.system('python main.py -S C -pi 1000 -bi 100 -i 2000 -o adam -exp_dir env_init -exp %.3fei --pretrain' % ei)