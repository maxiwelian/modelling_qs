import os

systems = ['Be', 'C', 'Ne']

for system in systems:
    args = (system, )
    os.system('python main.py'
              '-o adam -S %s'
              % args)
