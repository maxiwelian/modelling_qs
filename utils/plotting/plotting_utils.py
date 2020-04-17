import pickle as pk
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random

def save_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x

def convert_to_label(string):
    new_string = '$\mathrm{'
    for el in string:
        if el == '_':
            new_string += '\_'
        elif el == ' ':
            new_string += '\,\,'
        elif el == '%':
            new_string += '\%'
        elif el == '-':
            new_string += '}-\mathrm{'
        else:
            new_string += el
    new_string += '}$'
    return new_string

def set_mpl_tex():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    return

def normalize_tableau():
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    random.shuffle(tableau20)
    return tableau20

def init_fig(xlabel=None, ylabel=None, title=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.set_xlabel(xlabel, fontsize = 18, labelpad=5)
    ax.set_ylabel(ylabel, fontsize = 18, labelpad=5)
    ax.grid(True, which='major', axis='both')
    if not xlim==None:
        ax.set_xlim(*xlim)
    if not ylim==None:
        ax.set_ylim(*ylim)
    # ax.grid=True
    ax.set_title(title, fontsize=15)
    # self.colour_idx = 0
    ax.axes.tick_params('both', labelsize = 15)
    # self.xline = None
    return fig, ax




def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x