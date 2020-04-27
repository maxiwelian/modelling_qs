
from pyscf import gto
import tensorflow as tf
import pickle as pk
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def reader(path):
    mol = gto.Mole()
    with open(path, 'rb') as f:
        data = pk.load(f)
    mol.atom = data["mol"]
    mol.unit = "Bohr"
    mol.basis = data["basis"]
    mol.verbose = 4
    mol.spin = data["spin"]
    mol.build()
    number_of_electrons = mol.tot_electrons()
    number_of_atoms = mol.natm
    ST = data["super_twist"]
    print('atom: ', mol.atom)
    # mol
    return ST, mol


class Pretrainer():
    def __init__(self,
                 n_pretrain_iterations,
                 n_determinants,
                 n_electrons,
                 n_spin_up,
                 n_spin_down,
                 pretrain_path):


        try:
            # blockPrint()
            self.super_twist, self.mol = reader(pretrain_path)
            self.moT = self.super_twist.T
            # enablePrint()
        except:
            print('pretrain data does not exist...')

        self.n_spin_up = n_spin_up
        self.n_spin_down = n_spin_down

        self.n_electrons = n_electrons
        self.n_determinants = n_determinants
        self.n_iterations = n_pretrain_iterations

    # @tf.function
    def compute_orbital_probability(self, samples):
        up_dets, down_dets = self.wave_function(samples)

        spin_ups = up_dets ** 2
        spin_downs = down_dets ** 2

        p_up = tf.math.reduce_prod(spin_ups, axis=[1, 2])
        p_down = tf.math.reduce_prod(spin_downs, axis=[1, 2])

        probabilities = p_up * p_down

        return probabilities

    def pyscf_call(self, samples):
        samples = samples.numpy()
        ao_values = self.mol.eval_gto("GTOval_cart", samples)
        return ao_values

    def wave_function(self, coord):
        coord = tf.reshape(coord, (-1, 3))

        number_spin_down = self.n_electrons // 2
        number_spin_up = self.n_electrons - number_spin_down

        ao_values = tf.py_function(self.pyscf_call, inp=[coord], Tout=tf.float32)
        ao_values = tf.reshape(ao_values, (int(len(ao_values) / self.n_electrons), self.n_electrons, len(ao_values[0])))

        spin_up = tf.reshape(tf.stack(
            [tf.reduce_sum(self.moT[orb_number, :] * ao_values[:, el_number, :], axis=-1)
             for orb_number in range(number_spin_up) for el_number in
             range(number_spin_up)], axis=1), (-1, number_spin_up, number_spin_up))
        spin_down = tf.reshape(tf.stack(
            [tf.reduce_sum(self.moT[orb_number, :] * ao_values[:, el_number, :], axis=-1) for orb_number in
             range(number_spin_down) for el_number in
             range(number_spin_up, self.n_electrons)], axis=1), (-1, number_spin_down, number_spin_down))

        return spin_up, spin_down

    def compute_grads(self, samples, model):
        up_dets, down_dets = self.wave_function(samples)
        up_dets = tile_labels(up_dets, self.n_determinants)
        down_dets = tile_labels(down_dets, self.n_determinants)
        with tf.GradientTape() as t:
            _, _, _, _, model_up_dets, model_down_dets = model(samples)
            loss = tf.keras.losses.MSE(up_dets, model_up_dets)
            loss += tf.keras.losses.MSE(down_dets, model_down_dets)
        grads = t.gradient(loss, model.trainable_weights[:-1])
        return grads


def tile_labels(label, n_k):
    x = tf.tile(tf.expand_dims(label, axis=1), (1, n_k, 1, 1))
    return x


