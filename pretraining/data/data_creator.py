from pyscf import gto
from pyscf import scf
import numpy as np
import pickle

#
# def reader(path):
#     mol = gto.Mole()
#     with open(path, 'rb') as f:
#         data = pk.load(f)
#     mol.atom = data["mol"]
#     mol.unit = "Bohr"
#     mol.basis = data["basis"]
#     mol.verbose = 4
#     mol.spin = data["spin"]
#     mol.build()
#     number_of_electrons = mol.tot_electrons()
#     number_of_atoms = mol.natm
#     ST = data["super_twist"]
#     print('atom: ', mol.atom)
#     # mol
#     return ST, mol


mol = gto.Mole()
mol.atom = """
Ne 0.0 0.0 0.0
"""

mol.unit = "Bohr"
mol.basis = "sto3g"
mol.verbose = 4
mol.spin = 0
mol.build()

number_of_atoms = mol.natm
conv, e, mo_e, mo, mo_occ = scf.rhf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))

# scf.RHF(mol).run().analyze()
# basis = gto.mole.uncontract(gto.load("cc-pvDZ", "Be"))
# e_scf=mf.kernel()

data = {"super_twist": mo, "mol": mol.atom, "basis": mol.basis, "spin": mol.spin}
pickle.dump(data, open("Ne/Ne_data.p", "wb"))