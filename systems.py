import tensorflow as tf

systems = {}
# n_atoms, n_electrons, n_spin_up, e_min
systems['Be'] = {'n_atoms': 1,
                 'n_electrons': 4,
                 'n_spin_up': 2,
                 'n_spin_down': 2,
                 'e_min': -14.667333,
                 'ne_atoms': [4],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['B'] = {'n_atoms': 1,
                'n_electrons': 5,
                'n_spin_up': 3,
                'n_spin_down': 2,
                'e_min': -24.653703,
                'ne_atoms': [5],
                'atom_positions': [[0.0, 0.0, 0.0]]}

systems['C'] = {'n_atoms': 1,
                'n_electrons': 6,
                'n_spin_up': 3,
                'n_spin_down': 3,
                'e_min': -37.844715,
                'ne_atoms': [6],
                'atom_positions': [[0.0, 0.0, 0.0]]}

systems['N'] = {'n_atoms': 1,
                'n_electrons': 7,
                'n_spin_up': 4,
                'n_spin_down': 3,
                'e_min': -54.588826,
                'ne_atoms': [7],
                'atom_positions': [[0.0, 0.0, 0.0]]}

systems['O'] = {'n_atoms': 1,
                'n_electrons': 8,
                'n_spin_up': 4,
                'n_spin_down': 4,
                'e_min': -75.066557,
                'ne_atoms': [8],
                'atom_positions': [[0.0, 0.0, 0.0]]}

systems['F'] = {'n_atoms': 1,
                'n_electrons': 9,
                'n_spin_up': 5,
                'n_spin_down': 4,
                'e_min': -99.73291,
                'ne_atoms': [9],
                'atom_positions': [[0.0, 0.0, 0.0]]}

systems['Ne'] = {'n_atoms': 1,
                'n_electrons': 10,
                'n_spin_up': 5,
                'n_spin_down': 5,
                'e_min': -128.93661,
                'ne_atoms': [10],
                'atom_positions': [[0.0, 0.0, 0.0]]}

systems['Na'] = {'n_atoms': 1,
                 'n_electrons': 11,
                 'n_spin_up': 6,
                 'n_spin_down': 5,
                 'e_min': -0.,
                 'ne_atoms': [11],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['Mg'] = {'n_atoms': 1,
                 'n_electrons': 12,
                 'n_spin_up': 6,
                 'n_spin_down': 6,
                 'e_min': -0.,
                 'ne_atoms': [12],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['Al'] = {'n_atoms': 1,
                 'n_electrons': 13,
                 'n_spin_up': 7,
                 'n_spin_down': 6,
                 'e_min': -0.,
                 'ne_atoms': [13],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['Si'] = {'n_atoms': 1,
                 'n_electrons': 14,
                 'n_spin_up': 7,
                 'n_spin_down': 7,
                 'e_min': -0.,
                 'ne_atoms': [14],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['P'] = {'n_atoms': 1,
                 'n_electrons': 15,
                 'n_spin_up': 8,
                 'n_spin_down': 7,
                 'e_min': -0.,
                 'ne_atoms': [15],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['S'] = {'n_atoms': 1,
                 'n_electrons': 16,
                 'n_spin_up': 8,
                 'n_spin_down': 8,
                 'e_min': -0.,
                 'ne_atoms': [16],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['Cl'] = {'n_atoms': 1,
                 'n_electrons': 17,
                 'n_spin_up': 9,
                 'n_spin_down': 8,
                 'e_min': -0.,
                 'ne_atoms': [17],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['Ar'] = {'n_atoms': 1,
                 'n_electrons': 18,
                 'n_spin_up': 9,
                 'n_spin_down': 9,
                 'e_min': -0.,
                 'ne_atoms': [18],
                 'atom_positions': [[0.0, 0.0, 0.0]]}

systems['LiH'] = {'n_atoms': 2,
                  'n_electrons': 4,
                  'n_spin_up': 2,
                  'n_spin_down': 2,
                  'e_min': -8.070548,
                  'ne_atoms': [3, 1],
                  'atom_positions':
                      [[0.0, 0.0, 3.015],  # Li
                       [0, 0.0, 0.0]]}  # H



