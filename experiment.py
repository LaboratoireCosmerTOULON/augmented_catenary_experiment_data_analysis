import os
import warnings

import pandas as pd
from numpy.linalg import norm

warnings.simplefilter(action='ignore', category=Warning)

folder = '/ICRA_EXPORT'
# w_path = '../../Users/FILLIUNG Martin/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
w_path = '../../Users/marti/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
path = os.path.abspath(w_path)

directory = sorted(os.listdir(path))
directory = [file for file in directory if 'dynamique6' in file and '.csv' in file]
files_index = [1, 4, 7, 9, 12, 15]
directory = [directory[i] for i in files_index]

print(f'Found {len(directory)} files:')
for i, f in enumerate(directory):
    print(f'\t{i}\t{f}')

for file in directory:
    print(f'exporting {file} ...')
    data = pd.read_csv(f'{path}/{file}')

    position = data.loc[:, ['robot_cable_attach_point X', 'robot_cable_attach_point Y', 'rod_end X', 'rod_end Y']]
    position[['robot_cable_attach_point_cor X', 'robot_cable_attach_point_cor Y']] = \
        position.apply(lambda row: pd.Series([
            row['robot_cable_attach_point X'] - row['rod_end X'],
            row['robot_cable_attach_point Y'] - row['rod_end Y']
        ]), axis=1)

    with open(f'position_{file[:-4]}.txt', 'w') as write_file:
        write_file.write('%x y\n')
        for x, y in zip(position['robot_cable_attach_point_cor X'], position['robot_cable_attach_point_cor Y']):
            write_file.write(f'{x/1000.} {y/1000.}\n')
