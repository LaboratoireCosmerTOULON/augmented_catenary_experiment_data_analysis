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
directory = [
    file for file in directory
    if ('dynamique3' in file or 'dynamique6' in file)
    and '200' in file
    and 'dis2' in file
    and '.csv' in file
]

print(f'Found {len(directory)} files:')
for i, f in enumerate(directory):
    print(f'\t{i}\t{f}')

for file in directory:
    print(f'exporting {file} ...')
    data = pd.read_csv(f'{path}/{file}')

    vertical = data.loc[:, [f'vcat_{i} D' for i in range(16)]]
    inclined = data.loc[:, [f'tcat_{i} D' for i in range(16)]]
    speed = data.loc[:, ['rob_speed X', 'rob_speed Y', 'rob_speed Z']]

    vertical['mean'] = vertical.apply(lambda row: pd.Series((row.mean())), axis=1)
    inclined['mean'] = inclined.apply(lambda row: pd.Series((row.mean())), axis=1)
    speed['norm'] = speed.apply(lambda row: pd.Series((norm(row) / 1000.)), axis=1)

    with open(f'vertical_{file[:-4]}.txt', 'w') as write_file:
        write_file.write('%x y\n')
        for t, v in zip(data['Time'], vertical['mean']):
            write_file.write(f'{t} {v}\n')

    with open(f'inclined_{file[:-4]}.txt', 'w') as write_file:
        write_file.write('%x y\n')
        for t, v in zip(data['Time'], inclined['mean']):
            write_file.write(f'{t} {v}\n')

    with open(f'speed_{file[:-4]}.txt', 'w') as write_file:
        write_file.write('%x y\n')
        for t, v in zip(data['Time'], speed['norm']):
            write_file.write(f'{t} {v}\n')

    with open(f'theta_{file[:-4]}.txt', 'w') as write_file:
        write_file.write('%x y\n')
        for t, v in zip(data['Time'], data['Theta']):
            write_file.write(f'{t} {v}\n')

    with open(f'gamma_{file[:-4]}.txt', 'w') as write_file:
        write_file.write('%x y\n')
        for t, v in zip(data['Time'], data['Gamma']):
            write_file.write(f'{t} {v}\n')