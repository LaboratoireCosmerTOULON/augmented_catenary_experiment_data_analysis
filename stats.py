import os
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from alive_progress import alive_bar

warnings.simplefilter(action='ignore', category=Warning)

folder = '/ICRA_EXPORT'
# w_path = '../../Users/FILLIUNG Martin/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
w_path = '../../Users/marti/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
path = os.path.abspath(w_path)

directory = sorted(os.listdir(path))
directory = [file for file in directory if '.csv' in file and 'side_only' in file]
print(f'Found {len(directory)} files:')
for i, f in enumerate(directory):
    print(f'\t{i}\t{f}')

stats = {}

for cable in range(1, 9):
    stats[f'{cable}'] = {'x': 0, 'y': 0}
    for axis in ['x', 'y']:
        stats[f'{cable}'][axis] = {'100': 0, '200': 0}
        for speed in ['100', '200']:
            stats[f'{cable}'][axis][speed] = {'dis1': 0, 'dis2': 0}
            for dis in ['dis1', 'dis2']:
                stats[f'{cable}'][axis][speed][dis] = {'vertical': 0, 'inclined': 0}
                for model in ['vertical', 'inclined']:
                    stats[f'{cable}'][axis][speed][dis][model] = {
                        'mean': np.nan,
                        'stddev': np.nan,
                        'fill': np.nan,
                        'nframes': np.nan,
                    }

with alive_bar(len(directory), theme='musical') as bar:
    for file in directory:
        cable = file[11]
        axis = file[12]
        speed = file[13:16]
        distance = file[16:20]
        n_points = 16 if not cable == '7' else 13
        data = pd.read_csv(f'{path}/{file}')
        vertical = data.loc[:, [f'vcat_{i} D' for i in range(n_points)]]
        inclined = data.loc[:, [f'tcat_{i} D' for i in range(n_points)]]

        vertical['mean'] = vertical.apply(lambda row: pd.Series((row.mean())), axis=1)
        inclined['mean'] = inclined.apply(lambda row: pd.Series((row.mean())), axis=1)

        file_v_mean = vertical['mean'].mean()
        file_v_stddev = vertical['mean'].std()
        file_v_fill = data["vcat_0 D"].count()
        file_v_nframes = data.shape[0]

        file_t_mean = inclined['mean'].mean()
        file_t_stddev = inclined['mean'].std()
        file_t_fill = data["tcat_0 D"].count()
        file_t_nframes = data.shape[0]

        if stats[cable][axis][speed][distance]['vertical']['mean'] is np.nan:
            stats[cable][axis][speed][distance]['vertical']['mean'] = file_v_mean
            stats[cable][axis][speed][distance]['vertical']['stddev'] = file_v_stddev
            stats[cable][axis][speed][distance]['vertical']['fill'] = file_v_fill
            stats[cable][axis][speed][distance]['vertical']['nframes'] = file_v_nframes

            stats[cable][axis][speed][distance]['inclined']['mean'] = file_t_mean
            stats[cable][axis][speed][distance]['inclined']['stddev'] = file_t_stddev
            stats[cable][axis][speed][distance]['inclined']['fill'] = file_t_fill
            stats[cable][axis][speed][distance]['inclined']['nframes'] = file_t_nframes

        else:
            if not np.isnan(file_v_mean):
                stats[cable][axis][speed][distance]['vertical']['mean'] += file_v_mean
                stats[cable][axis][speed][distance]['vertical']['mean'] /= 2
                temp = stats[cable][axis][speed][distance]['vertical']['stddev']
                stats[cable][axis][speed][distance]['vertical']['stddev'] = sqrt(
                    pow(temp, 2) +
                    pow(file_v_stddev, 2)
                )
            stats[cable][axis][speed][distance]['vertical']['fill'] += file_v_fill
            stats[cable][axis][speed][distance]['vertical']['nframes'] += file_v_nframes

            if not np.isnan(file_t_mean):
                stats[cable][axis][speed][distance]['inclined']['mean'] += file_t_mean
                stats[cable][axis][speed][distance]['inclined']['mean'] /= 2
                temp = stats[cable][axis][speed][distance]['inclined']['stddev']
                stats[cable][axis][speed][distance]['inclined']['stddev'] = sqrt(
                    pow(temp, 2) +
                    pow(file_t_stddev, 2)
                )
            stats[cable][axis][speed][distance]['inclined']['fill'] += data["tcat_0 D"].count()
            stats[cable][axis][speed][distance]['inclined']['nframes'] += data.shape[0]

        bar()

for cable, cable_data in stats.items():
    v_mean = np.nan
    t_mean = np.nan
    v_stddev = np.nan
    t_stddev = np.nan
    for axis, axis_data in cable_data.items():
        for speed, speed_data, in axis_data.items():
            for distance, distance_data in speed_data.items():
                if not np.isnan(distance_data['vertical']['mean']):
                    if np.isnan(v_mean):
                        v_mean = distance_data['vertical']['mean']
                        v_stddev = distance_data['vertical']['stddev']
                    else:
                        v_mean = (v_mean + distance_data['vertical']['mean']) / 2
                        v_stddev = sqrt(pow(v_mean, 2) + pow(distance_data['vertical']['mean'], 2))
                if not np.isnan(distance_data['inclined']['mean']):
                    if np.isnan(t_mean):
                        t_mean = distance_data['inclined']['mean']
                        t_stddev = distance_data['inclined']['stddev']
                    else:
                        t_mean = (t_mean + distance_data['inclined']['mean']) / 2
                        t_stddev = sqrt(pow(t_mean, 2) + pow(distance_data['inclined']['mean'], 2))

    print(f'{cable}')
    print(f'\tvertical: {v_mean}; {v_stddev}')
    print(f'\tinclined: {t_mean}; {t_stddev}')
    for axis, axis_data in cable_data.items():
        print(f'\t{axis}')
        for speed, speed_data, in axis_data.items():
            print(f'\t\t{speed}')
            for distance, distance_data in speed_data.items():
                print(f'\t\t\t{distance}')
                print('\t\t\t\tvertical', distance_data['vertical'])
                print('\t\t\t\tinclined', distance_data['inclined'])
