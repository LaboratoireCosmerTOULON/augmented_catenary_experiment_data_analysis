import os
import warnings

import numpy as np
import pandas as pd
from alive_progress import alive_bar

warnings.simplefilter(action='ignore', category=Warning)

folder = '/ICRA_EXPORT'
w_path = '../../Users/FILLIUNG Martin/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
# w_path = '../../Users/marti/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
path = os.path.abspath(w_path)

directory = sorted(os.listdir(path))
directory = [file for file in directory if '.csv' in file and not 'side_only' in file]
print(f'Found {len(directory)} files:')
for i, f in enumerate(directory):
    print(f'\t{i}\t{f}')

stats = {}
datas = {}

for cable in range(1, 9):
    datas[f'{cable}'] = {'vertical': np.nan, 'inclined': np.nan, 'x': np.nan, 'y': np.nan}
    stats[f'{cable}'] = {'x': np.nan, 'y': np.nan}
    for model in ['vertical', 'inclined']:
        stats[f'{cable}'][model] = {
            'Q1': np.nan,
            'Q25': np.nan,
            'Q50': np.nan,
            'Q75': np.nan,
            'Q99': np.nan,
            'IQR': np.nan,
            'fill': 0,
            'nframes': 0
        }
    for axis in ['x', 'y']:
        datas[f'{cable}'][axis] = {'100': np.nan, '200': np.nan}
        stats[f'{cable}'][axis] = {'100': np.nan, '200': np.nan}
        for speed in ['100', '200']:
            datas[f'{cable}'][axis][speed] = {'dis1': np.nan, 'dis2': np.nan}
            stats[f'{cable}'][axis][speed] = {'dis1': np.nan, 'dis2': np.nan}
            for dis in ['dis1', 'dis2']:
                datas[f'{cable}'][axis][speed][dis] = {'vertical': np.nan, 'inclined': np.nan}
                stats[f'{cable}'][axis][speed][dis] = {'vertical': np.nan, 'inclined': np.nan}
                for model in ['vertical', 'inclined']:
                    stats[f'{cable}'][axis][speed][dis][model] = {
                        'Q1': np.nan,
                        'Q25': np.nan,
                        'Q50': np.nan,
                        'Q75': np.nan,
                        'Q99': np.nan,
                        'IQR': np.nan,
                        'fill': 0,
                        'nframes': 0,
                    }

with alive_bar(len(directory), title='Getting data', theme='musical') as bar:
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

        datas[cable]['vertical'] = vertical['mean'] if type(datas[cable]['vertical']) == float else pd.concat((
            datas[cable]['vertical'], vertical['mean']
        ))
        datas[cable]['inclined'] = inclined['mean'] if type(datas[cable]['inclined']) == float else pd.concat((
            datas[cable]['inclined'], inclined['mean']
        ))

        datas[cable][axis][speed][distance]['vertical'] = vertical['mean'] \
            if type(datas[cable][axis][speed][distance]['vertical']) == float \
            else pd.concat((datas[cable][axis][speed][distance]['vertical'], vertical['mean']))
        datas[cable][axis][speed][distance]['inclined'] = inclined['mean'] \
            if type(datas[cable][axis][speed][distance]['inclined']) == float \
            else pd.concat((datas[cable][axis][speed][distance]['inclined'], inclined['mean']))

        stats[cable]['vertical']['fill'] += data["vcat_0 D"].count()
        stats[cable]['inclined']['fill'] += data["tcat_0 D"].count()
        stats[cable]['vertical']['nframes'] += data.shape[0]
        stats[cable]['inclined']['nframes'] += data.shape[0]

        stats[cable][axis][speed][distance]['vertical']['fill'] += data["vcat_0 D"].count()
        stats[cable][axis][speed][distance]['inclined']['fill'] += data["tcat_0 D"].count()
        stats[cable][axis][speed][distance]['vertical']['nframes'] += data.shape[0]
        stats[cable][axis][speed][distance]['inclined']['nframes'] += data.shape[0]

        bar()

with alive_bar(8, title='Computing statistics', theme='musical') as bar:
    for cable, cable_data in datas.items():
        stats[cable]['vertical']['Q1'] = cable_data['vertical'].quantile(1 / 100)
        stats[cable]['vertical']['Q25'] = cable_data['vertical'].quantile(1 / 4)
        stats[cable]['vertical']['Q50'] = cable_data['vertical'].quantile(2 / 4)
        stats[cable]['vertical']['Q75'] = cable_data['vertical'].quantile(3 / 4)
        stats[cable]['vertical']['Q99'] = cable_data['vertical'].quantile(99 / 100)
        stats[cable]['vertical']['IQR'] = stats[cable]['vertical']['Q75'] - stats[cable]['vertical'][
            'Q25']

        stats[cable]['inclined']['Q1'] = cable_data['inclined'].quantile(1 / 100)
        stats[cable]['inclined']['Q25'] = cable_data['inclined'].quantile(1 / 4)
        stats[cable]['inclined']['Q50'] = cable_data['inclined'].quantile(2 / 4)
        stats[cable]['inclined']['Q75'] = cable_data['inclined'].quantile(3 / 4)
        stats[cable]['inclined']['Q99'] = cable_data['inclined'].quantile(99 / 100)
        stats[cable]['inclined']['IQR'] = stats[cable]['inclined']['Q75'] - stats[cable]['inclined'][
            'Q25']

        files_data = {'x': cable_data['x'], 'y': cable_data['y']}
        for axis, axis_data in files_data.items():
            for speed, speed_data, in axis_data.items():
                for distance, distance_data in speed_data.items():
                    if type(distance_data['vertical']) != float:
                        stats[cable][axis][speed][distance]['vertical']['Q1'] = \
                            distance_data['vertical'].quantile(1 / 100)
                        stats[cable][axis][speed][distance]['vertical']['Q25'] = \
                            distance_data['vertical'].quantile(1 / 4)
                        stats[cable][axis][speed][distance]['vertical']['Q50'] = \
                            distance_data['vertical'].quantile(2 / 4)
                        stats[cable][axis][speed][distance]['vertical']['Q75'] = \
                            distance_data['vertical'].quantile(3 / 4)
                        stats[cable][axis][speed][distance]['vertical']['Q99'] = \
                            distance_data['vertical'].quantile(99 / 100)
                        stats[cable][axis][speed][distance]['vertical']['IQR'] = \
                            stats[cable][axis][speed][distance]['vertical']['Q75'] - \
                            stats[cable][axis][speed][distance]['vertical']['Q25']

                    if type(distance_data['inclined']) != float:
                        stats[cable][axis][speed][distance]['inclined']['Q1'] = \
                            distance_data['inclined'].quantile(1 / 100)
                        stats[cable][axis][speed][distance]['inclined']['Q25'] = \
                            distance_data['inclined'].quantile(1 / 4)
                        stats[cable][axis][speed][distance]['inclined']['Q50'] = \
                            distance_data['inclined'].quantile(2 / 4)
                        stats[cable][axis][speed][distance]['inclined']['Q75'] = \
                            distance_data['inclined'].quantile(3 / 4)
                        stats[cable][axis][speed][distance]['inclined']['Q99'] = \
                            distance_data['inclined'].quantile(99 / 100)
                        stats[cable][axis][speed][distance]['inclined']['IQR'] = \
                            stats[cable][axis][speed][distance]['inclined']['Q75'] - \
                            stats[cable][axis][speed][distance]['inclined']['Q25']

        bar()

for cable, cable_data in stats.items():
    print(f'cable {cable}')
    print(
        f'vertical: '
        f'Q1={cable_data["vertical"]["Q1"]}; '
        f'Q25={cable_data["vertical"]["Q25"]}; '
        f'Q50={cable_data["vertical"]["Q50"]}; '
        f'Q75={cable_data["vertical"]["Q75"]} '
        f'Q99={cable_data["vertical"]["Q99"]}'
    )
    print(
        f'inclined: '
        f'Q1={cable_data["inclined"]["Q1"]}; '
        f'Q25={cable_data["inclined"]["Q25"]}; '
        f'Q50={cable_data["inclined"]["Q50"]}; '
        f'Q75={cable_data["inclined"]["Q75"]} '
        f'Q99={cable_data["inclined"]["Q99"]}'
    )
    files_data = {'x': cable_data['x'], 'y': cable_data['y']}
    for axis, axis_data in files_data.items():
        print(f'\tdirection {axis}')
        for speed, speed_data, in axis_data.items():
            print(f'\t\tspeed {speed}')
            for distance, distance_data in speed_data.items():
                print(f'\t\t\tdistance {distance}')
                print(
                    f'\t\t\t\tvertical: '
                    f'Q1={distance_data["vertical"]["Q1"]}; '
                    f'Q25={distance_data["vertical"]["Q25"]}; '
                    f'Q50={distance_data["vertical"]["Q50"]}; '
                    f'Q75={distance_data["vertical"]["Q75"]} '
                    f'Q99={distance_data["vertical"]["Q99"]}'
                )
                print(
                    f'\t\t\t\tinclined: '
                    f'Q1={distance_data["inclined"]["Q1"]}; '
                    f'Q25={distance_data["inclined"]["Q25"]}; '
                    f'Q50={distance_data["inclined"]["Q50"]}; '
                    f'Q75={distance_data["inclined"]["Q75"]} '
                    f'Q99={distance_data["inclined"]["Q99"]}'
                )
