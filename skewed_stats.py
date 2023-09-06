import os
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=Warning)

folder = '/ICRA_EXPORT'
# w_path = '../../Users/FILLIUNG Martin/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
w_path = '../../Users/marti/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
path = os.path.abspath(w_path)

directory = sorted(os.listdir(path))
directory = [file for file in directory if '.csv' in file]
print(f'Found {len(directory)} files:')
for i, f in enumerate(directory):
    print(f'\t{i}\t{f}')

stats = {}
datas = {}

for cable in range(1, 9):
    datas[f'{cable}'] = {
        'vertical': np.nan, 'theta': np.nan, 'gamma': np.nan, 'theta_gamma': np.nan, 'x': np.nan, 'y': np.nan
    }
    stats[f'{cable}'] = {'x': np.nan, 'y': np.nan}
    for model in ['vertical', 'theta', 'gamma', 'theta_gamma']:
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
                datas[f'{cable}'][axis][speed][dis] = {
                    'vertical': np.nan, 'theta': np.nan, 'gamma': np.nan, 'theta_gamma': np.nan
                }
                stats[f'{cable}'][axis][speed][dis] = {
                    'vertical': np.nan, 'theta': np.nan, 'gamma': np.nan, 'theta_gamma': np.nan
                }
                for model in ['vertical', 'theta', 'gamma', 'theta_gamma']:
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

for file in directory:
    print(f'loading {file}; ', end='')
    cable = file[11]
    axis = file[12]
    speed = file[13:16]
    distance = file[16:20]
    n_points = 16 if not cable == '7' else 13
    data = pd.read_csv(f'{path}/{file}')

    print(
        f'fill rates: vertical={100 * data["vcat_0 D"].count() / data.shape[0]:.2f}%; '
        f'theta={100 * data["t_acat_0 D"].count() / data.shape[0]:.2f}%; '
        f'gamma={100 * data["g_acat_0 D"].count() / data.shape[0]:.2f}%; '
        f'theta_gamma={100 * data["tg_acat_0 D"].count() / data.shape[0]:.2f}%;'
        )

    vertical = data.loc[:, [f'vcat_{i} D' for i in range(n_points)]]
    theta = data.loc[:, [f't_acat_{i} D' for i in range(n_points)]]
    gamma = data.loc[:, [f'g_acat_{i} D' for i in range(n_points)]]
    theta_gamma = data.loc[:, [f'tg_acat_{i} D' for i in range(n_points)]]

    vertical['mean'] = vertical.apply(lambda row: pd.Series((row.mean())), axis=1)
    theta['mean'] = theta.apply(lambda row: pd.Series((row.mean())), axis=1)
    gamma['mean'] = gamma.apply(lambda row: pd.Series((row.mean())), axis=1)
    theta_gamma['mean'] = theta_gamma.apply(lambda row: pd.Series((row.mean())), axis=1)

    datas[cable]['vertical'] = vertical['mean'] if type(datas[cable]['vertical']) == float else pd.concat(
        (datas[cable]['vertical'], vertical['mean'])
    )
    datas[cable]['theta'] = theta['mean'] if type(datas[cable]['theta']) == float else pd.concat(
        (datas[cable]['theta'], theta['mean'])
    )
    datas[cable]['gamma'] = gamma['mean'] if type(datas[cable]['gamma']) == float else pd.concat(
        (datas[cable]['gamma'], gamma['mean'])
    )
    datas[cable]['theta_gamma'] = theta_gamma['mean'] if type(datas[cable]['theta_gamma']) == float else pd.concat(
        (datas[cable]['theta_gamma'], theta_gamma['mean'])
    )

    datas[cable][axis][speed][distance]['vertical'] = vertical['mean'] if type(
        datas[cable][axis][speed][distance]['vertical']
    ) == float else pd.concat((datas[cable][axis][speed][distance]['vertical'], vertical['mean']))

    datas[cable][axis][speed][distance]['theta'] = theta['mean'] if type(
        datas[cable][axis][speed][distance]['theta']
    ) == float else pd.concat((datas[cable][axis][speed][distance]['theta'], theta['mean']))

    datas[cable][axis][speed][distance]['gamma'] = gamma['mean'] if type(
        datas[cable][axis][speed][distance]['gamma']
    ) == float else pd.concat((datas[cable][axis][speed][distance]['gamma'], gamma['mean']))

    datas[cable][axis][speed][distance]['theta_gamma'] = theta_gamma['mean'] if type(
        datas[cable][axis][speed][distance]['theta_gamma']
    ) == float else pd.concat((datas[cable][axis][speed][distance]['theta_gamma'], theta_gamma['mean']))

    stats[cable]['vertical']['fill'] += data["vcat_0 D"].count()
    stats[cable]['theta']['fill'] += data["t_acat_0 D"].count()
    stats[cable]['gamma']['fill'] += data["g_acat_0 D"].count()
    stats[cable]['theta_gamma']['fill'] += data["tg_acat_0 D"].count()

    stats[cable]['vertical']['nframes'] += data.shape[0]
    stats[cable]['theta']['nframes'] += data.shape[0]
    stats[cable]['gamma']['nframes'] += data.shape[0]
    stats[cable]['theta_gamma']['nframes'] += data.shape[0]

    stats[cable][axis][speed][distance]['vertical']['fill'] += data["vcat_0 D"].count()
    stats[cable][axis][speed][distance]['theta']['fill'] += data["t_acat_0 D"].count()
    stats[cable][axis][speed][distance]['gamma']['fill'] += data["g_acat_0 D"].count()
    stats[cable][axis][speed][distance]['theta_gamma']['fill'] += data["tg_acat_0 D"].count()

    stats[cable][axis][speed][distance]['vertical']['nframes'] += data.shape[0]
    stats[cable][axis][speed][distance]['theta']['nframes'] += data.shape[0]
    stats[cable][axis][speed][distance]['gamma']['nframes'] += data.shape[0]
    stats[cable][axis][speed][distance]['theta_gamma']['nframes'] += data.shape[0]

for cable, cable_data in datas.items():
    print(f'computing data for cable {cable}')
    stats[cable]['vertical']['Q1'] = cable_data['vertical'].quantile(1 / 100)
    stats[cable]['vertical']['Q25'] = cable_data['vertical'].quantile(1 / 4)
    stats[cable]['vertical']['Q50'] = cable_data['vertical'].quantile(2 / 4)
    stats[cable]['vertical']['Q75'] = cable_data['vertical'].quantile(3 / 4)
    stats[cable]['vertical']['Q99'] = cable_data['vertical'].quantile(99 / 100)
    stats[cable]['vertical']['IQR'] = stats[cable]['vertical']['Q75'] - stats[cable]['vertical']['Q25']

    stats[cable]['theta']['Q1'] = cable_data['theta'].quantile(1 / 100)
    stats[cable]['theta']['Q25'] = cable_data['theta'].quantile(1 / 4)
    stats[cable]['theta']['Q50'] = cable_data['theta'].quantile(2 / 4)
    stats[cable]['theta']['Q75'] = cable_data['theta'].quantile(3 / 4)
    stats[cable]['theta']['Q99'] = cable_data['theta'].quantile(99 / 100)
    stats[cable]['theta']['IQR'] = stats[cable]['theta']['Q75'] - stats[cable]['theta']['Q25']

    stats[cable]['gamma']['Q1'] = cable_data['gamma'].quantile(1 / 100)
    stats[cable]['gamma']['Q25'] = cable_data['gamma'].quantile(1 / 4)
    stats[cable]['gamma']['Q50'] = cable_data['gamma'].quantile(2 / 4)
    stats[cable]['gamma']['Q75'] = cable_data['gamma'].quantile(3 / 4)
    stats[cable]['gamma']['Q99'] = cable_data['gamma'].quantile(99 / 100)
    stats[cable]['gamma']['IQR'] = stats[cable]['gamma']['Q75'] - stats[cable]['gamma']['Q25']

    stats[cable]['theta_gamma']['Q1'] = cable_data['theta_gamma'].quantile(1 / 100)
    stats[cable]['theta_gamma']['Q25'] = cable_data['theta_gamma'].quantile(1 / 4)
    stats[cable]['theta_gamma']['Q50'] = cable_data['theta_gamma'].quantile(2 / 4)
    stats[cable]['theta_gamma']['Q75'] = cable_data['theta_gamma'].quantile(3 / 4)
    stats[cable]['theta_gamma']['Q99'] = cable_data['theta_gamma'].quantile(99 / 100)
    stats[cable]['theta_gamma']['IQR'] = stats[cable]['theta_gamma']['Q75'] - stats[cable]['theta_gamma']['Q25']

    files_data = {'x': cable_data['x'], 'y': cable_data['y']}
    for axis, axis_data in files_data.items():
        for speed, speed_data, in axis_data.items():
            for distance, distance_data in speed_data.items():
                if type(distance_data['vertical']) != float:
                    stats[cable][axis][speed][distance]['vertical']['Q1'] = distance_data['vertical'].quantile(1 / 100)
                    stats[cable][axis][speed][distance]['vertical']['Q25'] = distance_data['vertical'].quantile(1 / 4)
                    stats[cable][axis][speed][distance]['vertical']['Q50'] = distance_data['vertical'].quantile(2 / 4)
                    stats[cable][axis][speed][distance]['vertical']['Q75'] = distance_data['vertical'].quantile(3 / 4)
                    stats[cable][axis][speed][distance]['vertical']['Q99'] = distance_data['vertical'].quantile(
                        99 / 100
                    )
                    stats[cable][axis][speed][distance]['vertical']['IQR'] = \
                        stats[cable][axis][speed][distance]['vertical']['Q75'] - \
                        stats[cable][axis][speed][distance]['vertical']['Q25']

                if type(distance_data['theta']) != float:
                    stats[cable][axis][speed][distance]['theta']['Q1'] = distance_data['theta'].quantile(1 / 100)
                    stats[cable][axis][speed][distance]['theta']['Q25'] = distance_data['theta'].quantile(1 / 4)
                    stats[cable][axis][speed][distance]['theta']['Q50'] = distance_data['theta'].quantile(2 / 4)
                    stats[cable][axis][speed][distance]['theta']['Q75'] = distance_data['theta'].quantile(3 / 4)
                    stats[cable][axis][speed][distance]['theta']['Q99'] = distance_data['theta'].quantile(99 / 100)
                    stats[cable][axis][speed][distance]['theta']['IQR'] = stats[cable][axis][speed][distance]['theta'][
                                                                              'Q75'] - \
                                                                          stats[cable][axis][speed][distance]['theta'][
                                                                              'Q25']

                if type(distance_data['gamma']) != float:
                    stats[cable][axis][speed][distance]['gamma']['Q1'] = distance_data['gamma'].quantile(1 / 100)
                    stats[cable][axis][speed][distance]['gamma']['Q25'] = distance_data['gamma'].quantile(1 / 4)
                    stats[cable][axis][speed][distance]['gamma']['Q50'] = distance_data['gamma'].quantile(2 / 4)
                    stats[cable][axis][speed][distance]['gamma']['Q75'] = distance_data['gamma'].quantile(3 / 4)
                    stats[cable][axis][speed][distance]['gamma']['Q99'] = distance_data['gamma'].quantile(99 / 100)
                    stats[cable][axis][speed][distance]['gamma']['IQR'] = stats[cable][axis][speed][distance]['gamma'][
                                                                              'Q75'] - \
                                                                          stats[cable][axis][speed][distance]['gamma'][
                                                                              'Q25']

                if type(distance_data['theta_gamma']) != float:
                    stats[cable][axis][speed][distance]['theta_gamma']['Q1'] = distance_data['theta_gamma'].quantile(
                        1 / 100
                    )
                    stats[cable][axis][speed][distance]['theta_gamma']['Q25'] = distance_data['theta_gamma'].quantile(
                        1 / 4
                    )
                    stats[cable][axis][speed][distance]['theta_gamma']['Q50'] = distance_data['theta_gamma'].quantile(
                        2 / 4
                    )
                    stats[cable][axis][speed][distance]['theta_gamma']['Q75'] = distance_data['theta_gamma'].quantile(
                        3 / 4
                    )
                    stats[cable][axis][speed][distance]['theta_gamma']['Q99'] = distance_data['theta_gamma'].quantile(
                        99 / 100
                    )
                    stats[cable][axis][speed][distance]['theta_gamma']['IQR'] = \
                        stats[cable][axis][speed][distance]['theta_gamma']['Q75'] - \
                        stats[cable][axis][speed][distance]['theta_gamma']['Q25']

for cable, cable_data in stats.items():
    print(f'cable {cable}')
    print(
        f'vertical: {cable_data["vertical"]["fill"]}/{cable_data["vertical"]["nframes"]}; '
        f'Q1={cable_data["vertical"]["Q1"]}; '
        f'Q25={cable_data["vertical"]["Q25"]}; '
        f'Q50={cable_data["vertical"]["Q50"]}; '
        f'Q75={cable_data["vertical"]["Q75"]}; '
        f'Q99={cable_data["vertical"]["Q99"]}'
    )
    print(
        f'theta: {cable_data["theta"]["fill"]}/{cable_data["theta"]["nframes"]}; '
        f'Q1={cable_data["theta"]["Q1"]}; '
        f'Q25={cable_data["theta"]["Q25"]}; '
        f'Q50={cable_data["theta"]["Q50"]}; '
        f'Q75={cable_data["theta"]["Q75"]}; '
        f'Q99={cable_data["theta"]["Q99"]}'
    )
    print(
        f'gamma: {cable_data["gamma"]["fill"]}/{cable_data["gamma"]["nframes"]}; '
        f'Q1={cable_data["gamma"]["Q1"]}; '
        f'Q25={cable_data["gamma"]["Q25"]}; '
        f'Q50={cable_data["gamma"]["Q50"]}; '
        f'Q75={cable_data["gamma"]["Q75"]}; '
        f'Q99={cable_data["gamma"]["Q99"]}'
    )
    print(
        f'theta_gamma: {cable_data["theta_gamma"]["fill"]}/{cable_data["theta_gamma"]["nframes"]}; '
        f'Q1={cable_data["theta_gamma"]["Q1"]}; '
        f'Q25={cable_data["theta_gamma"]["Q25"]}; '
        f'Q50={cable_data["theta_gamma"]["Q50"]}; '
        f'Q75={cable_data["theta_gamma"]["Q75"]}; '
        f'Q99={cable_data["theta_gamma"]["Q99"]}'
    )
    files_data = {'x': cable_data['x'], 'y': cable_data['y']}
    for axis, axis_data in files_data.items():
        print(f'\tdirection {axis}')
        for speed, speed_data, in axis_data.items():
            print(f'\t\tspeed {speed}')
            for distance, distance_data in speed_data.items():
                print(f'\t\t\tdistance {distance}')
                print(
                    f'\t\t\t\tvertical: {distance_data["vertical"]["fill"]}/{distance_data["vertical"]["nframes"]}; '
                    f'Q1={distance_data["vertical"]["Q1"]}; '
                    f'Q25={distance_data["vertical"]["Q25"]}; '
                    f'Q50={distance_data["vertical"]["Q50"]}; '
                    f'Q75={distance_data["vertical"]["Q75"]}; '
                    f'Q99={distance_data["vertical"]["Q99"]}'
                )
                print(
                    f'\t\t\t\ttheta: {distance_data["theta"]["fill"]}/{distance_data["theta"]["nframes"]}; '
                    f'Q1={distance_data["theta"]["Q1"]}; '
                    f'Q25={distance_data["theta"]["Q25"]}; '
                    f'Q50={distance_data["theta"]["Q50"]}; '
                    f'Q75={distance_data["theta"]["Q75"]}; '
                    f'Q99={distance_data["theta"]["Q99"]}'
                )
                print(
                    f'\t\t\t\tgamma: {distance_data["gamma"]["fill"]}/{distance_data["gamma"]["nframes"]}; '
                    f'Q1={distance_data["gamma"]["Q1"]}; '
                    f'Q25={distance_data["gamma"]["Q25"]}; '
                    f'Q50={distance_data["gamma"]["Q50"]}; '
                    f'Q75={distance_data["gamma"]["Q75"]}; '
                    f'Q99={distance_data["gamma"]["Q99"]}'
                )
                print(
                    f'\t\t\t\ttheta_gamma: {distance_data["theta_gamma"]["fill"]}/{distance_data["theta_gamma"]["nframes"]}; '
                    f'Q1={distance_data["theta_gamma"]["Q1"]}; '
                    f'Q25={distance_data["theta_gamma"]["Q25"]}; '
                    f'Q50={distance_data["theta_gamma"]["Q50"]}; '
                    f'Q75={distance_data["theta_gamma"]["Q75"]}; '
                    f'Q99={distance_data["theta_gamma"]["Q99"]}'
                )
