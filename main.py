import json
import os
import sys
import warnings

from alive_progress import alive_bar

from funcs import *

warnings.simplefilter(action='ignore', category=Warning)

folder = '/ICRA_EXPORT'
# w_path = '../../Users/FILLIUNG Martin/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
w_path = '../../Users/marti/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
path = os.path.abspath(w_path)
if 9 > int(sys.argv[1]) > 0:
    cable = f'dynamique{sys.argv[1]}'
else:
    raise RuntimeError('invalid argument')
directory = sorted(os.listdir(path))
directory = [file for file in directory if '.tsv' in file and cable in file]
print(f'Found {len(directory)} files:')
for i, f in enumerate(directory):
    print(f'\t{i}\t{f}')

with open('file_specific_data.json') as data:
    file_specific_data = json.load(data)

with open('cable_specific_data.json') as data:
    cable_specific_data = json.load(data)

v23 = np.load('v23.npy')
v24 = np.load('v24.npy')

for i, file in enumerate(directory):

    if f'{file[:-4]}_side_only.csv' in os.listdir(path):
        print(f'{file} already processed')
        continue

    with alive_bar(5, title=f'{i + 1}/{len(directory)} {file}', theme='musical') as bar:

        is_float = 'dynamique4' in file
        n_points = 16 if not 'dynamique7' in file else 13

        # cable data
        l = cable_specific_data[file[file.find('dynamique'):file.find('dynamique') + 10]]['L']
        d = cable_specific_data[file[file.find('dynamique'):file.find('dynamique') + 10]]['dL']
        d0 = cable_specific_data[file[file.find('dynamique'):file.find('dynamique') + 10]]['d0']

        if file_specific_data[file]['day'] == '23/11/2022':
            ez = v23
        elif file_specific_data[file]['day'] == '24/11/2022':
            ez = v24
        else:
            raise RuntimeError('Day not found')

        data = pd.read_csv(f'{path}/{file}', sep='\t', header=11)
        dt = 0.01

        # get relevant data
        dataframe = data.loc[:, [
                                    'Time',
                                    'rod_end X', 'rod_end Y', 'rod_end Z',
                                    'robot_cable_attach_point X', 'robot_cable_attach_point Y',
                                    'robot_cable_attach_point Z',
                                    'cable_1 X', 'cable_1 Y', 'cable_1 Z',
                                    'cable_2 X', 'cable_2 Y', 'cable_2 Z',
                                    'cable_3 X', 'cable_3 Y', 'cable_3 Z',
                                    'cable_4 X', 'cable_4 Y', 'cable_4 Z',
                                    'cable_5 X', 'cable_5 Y', 'cable_5 Z',
                                    'cable_6 X', 'cable_6 Y', 'cable_6 Z',
                                    'cable_7 X', 'cable_7 Y', 'cable_7 Z',
                                    'cable_8 X', 'cable_8 Y', 'cable_8 Z',
                                    'cable_9 X', 'cable_9 Y', 'cable_9 Z',
                                    'cable_10 X', 'cable_10 Y', 'cable_10 Z',
                                    'cable_11 X', 'cable_11 Y', 'cable_11 Z',
                                    'cable_12 X', 'cable_12 Y', 'cable_12 Z',
                                    'cable_13 X', 'cable_13 Y', 'cable_13 Z',
                                    'cable_14 X', 'cable_14 Y', 'cable_14 Z'
                                ]]

        _, dataframe['rob_speed X'], _ = alpha_beta_gamma(data['robot_cable_attach_point X'], 0,
                                                          data['Time'][1] - data['Time'][0])

        _, dataframe['rob_speed Y'], _ = alpha_beta_gamma(data['robot_cable_attach_point Y'], 0,
                                                          data['Time'][1] - data['Time'][0])

        _, dataframe['rob_speed Z'], _ = alpha_beta_gamma(data['robot_cable_attach_point Z'], 0,
                                                          data['Time'][1] - data['Time'][0])

        # setting vertical axis for correction across whole sequence
        dataframe[['ezv1', 'ezv2', 'ezv3']] = ez

        # computing catenary axis in qs frame across whole sequence
        dataframe[['caq1', 'caq2', 'caq3']] = np.array([
            dataframe['robot_cable_attach_point X'] - dataframe['rod_end X'],
            dataframe['robot_cable_attach_point Y'] - dataframe['rod_end Y'],
            dataframe['robot_cable_attach_point Z'] - dataframe['rod_end Z']
        ]).T

        # computing vertical correction y axis across whole sequence
        dataframe[['eyv1', 'eyv2', 'eyv3']] = dataframe.apply(lambda row: df_normalized_cross(
            row[['ezv1', 'ezv2', 'ezv3']].to_numpy(),
            row[['caq1', 'caq2', 'caq3']].to_numpy()
        ), axis=1)

        # computing vertical correction x axis across whole sequence
        dataframe[['exv1', 'exv2', 'exv3']] = dataframe.apply(lambda row: df_normalized_cross(
            row[['eyv1', 'eyv2', 'eyv3']].to_numpy(),
            row[['ezv1', 'ezv2', 'ezv3']].to_numpy()
        ), axis=1)

        bar()
        # computing catenary axis in corrected frame across whole sequence
        dataframe[['cav1', 'cav2', 'cav3']] = dataframe.apply(lambda row: df_rotate(
            row[['caq1', 'caq2', 'caq3']],
            row[['exv1', 'exv2', 'exv3']],
            row[['eyv1', 'eyv2', 'eyv3']],
            row[['ezv1', 'ezv2', 'ezv3']]
        ), axis=1)

        # setting catenary y axis across whole sequence
        dataframe[['eyc1', 'eyc2', 'eyc3']] = np.array([0., 1., 0.])

        # computing catenary z axis across whole sequence
        dataframe[['ezc1', 'ezc2', 'ezc3']] = dataframe.apply(lambda row: df_normalized_cross(
            row[['cav1', 'cav2', 'cav3']].to_numpy(),
            row[['eyc1', 'eyc2', 'eyc3']].to_numpy()
        ), axis=1)

        # computing catenary x axis across whole sequence
        dataframe[['exc1', 'exc2', 'exc3']] = dataframe.apply(lambda row: df_normalized_cross(
            row[['eyc1', 'eyc2', 'eyc3']].to_numpy(),
            row[['ezc1', 'ezc2', 'ezc3']].to_numpy()
        ), axis=1)

        bar()
        # compute fully corrected
        Y0 = np.zeros((dataframe.shape[0], 3))
        temp = dataframe.apply(lambda row: df_rotate(
            row[['rod_end X', 'rod_end Y', 'rod_end Z']],
            row[['exv1', 'exv2', 'exv3']],
            row[['eyv1', 'eyv2', 'eyv3']],
            row[['ezv1', 'ezv2', 'ezv3']]
        ), axis=1)
        Y0[:, 1] = temp[1]

        dataframe[['cable_cor_0 X', 'cable_cor_0 Y', 'cable_cor_0 Z']] = temp.to_numpy() - Y0

        for i in range(1, n_points - 1):
            dataframe[[f'cable_cor_{i} X', f'cable_cor_{i} Y', f'cable_cor_{i} Z']] \
                = dataframe.apply(lambda row: df_rotate(
                row[[f'cable_{i} X', f'cable_{i} Y', f'cable_{i} Z']],
                row[['exv1', 'exv2', 'exv3']],
                row[['eyv1', 'eyv2', 'eyv3']],
                row[['ezv1', 'ezv2', 'ezv3']]
            ), axis=1).to_numpy() - Y0

        dataframe[[
            f'cable_cor_{n_points - 1} X',
            f'cable_cor_{n_points - 1} Y',
            f'cable_cor_{n_points - 1} Z'
        ]] = dataframe.apply(lambda row: df_rotate(
            row[['robot_cable_attach_point X', 'robot_cable_attach_point Y', 'robot_cable_attach_point Z']],
            row[['exv1', 'exv2', 'exv3']],
            row[['eyv1', 'eyv2', 'eyv3']],
            row[['ezv1', 'ezv2', 'ezv3']]
        ), axis=1).to_numpy() - Y0

        if is_float:
            bar()
            # compute theta and gamma
            for i in range(n_points):
                dataframe[f"cable_cor_inv_{i} X"] = dataframe[f"cable_cor_{i} X"]
                dataframe[f"cable_cor_inv_{i} Y"] = - dataframe[f"cable_cor_{i} Y"]
                dataframe[f"cable_cor_inv_{i} Z"] = - dataframe[f"cable_cor_{i} Z"]


            dataframe[['Theta', 'Gamma']] = \
                dataframe.apply(lambda row: df_compute_theta_gamma(
                    np.array([row[f'cable_cor_inv_{i} X'] for i in range(n_points)]),
                    np.array([row[f'cable_cor_inv_{i} Y'] for i in range(n_points)]),
                    np.array([row[f'cable_cor_inv_{i} Z'] for i in range(n_points)]),
                    l, d, d0, n_points
                ), axis=1)

            # for side tilt only
            dataframe['Theta'] = 0.

            bar()
            # compute catenary
            dataframe = pd.merge(dataframe, dataframe.apply(lambda row: df_compute_catenary(
                row[[
                    f'cable_cor_inv_{0} X',
                    f'cable_cor_inv_{0} Y',
                    f'cable_cor_inv_{0} Z'
                ]].to_numpy(),
                row[[
                    f'cable_cor_inv_{n_points - 1} X',
                    f'cable_cor_inv_{n_points - 1} Y',
                    f'cable_cor_inv_{n_points - 1} Z'
                ]].to_numpy(),
                l, d, d0, 'vcat_inv'
            ), axis=1), left_index=True, right_index=True)

            dataframe = pd.merge(dataframe, dataframe.apply(lambda row: df_compute_catenary(
                row[[
                    f'cable_cor_inv_{0} X',
                    f'cable_cor_inv_{0} Y',
                    f'cable_cor_inv_{0} Z'
                ]].to_numpy(),
                df_rotate_angle(
                    row[[
                        f'cable_cor_inv_{n_points - 1} X',
                        f'cable_cor_inv_{n_points - 1} Y',
                        f'cable_cor_inv_{n_points - 1} Z'
                    ]].to_numpy() - row[[
                        f'cable_cor_inv_{0} X',
                        f'cable_cor_inv_{0} Y',
                        f'cable_cor_inv_{0} Z'
                    ]].to_numpy(),
                    row['Theta'],
                    1
                ) + row[[
                    f'cable_cor_inv_{0} X',
                    f'cable_cor_inv_{0} Y',
                    f'cable_cor_inv_{0} Z'
                ]].to_numpy(),
                l, d, d0, 'v_robot_cat'
            ), axis=1), left_index=True, right_index=True)

            for i in range(n_points):
                dataframe[[f'tcat_inv_{i} X', f'tcat_inv_{i} Y', f'tcat_inv_{i} Z']] \
                    = dataframe.apply(lambda row:
                                      pd.Series(
                                          df_rotate(
                                              df_rotate_angle(
                                                  df_rotate(
                                                      df_rotate_angle(
                                                          row[[
                                                              f'v_robot_cat_{i} X',
                                                              f'v_robot_cat_{i} Y',
                                                              f'v_robot_cat_{i} Z'
                                                          ]].to_numpy() - row[[
                                                              f'cable_cor_inv_{0} X',
                                                              f'cable_cor_inv_{0} Y',
                                                              f'cable_cor_inv_{0} Z'
                                                          ]].to_numpy(),
                                                          - row['Theta'],
                                                          1
                                                      ),
                                                      np.array([- row['exc1'], row['exc2'], row['exc3']]),
                                                      np.array([- row['eyc1'], row['eyc2'], row['eyc3']]),
                                                      np.array([- row['ezc1'], row['ezc2'], row['ezc3']])
                                                  ),
                                                  row['Gamma'],
                                                  0
                                              ),
                                              np.array([- row['exc1'], - row['eyc1'], - row['ezc1']]),
                                              np.array([row['exc2'], row['eyc2'], row['ezc2']]),
                                              np.array([row['exc3'], row['eyc3'], row['ezc3']]),
                                          ).to_numpy() +
                                          row[[
                                              f'cable_cor_inv_{0} X',
                                              f'cable_cor_inv_{0} Y',
                                              f'cable_cor_inv_{0} Z'
                                          ]].to_numpy()
                                          , index=[f'tcat_{i} X', f'tcat_{i} Y', f'tcat_{i} Z'])
                                      , axis=1)


            for i in range(n_points):
                dataframe[f"vcat_{i} X"] = dataframe[f"vcat_inv_{i} X"]
                dataframe[f"vcat_{i} Y"] = - dataframe[f"vcat_inv_{i} Y"]
                dataframe[f"vcat_{i} Z"] = - dataframe[f"vcat_inv_{i} Z"]
                dataframe[f"tcat_{i} X"] = dataframe[f"tcat_inv_{i} X"]
                dataframe[f"tcat_{i} Y"] = - dataframe[f"tcat_inv_{i} Y"]
                dataframe[f"tcat_{i} Z"] = - dataframe[f"tcat_inv_{i} Z"]

        else:
            bar()
            # compute theta and gamma
            dataframe[['Theta', 'Gamma']] = \
                dataframe.apply(lambda row: df_compute_theta_gamma(
                np.array([row[f'cable_cor_{i} X'] for i in range(n_points)]),
                np.array([row[f'cable_cor_{i} Y'] for i in range(n_points)]),
                np.array([row[f'cable_cor_{i} Z'] for i in range(n_points)]),
                l, d, d0, n_points
            ), axis=1)

            # # for side tilt only
            dataframe['Theta'] = 0.

            bar()
            # compute catenary
            dataframe = pd.merge(dataframe, dataframe.apply(lambda row: df_compute_catenary(
                row[[
                    f'cable_cor_{0} X',
                    f'cable_cor_{0} Y',
                    f'cable_cor_{0} Z'
                ]].to_numpy(),
                row[[
                    f'cable_cor_{n_points - 1} X',
                    f'cable_cor_{n_points - 1} Y',
                    f'cable_cor_{n_points - 1} Z'
                ]].to_numpy(),
                l, d, d0, 'vcat'
            ), axis=1), left_index=True, right_index=True)

            dataframe = pd.merge(dataframe, dataframe.apply(lambda row: df_compute_catenary(
                row[[
                    f'cable_cor_{0} X',
                    f'cable_cor_{0} Y',
                    f'cable_cor_{0} Z'
                ]].to_numpy(),
                df_rotate_angle(
                    row[[
                        f'cable_cor_{n_points - 1} X',
                        f'cable_cor_{n_points - 1} Y',
                        f'cable_cor_{n_points - 1} Z'
                    ]].to_numpy() - row[[
                        f'cable_cor_{0} X',
                        f'cable_cor_{0} Y',
                        f'cable_cor_{0} Z'
                    ]].to_numpy(),
                    row['Theta'],
                    1
                ) + row[[
                    f'cable_cor_{0} X',
                    f'cable_cor_{0} Y',
                    f'cable_cor_{0} Z'
                ]].to_numpy(),
                l, d, d0, 'v_robot_cat'
            ), axis=1), left_index=True, right_index=True)

            for i in range(n_points):
                dataframe[[f'tcat_{i} X', f'tcat_{i} Y', f'tcat_{i} Z']] \
                    = dataframe.apply(lambda row:
                                      pd.Series(
                                          df_rotate(
                                              df_rotate_angle(
                                                  df_rotate_angle(
                                                      df_rotate(
                                                          row[[f'v_robot_cat_{i} X', f'v_robot_cat_{i} Y',
                                                               f'v_robot_cat_{i} Z']].to_numpy() -
                                                          row[[
                                                              f'cable_cor_{0} X',
                                                              f'cable_cor_{0} Y',
                                                              f'cable_cor_{0} Z'
                                                          ]].to_numpy(),
                                                          row[['exc1', 'exc2', 'exc3']],
                                                          row[['eyc1', 'eyc2', 'eyc3']],
                                                          row[['ezc1', 'ezc2', 'ezc3']]
                                                      ),
                                                      - row['Theta'],
                                                      1
                                                  ),
                                                  row['Gamma'],
                                                  0
                                              ),
                                              row[['exc1', 'eyc1', 'ezc1']],
                                              row[['exc2', 'eyc2', 'ezc2']],
                                              row[['exc3', 'eyc3', 'ezc3']]
                                          ).to_numpy() +
                                          row[[
                                              f'cable_cor_{0} X',
                                              f'cable_cor_{0} Y',
                                              f'cable_cor_{0} Z'
                                          ]].to_numpy()
                                          , index=[f'tcat_{i} X', f'tcat_{i} Y', f'tcat_{i} Z'])
                                      , axis=1)

        bar()
        # compute distance of measure to catenary
        dataframe = pd.merge(dataframe,
                             dataframe.apply(lambda row:
                                             df_compute_distance_to_catenary(
                                                 row[[f'cable_cor_{i} X' for i in range(n_points)]],
                                                 row[[f'vcat_{i} X' for i in range(n_points)]],
                                                 row[[f'cable_cor_{i} Y' for i in range(n_points)]],
                                                 row[[f'vcat_{i} Y' for i in range(n_points)]],
                                                 row[[f'cable_cor_{i} Z' for i in range(n_points)]],
                                                 row[[f'vcat_{i} Z' for i in range(n_points)]],
                                                 'vcat'
                                             )
                                             , axis=1),
                             left_index=True, right_index=True)

        dataframe = pd.merge(dataframe,
                             dataframe.apply(lambda row:
                                             df_compute_distance_to_catenary(
                                                 row[[f'cable_cor_{i} X' for i in range(n_points)]],
                                                 row[[f'tcat_{i} X' for i in
                                                      range(n_points)]],
                                                 row[[f'cable_cor_{i} Y' for i in range(n_points)]],
                                                 row[[f'tcat_{i} Y' for i in
                                                      range(n_points)]],
                                                 row[[f'cable_cor_{i} Z' for i in range(n_points)]],
                                                 row[[f'tcat_{i} Z' for i in
                                                      range(n_points)]],
                                                 'tcat'
                                             )
                                             , axis=1),
                             left_index=True, right_index=True)

        dataframe.to_csv(f'{path}/{file[:-4]}_side_only.csv')
