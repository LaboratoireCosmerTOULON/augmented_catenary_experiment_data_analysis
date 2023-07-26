from math import sin, cos, pi, sqrt

import pandas as pd
from numpy import nan
from numpy.linalg import norm
from pandas import DataFrame

from calc_catenary_from_ext_points import *


def get_cable_points_at_index(df: DataFrame, index: int):
    X, Y, Z = [], [], []

    X.append(df['rod_end X'][index])
    Y.append(df['rod_end Y'][index])
    Z.append(df['rod_end Z'][index])

    for i in range(1, 15):
        X.append(df['cable_' + str(i) + ' X'][index])
        Y.append(df['cable_' + str(i) + ' Y'][index])
        Z.append(df['cable_' + str(i) + ' Z'][index])

    X.append(df['robot_cable_attach_point X'][index])
    Y.append(df['robot_cable_attach_point Y'][index])
    Z.append(df['robot_cable_attach_point Z'][index])

    return X, Y, Z


def get_vcable_points_at_index(df: DataFrame, index: int, n_points):
    X, Y, Z = [], [], []

    for i in range(n_points):
        X.append(df[f'cable_cor_{i} X'][index])
        Y.append(df[f'cable_cor_{i} Y'][index])
        Z.append(df[f'cable_cor_{i} Z'][index])

    return X, Y, Z


def get_vcat_points_at_index(df: DataFrame, index: int, n_points):
    X, Y, Z = [], [], []
    for i in range(n_points):
        X.append(df[f'vcat_{i} X'][index])
        Y.append(df[f'vcat_{i} Y'][index])
        Z.append(df[f'vcat_{i} Z'][index])

    return X, Y, Z


def get_tcat_points_at_index(df: DataFrame, index: int, n_points):
    X, Y, Z = [], [], []
    for i in range(n_points):
        X.append(df[f'tcat_{i} X'][index])
        Y.append(df[f'tcat_{i} Y'][index])
        Z.append(df[f'tcat_{i} Z'][index])

    return X, Y, Z


def get_dist_to_vcat_at_index(df: DataFrame, index: int, n_points):
    X, Y, Z, D = [], [], [], []
    for i in range(n_points):
        X.append(df[f'vcat_{i} dX'][index])
        Y.append(df[f'vcat_{i} dY'][index])
        Z.append(df[f'vcat_{i} dZ'][index])
        D.append(df[f'vcat_{i} D'][index])

    return X, Y, Z, D


def get_dist_to_tcat_at_index(df: DataFrame, index: int, n_points):
    X, Y, Z, D = [], [], [], []
    for i in range(n_points):
        X.append(df[f'tcat_{i} dX'][index])
        Y.append(df[f'tcat_{i} dY'][index])
        Z.append(df[f'tcat_{i} dZ'][index])
        D.append(df[f'tcat_{i} D'][index])

    return X, Y, Z, D


def compute_distance_to_catenary(X, Xc, Y, Yc, Z, Zc):
    DistancesX = []
    DistancesY = []
    DistancesZ = []
    for x, y, z, xc, yc, zc in zip(X, Y, Z, Xc, Yc, Zc):
        DistancesX.append(nan)
        DistancesY.append(nan)
        DistancesZ.append(nan)
        if x != 0. or y != 0. or z != 0.:
            DistancesX[-1] = x - xc
            DistancesY[-1] = y - yc
            DistancesZ[-1] = z - zc
    return DistancesX, DistancesY, DistancesZ


def alpha_beta_gamma(X: list, V0: float, dt: float, alpha: float = .1, beta: float = .1):
    assert 0 < alpha < 1
    assert 0 < beta < 1
    assert 0 < 4 - 2 * alpha - beta
    assert dt > 0
    gamma = pow(beta, 2) / (2 * alpha)
    x_est, v_est, a_est = [X[0]], [V0], [0]
    for x in X[1:]:
        x_est.append(x_est[-1] + dt * v_est[-1])
        v_est.append(v_est[-1])
        a_est.append(a_est[-1])
        r = x - x_est[-1]
        x_est[-1] += alpha * r
        v_est[-1] += beta * r / dt
        a_est[-1] += 2 * gamma * r / pow(r, 2)
    return x_est, v_est, a_est


def df_normalized_cross(v1, v2):
    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)
    return pd.Series(v3)


def df_dot(v1, v2):
    v3 = np.dot(v1, v2)
    return pd.Series(v3)


def df_rotate(v, ex, ey, ez):
    R = np.zeros((3, 3))
    R[:, 0] = ex
    R[:, 1] = ey
    R[:, 2] = ez
    new_v = pd.Series(np.matmul(v, R))
    return new_v


def df_rotate_angle(v, angle, axis):
    if axis == 0:
        R = np.array([
            [1, 0, 0],
            [0, cos(angle), -sin(angle)],
            [0, sin(angle), cos(angle)],
        ])
    elif axis == 1:
        R = np.array([
            [cos(angle), 0, sin(angle)],
            [0, 1, 0],
            [-sin(angle), 0, cos(angle)]
        ])
    elif axis == 2:
        R = np.array([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise RuntimeError("axis out of range for rotation in 3D")
    return pd.Series(np.matmul(v, R))


def df_compute_catenary(p1, p2, l, d, d0, name):
    p1 /= 1000.
    p2 /= 1000.
    points, _ = get_coor_marker_points_ideal_catenary(
        p1[0], p1[1], p1[2],
        p2[0], p2[1], p2[2],
        l, d, d0, 16
    )

    S = pd.Series(dtype='float64')
    for i, p in enumerate(points):
        S = pd.concat((S, pd.Series(data=p * 1000., index=[f'{name}_{i} X', f'{name}_{i} Y', f'{name}_{i} Z'])))
    return S


def df_compute_distance_to_catenary(X, Xc, Y, Yc, Z, Zc, name):
    dX, dY, dZ = compute_distance_to_catenary(X, Xc, Y, Yc, Z, Zc)
    D = [sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)) for x, y, z in zip(dX, dY, dZ)]
    S = pd.Series(dtype='float64')
    for i in range(len(X)):
        S = pd.concat((S, pd.Series([dX[i], dY[i], dZ[i], D[i]],
                                    index=[f'{name}_{i} dX', f'{name}_{i} dY', f'{name}_{i} dZ', f'{name}_{i} D'])))
    return S


def angles_cost_function(angles, X, Y, Z, L, dL, d0, is_float, n_points, exc, eyc, ezc):
    theta, gamma = angles

    cat_R = np.zeros((3, 3))
    cat_R[:, 0] = exc
    cat_R[:, 1] = eyc
    cat_R[:, 2] = ezc

    R_theta = np.array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    ])

    R_gamma = np.array([
        [1, 0, 0],
        [0, cos(gamma), -sin(gamma)],
        [0, sin(gamma), cos(gamma)],
    ])

    origin = np.array([
        X[0 if not is_float else 15],
        Y[0 if not is_float else 15],
        Z[0 if not is_float else 15]
    ])

    vB = np.array([
        X[n_points - 1 if not is_float else 0],
        Y[n_points - 1 if not is_float else 0],
        Z[n_points - 1 if not is_float else 0]
    ])
    vB = np.matmul(vB - origin, R_theta.T) + origin

    points, _ = get_coor_marker_points_ideal_catenary(
        X[0 if not is_float else 15] / 1000., Y[0 if not is_float else 15] / 1000., Z[0 if not is_float else 15] / 1000.,
        vB[0] / 1000., vB[1] / 1000., vB[2] / 1000.,
        L, dL, d0, n_points
    )
    points *= 1000.

    tilted = np.zeros(points.shape)
    for i, point in enumerate(points):
        tilted[i, :] = np.matmul(point - origin, cat_R)
        tilted[i, :] = np.matmul(tilted[i, :], R_theta)
        tilted[i, :] = np.matmul(tilted[i, :], R_gamma)
        tilted[i, :] = np.matmul(tilted[i, :], cat_R.T) + origin

    mean = 0
    n_not_nan = 0
    for meas, point in zip(zip(X, Y, Z), tilted):
        if np.any(np.isnan(meas)):
            continue
        mean += norm(point - meas)
        n_not_nan += 1

    return mean / n_not_nan

def df_compute_theta_gamma(X, Y, Z, L, dL, d0, is_float, n_points, exc, eyc, ezc):
    res = opt.minimize(
        angles_cost_function,
        np.array([0., 0.]),
        (X, Y, Z, L, dL, d0, is_float, n_points, exc, eyc, ezc),
        bounds=([-pi/2, pi/2], [-pi/2, pi/2])
    )
    theta, gamma = res.x
    S = pd.Series((theta, gamma))
    return S
