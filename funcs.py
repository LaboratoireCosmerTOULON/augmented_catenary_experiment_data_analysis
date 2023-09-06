from math import acos, cos, sin, sqrt

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


def get_named_points_at_index(key: str, df: DataFrame, index: int, n_points: int):
    X, Y, Z = [], [], []

    for i in range(n_points):
        X.append(df[f'{key}_{i} X'][index])
        Y.append(df[f'{key}_{i} Y'][index])
        Z.append(df[f'{key}_{i} Z'][index])

    return X, Y, Z


def get_named_dists_at_index(key: str, df: DataFrame, index: int, n_points: int):
    X, Y, Z, D = [], [], [], []
    for i in range(n_points):
        X.append(df[f'{key}_{i} dX'][index])
        Y.append(df[f'{key}_{i} dY'][index])
        Z.append(df[f'{key}_{i} dZ'][index])
        D.append(df[f'{key}_{i} D'][index])

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
    new_v = pd.Series(np.matmul(R.T, v))
    return new_v


def df_rotate_angle(v, angle, axis):
    if axis == 0:
        R = np.array(
            [[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]]
        )
    elif axis == 1:
        R = np.array(
            [[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]]
        )
    elif axis == 2:
        R = np.array(
            [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
        )
    else:
        raise RuntimeError("axis out of range for rotation in 3D")
    new_v = pd.Series(np.matmul(R.T, v))
    return new_v


def df_compute_catenary(p1, p2, l, d, d0, name):
    p1 /= 1000.
    p2 /= 1000.
    try:
        points, _ = get_coor_marker_points_ideal_catenary(
            p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], l, d, d0, 16
        )
    except:
        points = np.full((16, 3), np.nan)

    if np.any(np.isnan(points)):
        points[:] = np.nan
    for p1, p2 in zip(points[:-1], points[1:]):
        if norm(p1 - p2) > d * 2 or p1[0] > p2[0]:
            points[:] = np.nan
            break

    S = pd.Series(dtype='float64')
    for i, p in enumerate(points):
        S = pd.concat((S, pd.Series(data=p * 1000., index=[f'{name}_{i} X', f'{name}_{i} Y', f'{name}_{i} Z'])))
    return S


def df_compute_distance_to_catenary(X, Xc, Y, Yc, Z, Zc, name):
    dX, dY, dZ = compute_distance_to_catenary(X, Xc, Y, Yc, Z, Zc)
    D = [sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)) for x, y, z in zip(dX, dY, dZ)]
    S = pd.Series(dtype='float64')
    for i in range(len(X)):
        S = pd.concat(
            (S, pd.Series(
                [dX[i], dY[i], dZ[i], D[i]],
                index=[f'{name}_{i} dX', f'{name}_{i} dY', f'{name}_{i} dZ', f'{name}_{i} D']
            ))
        )
    return S


def angles_cost_function(angles, X, Y, Z, L, dL, d0, n_points, exc, eyc, ezc):
    theta, gamma = angles

    cat_R = np.zeros((3, 3))
    cat_R[:, 0] = exc
    cat_R[:, 1] = eyc
    cat_R[:, 2] = ezc

    R_theta = np.array(
        [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    )

    R_gamma = np.array(
        [[1, 0, 0], [0, cos(gamma), -sin(gamma)], [0, sin(gamma), cos(gamma)], ]
    )

    origin = np.array(
        [X[0], Y[0], Z[0]]
    )

    vB = np.array(
        [X[n_points - 1], Y[n_points - 1], Z[n_points - 1]]
    )
    vB = np.matmul(R_theta.T, vB - origin) + origin

    points, _ = get_coor_marker_points_ideal_catenary(
        X[0] / 1000., Y[0] / 1000., Z[0] / 1000., vB[0] / 1000., vB[1] / 1000., vB[2] / 1000., L, dL, d0, n_points
    )

    points *= 1000.

    tilted = np.zeros(points.shape)
    for i, point in enumerate(points):
        tilted[i, :] = np.matmul(cat_R.T, point - origin)
        tilted[i, :] = np.matmul(R_theta, tilted[i, :])
        tilted[i, :] = np.matmul(R_gamma.T, tilted[i, :])
        tilted[i, :] = np.matmul(cat_R, tilted[i, :]) + origin

    mean = 0
    n_not_nan = 0
    for meas, point in zip(zip(X, Y, Z), tilted):
        if np.any(np.isnan(meas)):
            continue
        mean += norm(point - meas)
        n_not_nan += 1

    return mean / n_not_nan


def theta_cost_function(theta, X, Y, Z, L, dL, d0, n_points):
    R = np.array(
        [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    )

    origin = np.array(
        [X[0], Y[0], Z[0]]
    )

    xB = np.array(
        [X[n_points - 1], Y[n_points - 1], Z[n_points - 1]]
    )

    xB = np.matmul(R.T, xB - origin) + origin

    points, _ = get_coor_marker_points_ideal_catenary(
        X[0] / 1000., Y[0] / 1000., Z[0] / 1000., xB[0] / 1000., xB[1] / 1000., xB[2] / 1000., L, dL, d0, n_points
    )
    points *= 1000.

    dist = 0.
    n = 0
    for i, point in enumerate(points):
        measure = np.array([X[i], Y[i], Z[i]])
        temp = np.matmul(R, point - origin) + origin - measure
        if not np.any(np.isnan(temp)):
            dist += norm(temp)
            n += 1

    return dist / n


def tilt_plane_cost_function(abcd, X, Y, Z):
    a, b, c, d = abcd[0], abcd[1], abcd[2], abcd[3]
    dist = 0.
    for x, y, z in zip(X, Y, Z):
        if not pd.isna(x):
            dist += pow((x * a + y * b + z * c) / 1000. + d, 2)
    return dist / len(X)


def df_compute_theta_gamma(X, Y, Z, L, dL, d0, n_points):
    try:
        theta = opt.least_squares(theta_cost_function, 0, args=(X, Y, Z, L, dL, d0, n_points))
        theta = theta.x[0]
        tilt_plane = opt.least_squares(tilt_plane_cost_function, [0, 1, 0, 1], args=(X, Y, Z))
        tilt_normal = tilt_plane.x[:3] / norm(tilt_plane.x[:3])
        if tilt_normal[1] < 0: tilt_normal *= -1
        gamma = acos(tilt_normal[1]) * (-1 if tilt_normal[2] >= 0 else 1)

    except:
        theta = np.nan
        gamma = np.nan

    S = pd.Series((theta, gamma))
    return S


def df_compute_theta_gamma_coupled(X, Y, Z, exc, eyc, ezc, L, dL, d0, n_points):
    try:
        # initial guess for gamma
        tilt_plane = opt.least_squares(tilt_plane_cost_function, [0, 1, 0, 1], args=(X, Y, Z))
        tilt_normal = tilt_plane.x[:3] / norm(tilt_plane.x[:3])
        if tilt_normal[1] < 0: tilt_normal *= -1
        gamma = acos(tilt_normal[1]) * (-1 if tilt_normal[2] >= 0 else 1)

        # optimization
        res = opt.least_squares(
            theta_gamma_cost_function, (0, gamma), args=(X, Y, Z, exc, eyc, ezc, L, dL, d0, n_points)
        )
        theta, gamma = res.x

    except:
        theta = np.nan
        gamma = np.nan

    S = pd.Series((theta, gamma))
    return S


def theta_gamma_cost_function(theta_gamma, X, Y, Z, exc, eyc, ezc, L, dL, d0, n_points):
    theta, gamma = theta_gamma

    R_theta = np.array(
        [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    )

    R_gamma = np.array(
        [[1, 0, 0], [0, cos(gamma), -sin(gamma)], [0, sin(gamma), cos(gamma)]]
    )

    R_cat = np.array(
        [exc, eyc, ezc]
    )

    origin = np.array(
        [X[0], Y[0], Z[0]]
    )

    xB = np.array(
        [X[n_points - 1], Y[n_points - 1], Z[n_points - 1]]
    )

    xB = np.matmul(R_theta.T, xB - origin) + origin

    points, _ = get_coor_marker_points_ideal_catenary(
        X[0] / 1000., Y[0] / 1000., Z[0] / 1000., xB[0] / 1000., xB[1] / 1000., xB[2] / 1000., L, dL, d0, n_points
    )
    points *= 1000.

    dist = 0.
    n = 0
    for i, point in enumerate(points):
        measure = np.array([X[i], Y[i], Z[i]])
        temp = point - origin
        temp = np.matmul(R_theta, temp)
        temp = np.matmul(R_cat, temp)
        temp = np.matmul(R_gamma.T, temp)
        temp = np.matmul(R_cat.T, temp)
        temp += origin - measure
        if not np.any(np.isnan(temp)):
            dist += norm(temp)
            n += 1

    return dist / n
