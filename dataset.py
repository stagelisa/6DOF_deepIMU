import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate

from tensorflow.keras.utils import Sequence
from scipy.spatial.transform import Rotation


def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def load_cea_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 1:4]
    
    pos_data = gt_data[:, 1:4]
    ori_data = gt_data[:, 4:8]

    return gyro_data, acc_data, pos_data, ori_data


def force_quaternion_uniqueness(q):

    if np.absolute(q[3]) > 1e-05:
        if q[3] < 0:
            return -q
    elif np.absolute(q[0]) > 1e-05:
        if q[0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q[1]) > 1e-05:
        if q[1] < 0:
            return -q
        else:
            return q
    else:
        if q[2] < 0:
            return -q
        else:
            return q


def cartesian_to_spherical_coordinates(point_cartesian):
    delta_l = np.linalg.norm(point_cartesian)

    if np.absolute(delta_l) > 1e-05:
        theta = np.arccos(point_cartesian[2] / delta_l)
        psi = np.arctan2(point_cartesian[1], point_cartesian[0])
        return delta_l, theta, psi
    else:
        return 0, 0, 0


def load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, [1, 2, 3, 0]]

    #x = []
    x_gyro = []
    x_acc = []
    y_delta_p = []
    y_delta_q = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = Rotation.from_quat(ori_data[idx + window_size//2 - stride//2, [1, 2, 3, 0]])
        q_b = Rotation.from_quat(ori_data[idx + window_size//2 + stride//2, [1, 2, 3, 0]])

        delta_p = (q_a.inv().as_matrix() @ (p_b.T - p_a.T)).T
        delta_q = q_a.inv() * q_b
        delta_q = force_quaternion_uniqueness(delta_q.as_quat())

        y_delta_p.append(delta_p)
        y_delta_q.append(delta_q)


    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    #return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q