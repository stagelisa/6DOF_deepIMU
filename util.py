import numpy as np

from scipy.spatial.transform import Rotation


def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = Rotation.from_quat(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + (cur_q.as_matrix() @ delta_p.T).T
        cur_q = cur_q * Rotation.from_quat(delta_q)
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))


def generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi):
    cur_l = np.array(init_l)
    cur_theta = np.array(init_theta)
    cur_psi = np.array(init_psi)
    pred_l = []
    pred_l.append(np.array(cur_l))

    for [delta_l, delta_theta, delta_psi] in zip(y_delta_l, y_delta_theta, y_delta_psi):
        cur_theta = cur_theta + delta_theta
        cur_psi = cur_psi + delta_psi
        cur_l[0] = cur_l[0] + delta_l * np.sin(cur_theta) * np.cos(cur_psi)
        cur_l[1] = cur_l[1] + delta_l * np.sin(cur_theta) * np.sin(cur_psi)
        cur_l[2] = cur_l[2] + delta_l * np.cos(cur_theta)
        pred_l.append(np.array(cur_l))

    return np.reshape(pred_l, (len(pred_l), 3))

def quat_to_euler(quat):
    euler = []
    for i in range(len(quat)):
        r = Rotation.from_quat(quat[i])
        angle = r.as_euler('xyz', degrees=True)
        euler.append(angle)
    return np.reshape(euler, (len(euler), 3))

def generate_orientation(init_q, y_delta_q):
    cur_q = Rotation.from_quat(init_q)
    pred_q = []
    pred_q.append(cur_q.as_quat())

    for delta_q in y_delta_q:
        cur_q = cur_q * Rotation.from_quat(delta_q)
        pred_q.append(cur_q.as_quat())
    
    return np.reshape(pred_q, (len(pred_q), 4))