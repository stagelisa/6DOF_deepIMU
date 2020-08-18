import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model

from dataset import *
from util import *
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc', 'cea'], help='Training dataset name (\'oxiod\' or \'euroc\' or \'cea\')')
    parser.add_argument('model', help='Model path')
    parser.add_argument('input', help='Input sequence path (e.g. \"Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv\" for OxIOD, \"MH_02_easy/mav0/imu0/data.csv\" for EuRoC)')
    parser.add_argument('gt', help='Ground truth path (e.g. \"Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv\" for OxIOD, \"MH_02_easy/mav0/state_groundtruth_estimate0/data.csv\" for EuRoC)')
    parser.add_argument('n_points_test', type=int, help='Number of points to visualize')
    args = parser.parse_args()

    window_size = 150
    stride = 10

    model = load_model(args.model)
    # window_size = 200
    # model = create_pred_model_6d_quat(window_size)
    # train_model = create_train_model_6d_quat(model, window_size)

    # train_model.load_weights('./models/model_checkpoint_9DOF.hdf5')
    # model.set_weights(train_model.get_weights()[:-2])
    # model.compile(optimizer=Adam(0.0001), loss='MSE')

    if args.dataset == 'oxiod':
        gyro_data, acc_data, mag_data, pos_data, ori_data = load_oxiod_dataset(args.input, args.gt)
    elif args.dataset == 'euroc':
        gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(args.input, args.gt)
    elif args.dataset == 'cea':
        gyro_data, acc_data, pos_data, ori_data = load_cea_dataset(args.input, args.gt)

    [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)

    if args.dataset == 'oxiod':
        [yhat_delta_p, yhat_delta_q] = model.predict(   [x_gyro[0:args.n_points_test, :, :], x_acc[0:args.n_points_test, :, :], \
                                                        x_mag[0:args.n_points_test, :, :]], batch_size=1, verbose=1)
    elif args.dataset == 'euroc':
        [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=1)

    elif args.dataset == 'cea':
        [yhat_delta_p, yhat_delta_q] = model.predict(   [x_gyro[0:args.n_points_test, :, :], x_acc[0:args.n_points_test, :, :]], batch_size=1, verbose=1)

    gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
    pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, y_delta_q)

    if args.dataset == 'oxiod':
        gt_trajectory = gt_trajectory[0:args.n_points_test, :]
    elif args.dataset == 'cea':
        gt_trajectory = gt_trajectory[0:args.n_points_test, :]

    matplotlib.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=[14.4, 10.8])
    ax = fig.gca(projection='3d')
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], marker='.')
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], marker='.')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    min_x = np.minimum(np.amin(gt_trajectory[:, 0]), np.amin(pred_trajectory[:, 0]))
    min_y = np.minimum(np.amin(gt_trajectory[:, 1]), np.amin(pred_trajectory[:, 1]))
    min_z = np.minimum(np.amin(gt_trajectory[:, 2]), np.amin(pred_trajectory[:, 2]))
    max_x = np.maximum(np.amax(gt_trajectory[:, 0]), np.amax(pred_trajectory[:, 0]))
    max_y = np.maximum(np.amax(gt_trajectory[:, 1]), np.amax(pred_trajectory[:, 1]))
    max_z = np.maximum(np.amax(gt_trajectory[:, 2]), np.amax(pred_trajectory[:, 2]))
    range_x = np.absolute(max_x - min_x)
    range_y = np.absolute(max_y - min_y)
    range_z = np.absolute(max_z - min_z)
    max_range = np.maximum(np.maximum(range_x, range_y), range_z)
    ax.set_xlim(min_x, min_x + max_range)
    ax.set_ylim(min_y, min_y + max_range)
    ax.set_zlim(min_z, min_z + max_range)
    ax.legend(['ground truth', 'predicted'], loc='upper right')

    a = quat_to_euler(generate_orientation(init_q, yhat_delta_q))
    b = quat_to_euler(generate_orientation(init_q, y_delta_q))
    fig, axs = plt.subplots(3)
    axs[0].plot(a[:,0], label="predict_ori_x", marker='.')
    axs[0].plot(b[:,0], label="gt", marker='.')
    axs[0].legend()
    axs[1].plot(a[:,1], label="predict_ori_y", marker='.')
    axs[1].plot(b[:,1], label="gt", marker='.')
    axs[1].legend()
    axs[2].plot(a[:,2], label="predict_ori_z", marker='.')
    axs[2].plot(b[:,2], label="gt", marker='.')
    axs[2].legend()
    plt.show()

if __name__ == '__main__':
    main()