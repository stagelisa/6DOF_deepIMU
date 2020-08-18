import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model
from scipy.spatial.transform import Rotation

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
    pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

    if args.dataset == 'oxiod':
        gt_trajectory = gt_trajectory[0:args.n_points_test, :]
    elif args.dataset == 'cea':
        gt_trajectory = gt_trajectory[0:args.n_points_test, :]


    df = pd.read_csv(args.input)
    df2 = pd.read_csv(args.gt)
    # n = 50
    # g = np.zeros((n, 3))
    luu = np.zeros((df.shape[0], 3))
    intg1 = np.zeros((df.shape[0], 3))
    intg2 = np.zeros((df.shape[0], 3))
    time = df.iloc[:,0]

    # for i in range(n):
    #     v = np.array([*df.iloc[i, 1:4]])
    #     a = Rotation.from_quat(df2.iloc[i, [5, 6, 7, 4]])
    #     b = a.as_matrix() @ v
    #     g[i] = b
    # true_g = np.mean(g, axis=0)
    true_g = np.array([-0.3, 0.1, 9.77])
    print(true_g)
    for i in range(df.shape[0]):
        v = np.array([*df.iloc[i, 1:4]])
        a = Rotation.from_quat(df2.iloc[i, [5, 6, 7, 4]])
        b = a.as_matrix() @ v
        luu[i] = b - true_g
        if i%100 == 0:
            print(f"{i} ----------------------------- {luu[i] + true_g}")
    for i in range(1, time.shape[0]):
        intg1[i] = intg1[i-1] + 0.5 * (luu[i] + luu[i-1]) * (time.iloc[i] - time.iloc[i-1])
        
    for i in range(1, time.shape[0]):
        intg2[i] = intg2[i-1] + 0.5 * (intg1[i] + intg1[i-1]) * (time.iloc[i] - time.iloc[i-1]) 
    intg2 = intg2[window_size//2 - 5:intg2.shape[0] - window_size//2 + 5 + 1]
    intg2 = intg2 + np.array(df2.iloc[window_size//2 - 5, 1:4]) - intg2[0]


    matplotlib.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=[14.4, 10.8])
    ax = fig.gca(projection='3d')
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], marker='.')
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], marker='.')
    ax.plot(intg2[:, 0], intg2[:, 1], intg2[:, 2], marker='.')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    print(np.amin(intg2[:, 0]))
    min_x = np.minimum(np.amin(gt_trajectory[:, 0]), np.amin(pred_trajectory[:, 0]))
    # min_x = np.minimum(min_x, np.amin(intg2[:, 0]))
    min_y = np.minimum(np.amin(gt_trajectory[:, 1]), np.amin(pred_trajectory[:, 1]))
    # min_y = np.minimum(min_y, np.amin(intg2[:, 1]))
    min_z = np.minimum(np.amin(gt_trajectory[:, 2]), np.amin(pred_trajectory[:, 2]))
    # min_z = np.minimum(min_z, np.amin(intg2[:, 2]))
    max_x = np.maximum(np.amax(gt_trajectory[:, 0]), np.amax(pred_trajectory[:, 0]))
    # max_x = np.maximum(max_x, np.amax(intg2[:, 0]))
    max_y = np.maximum(np.amax(gt_trajectory[:, 1]), np.amax(pred_trajectory[:, 1]))
    # max_y = np.maximum(max_y, np.amax(intg2[:, 1]))
    max_z = np.maximum(np.amax(gt_trajectory[:, 2]), np.amax(pred_trajectory[:, 2]))
    # max_z = np.maximum(max_z, np.amax(intg2[:, 2]))
    range_x = np.absolute(max_x - min_x)
    range_y = np.absolute(max_y - min_y)
    range_z = np.absolute(max_z - min_z)
    max_range = np.maximum(np.maximum(range_x, range_y), range_z)
    ax.set_xlim(min_x, min_x + max_range)
    ax.set_ylim(min_y, min_y + max_range)
    ax.set_zlim(min_z, min_z + max_range)
    ax.legend(['ground truth', 'predicted', 'integral'], loc='upper right')

    fig, axs = plt.subplots(3)
    axs[0].plot(gt_trajectory[:, 0], label='gt_x', marker='.')
    axs[0].plot(pred_trajectory[:, 0], label='pred_x', marker='.')
    axs[0].plot(intg2[::10, 0], label='intergral_x', marker='.')
    axs[1].plot(gt_trajectory[:, 1], label='gt_y', marker='.')
    axs[1].plot(pred_trajectory[:, 1], label='pred_y', marker='.')
    axs[1].plot(intg2[::10, 1], label='intergral_y', marker='.')
    axs[2].plot(gt_trajectory[:, 2], label='gt_z', marker='.')
    axs[2].plot(pred_trajectory[:, 2], label='pred_z', marker='.')
    axs[2].plot(intg2[::10, 2], label='intergral_z', marker='.')
    # axs[2].set_ylim([-0.05, 0.1])
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    axs[2].legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()