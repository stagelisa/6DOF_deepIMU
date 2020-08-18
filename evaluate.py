import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

from dataset import *
from util import *
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model path')
    args = parser.parse_args()

    model = load_model(args.model)

    window_size = 150
    stride = 10

    imu_data_filenames = []
    gt_data_filenames = []

    for i in range(16, 18):
        data_imu_path = f'/home/huydung/devel/intern/data/3eme/{i}/data_deep/imu/'
        data_gt_path = f'/home/huydung/devel/intern/data/3eme/{i}/data_deep/gt/'
        for j in range(len([name for name in os.listdir(data_imu_path) if os.path.isfile(os.path.join(data_imu_path, name))])):
            imu_data_filenames.append(data_imu_path + f'{j}.csv')
            gt_data_filenames.append(data_gt_path + f'{j}.csv')
    count = 1
    traj, x_rmses, y_rmses, z_rmses = [], [], [], []
    for (cur_imu_data_filename, cur_gt_data_filename) in zip(imu_data_filenames, gt_data_filenames):
        gyro_data, acc_data, pos_data, ori_data = load_cea_dataset(cur_imu_data_filename, cur_gt_data_filename)

        [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
        
        [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=0)

        print(yhat_delta_p.shape)
        gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
        pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

        pred_trajectory = pred_trajectory
        gt_trajectory = gt_trajectory

        trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))
        x_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, 0] - gt_trajectory[:, 0], axis=-1))))
        y_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, 1] - gt_trajectory[:, 1], axis=-1))))
        z_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, 2] - gt_trajectory[:, 2], axis=-1))))

        print(f'Trajectory RMSE, sequence {cur_imu_data_filename}: {trajectory_rmse}\nx:{x_rmse}\ty:{y_rmse}\tz:{z_rmse}')
        traj.append(trajectory_rmse)
        if trajectory_rmse > 0.06:
            print('\n\n\n')
            print(count)
            count += 1
        x_rmses.append(x_rmse)
        y_rmses.append(y_rmse)
        z_rmses.append(z_rmse)
    
    plt.figure()
    plt.boxplot([traj, x_rmses, y_rmses, z_rmses], labels=['trajectory rmse', 'x rmse', 'y rmse', 'z rmse'])
    plt.show()

if __name__ == '__main__':
    main()