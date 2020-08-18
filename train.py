import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.utils import shuffle

from time import time

from dataset import *
from model import *
from util import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Model output name')
    args = parser.parse_args()

    np.random.seed(0)

    window_size = 150
    stride = 10

    x_gyro = []
    x_acc = []

    y_delta_p = []
    y_delta_q = []

    imu_data_filenames = []
    gt_data_filenames = []

    for i in range(4):
        data_imu_path = f'/home/huydung/devel/intern/data/2bis/{i}/data_deep/imu/'
        data_gt_path = f'/home/huydung/devel/intern/data/2bis/{i}/data_deep/gt/'
        for j in range(len([name for name in os.listdir(data_imu_path) if os.path.isfile(os.path.join(data_imu_path, name))])):
            imu_data_filenames.append(data_imu_path + f'{j}.csv')
            gt_data_filenames.append(data_gt_path + f'{j}.csv')

    for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):
        cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_cea_dataset(cur_imu_data_filename, cur_gt_data_filename)
        [cur_x_gyro, cur_x_acc], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)
        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)
        y_delta_p.append(cur_y_delta_p)
        y_delta_q.append(cur_y_delta_q)

    
    x_gyro = np.vstack(x_gyro)
    x_acc = np.vstack(x_acc)

    y_delta_p = np.vstack(y_delta_p)
    y_delta_q = np.vstack(y_delta_q)

    x_gyro, x_acc, y_delta_p, y_delta_q = shuffle(x_gyro, x_acc, y_delta_p, y_delta_q)

    initial_learning_rate = 3e-4
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.97,
        staircase=True)
    pred_model = create_pred_model_6d_quat(window_size)
    train_model = create_train_model_6d_quat(pred_model, window_size)
    train_model.compile(optimizer=Adam(lr_schedule), loss=None)

    filepath = "model_checkpoint.hdf5"
    model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), profile_batch=0)

    try:
        history = train_model.fit([x_gyro, x_acc, y_delta_p, y_delta_q], epochs=200, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)
        train_model.load_weights(filepath)
        train_model.save('last_best_model_with_custom_layer.hdf5')
        pred_model = create_pred_model_6d_quat(window_size)
        pred_model.set_weights(train_model.get_weights()[:-2])
        pred_model.save('%s.hdf5' % args.output)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    except KeyboardInterrupt:
        train_model.load_weights(filepath)
        train_model.save('last_best_model_with_custom_layer.hdf5')
        pred_model = create_pred_model_6d_quat(window_size)
        pred_model.set_weights(train_model.get_weights()[:-2])
        pred_model.save('%s.hdf5' % args.output)
        print('Early terminate')

    print('Training complete')

if __name__ == '__main__':
    main()