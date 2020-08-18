import tfquaternion as tfq
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras import backend as K
from dataset import *
from util import *

def quaternion_phi_3_error(y_true, y_pred):
    assert 0 == 1
    return tf.acos(K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1)))


def quaternion_phi_4_error(y_true, y_pred):
    assert 0 == 1
    return 1 - K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1))


def quaternion_log_phi_4_error(y_true, y_pred):
    assert 0 == 1
    return K.log(1e-4 + quaternion_phi_4_error(y_true, y_pred))


def quat_mult_error(y_true, y_pred):
    q_hat = tfq.Quaternion(tf.gather(y_true, [3, 0, 1, 2], axis=1))
    q = tfq.Quaternion(tf.gather(y_pred, [3, 0, 1, 2], axis=1)).normalized()
    q_prod = q * q_hat.conjugate()
    w, x, y, z = tf.split(q_prod, num_or_size_splits=4, axis=-1)
    return tf.abs(tf.multiply(2.0, tf.concat(values=[x, y, z], axis=-1)))


def quaternion_mean_multiplicative_error(y_true, y_pred):
    return tf.reduce_mean(quat_mult_error(y_true, y_pred))


# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
    #def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'nb_outputs': self.nb_outputs,
            'is_placeholder': self.is_placeholder
        })
        return config
    
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0

        #for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #    precision = K.exp(-log_var[0])
        #    loss += K.sum(precision * (y_true - y_pred)**2., -1) + log_var[0]

        precision = K.exp(-self.log_vars[0][0])
        loss += precision * mean_absolute_error(ys_true[0], ys_pred[0]) + self.log_vars[0][0]
        precision = K.exp(-self.log_vars[1][0])
        loss += precision * quaternion_mean_multiplicative_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        #loss += precision * quaternion_phi_4_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]

        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


def create_pred_model_6d_quat(window_size=200):
    #inp = Input((window_size, 6), name='inp')
    #lstm1 = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    convA1 = Conv1D(128, 11)(x1)
    convA2 = Conv1D(128, 11)(convA1)
    poolA = MaxPooling1D(3)(convA2)
    convB1 = Conv1D(128, 11)(x2)
    convB2 = Conv1D(128, 11)(convB1)
    poolB = MaxPooling1D(3)(convB2)
    AB = concatenate([poolA, poolB])
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(AB)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(LSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    
    y1_pred = Dense(3)(drop2)
    y2_pred = Dense(4)(drop2)

    #model = Model(inp, [y1_pred, y2_pred])
    model = Model([x1, x2], [y1_pred, y2_pred])

    model.summary()
    
    return model


def create_train_model_6d_quat(pred_model, window_size=200):
    #inp = Input(shape=(window_size, 6), name='inp')
    #y1_pred, y2_pred = pred_model(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    y1_pred, y2_pred = pred_model([x1, x2])
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    #train_model = Model([inp, y1_true, y2_true], out)
    train_model = Model([x1, x2, y1_true, y2_true], out)
    train_model.summary()
    return train_model