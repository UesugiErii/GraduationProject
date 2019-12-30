import time
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, LSTMCell
import numpy as np
from config import *
from datetime import datetime


index = 0
if index == 0:
    restore = False
else:
    restore = True
restore_weight_dir = "./logs/weight/{}".format(index)


class selfattLayer(tf.keras.layers.Layer):
    def __init__(self, original_features, attention_features=None):
        super(selfattLayer, self).__init__()

        if not attention_features:
            attention_features = original_features // 4

        self.f = tf.keras.layers.Conv2D(
            filters=attention_features,
            kernel_size=(1, 1),
            strides=(1, 1),
        )
        self.g = tf.keras.layers.Conv2D(
            filters=attention_features,
            kernel_size=(1, 1),
            strides=(1, 1),
        )
        self.h = tf.keras.layers.Conv2D(
            filters=original_features,
            kernel_size=(1, 1),
            strides=(1, 1),
        )
        self.scale = tf.Variable(0., trainable=True)

    def call(self, x):
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        shape = f.shape
        f_flatten = tf.reshape(f, (shape[0], shape[1] * shape[2], shape[3]))  # (*,hw,c')
        shape = g.shape
        g_flatten = tf.reshape(g, (shape[0], shape[1] * shape[2], shape[3]))  # (*,hw,c')
        shape = h.shape
        h_flatten = tf.reshape(h, (shape[0], shape[1] * shape[2], shape[3]))  # (*,hw,c)

        s = tf.matmul(g_flatten, f_flatten, transpose_b=True)  # (hw,hw)
        b = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(b, h_flatten)  # (hw,c)
        y = self.scale * tf.reshape(o, x.shape) + x

        return y


class selfattLayer2(tf.keras.layers.Layer):
    def __init__(self):
        super(selfattLayer2, self).__init__()
        self.scale = tf.Variable(0., trainable=True)

    def call(self, x):
        # (b,h,w,c)
        shape = x.shape

        proj_query = tf.reshape(x, (shape[0], shape[3], shape[1] * shape[2]))  # (*,c,hw)
        proj_key = tf.reshape(x, (shape[0], shape[3], shape[1] * shape[2]))  # (*,c,hw)
        energy = tf.matmul(proj_query, proj_key, transpose_b=True)  # (*,c,c)
        energy_new = tf.broadcast_to(tf.math.reduce_max(energy, axis=-1, keepdims=True)[0], energy.shape) - energy  # (*,c,c)
        attention = tf.nn.softmax(energy_new, axis=-1)  # (*,c,c)
        proj_value = tf.reshape(x, (shape[0], shape[1] * shape[2], shape[3]))  # (*,hw,c)
        o = tf.matmul(proj_value, attention)
        y = self.scale * tf.reshape(o, x.shape) + x

        return y


class CNNModel(Model):
    def __init__(self,env_name,test=False):
        super(CNNModel, self).__init__()
        if not test:
            logdir = "./logs/scalars/" + env_name
            self.weight_dir = "./logs/weight/" + env_name + '/'
            file_writer = tf.summary.create_file_writer(logdir + "/metrics")
            file_writer.set_as_default()
        self.c1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                         activation='relu')
        self.att11 = selfattLayer(32)
        self.att12 = selfattLayer2()
        self.c2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
        self.att21 = selfattLayer(64)
        self.att22 = selfattLayer2()
        self.c3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
        self.att3 = selfattLayer(64)
        self.att4 = selfattLayer2()
        self.flatten = Flatten()
        self.d1 = Dense(512, activation="relu")
        self.d2 = Dense(1)  # C
        self.d3 = Dense(a_num, activation='softmax')  # A
        self.total_index = index
        self.call(np.random.random((batch_size, IMG_H, IMG_W, k)).astype(np.float32))
        if restore:
            self.load_weights(restore_weight_dir)

    # @tf.function
    def call(self, inputs):
        x = inputs / 255.0   #(1,84,84,4)
        x = self.c1(x)       #(1,20,20,32)
        t1 = self.att11(x)
        t2 = self.att12(x)
        x = t1 + t2
        x = self.c2(x)
        t1 = self.att21(x)   #(1,9,9,64)
        t2 = self.att22(x)
        x = t1 + t2
        x = self.c3(x)
        t1 = self.att3(x)    #(1,7,7,64)
        t2 = self.att4(x)
        x = t1 + t2
        self.show_att = x           # Comment this line when train
        x = self.flatten(x)
        x = self.d1(x)
        a = self.d3(x)
        v = self.d2(x)
        return a, v

    def loss(self, inputs, action_index, adv, targets, ep_old_ap):
        res = self.call(inputs)
        error = res[1][:, 0] - targets
        L = tf.reduce_sum(tf.square(error))

        adv = tf.dtypes.cast(tf.stop_gradient(adv), tf.float32)
        batch_size = inputs.shape[0]
        all_act_prob = res[0]
        selected_prob = tf.reduce_sum(action_index * all_act_prob, axis=1)
        old_prob = tf.reduce_sum(action_index * ep_old_ap, axis=1)

        r = selected_prob / (old_prob + 1e-6)

        H = -tf.reduce_sum(all_act_prob * tf.math.log(all_act_prob + 1e-6))

        Lclip = tf.reduce_sum(
            tf.minimum(
                tf.multiply(r, adv),
                tf.multiply(
                    tf.clip_by_value(
                        r,
                        1 - clip_epsilon,
                        1 + clip_epsilon
                    ),
                    adv
                )
            )
        )

        return -(Lclip - VFcoeff * L + beta * H) / batch_size, Lclip, H, L

    def total_grad(self, ep_stack_obs, ep_as, adv, realv, ep_old_ap):
        with tf.GradientTape() as tape:
            loss_value, Lclip, H, L = self.loss(ep_stack_obs, ep_as, adv, realv, ep_old_ap)
            self.total_index += 1

            if self.total_index % recode_span == 1:
                self.record('total loss', loss_value * len(ep_as))
                self.record('Lclip', Lclip)
                self.record('H', H)
                self.record('c loss', L)

            if self.total_index % save_span == 0:
                self.save_weights(self.weight_dir + str(self.total_index), save_format='tf')
        return tape.gradient(loss_value, self.trainable_weights), loss_value

    def total_grad2(self, ep_stack_obs, ep_as, adv, realv, ep_old_ap):
        with tf.GradientTape() as tape:
            loss_value, Lclip, H, L = self.loss(ep_stack_obs, ep_as, adv, realv, ep_old_ap)

        return tape.gradient(loss_value,self.show_att), loss_value

    def record(self, name, data, step=None):
        if not step:
            step = self.total_index
        tf.summary.scalar(name, data=data, step=step)


class RNNModel(Model):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.c1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                         activation='relu')
        self.c2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
        self.c3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation="relu")
        self.d2 = Dense(1)  # C
        self.d3 = Dense(a_num, activation='softmax')  # A
        self.lstm_cell = LSTMCell(hidden_unit_num)
        self.total_index = index
        self.call(
            np.random.random((batch_size, IMG_H, IMG_W, k)).astype(np.float32),
            tf.convert_to_tensor(np.zeros((batch_size, hidden_unit_num), dtype=np.float32)),
            tf.convert_to_tensor(np.zeros((batch_size, hidden_unit_num), dtype=np.float32))
        )
        if restore:
            self.load_weights(restore_weight_dir)

    @tf.function
    def call(self, inputs, h, c):
        x = inputs / 255.0
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x, hc = self.lstm_cell(inputs=x, states=(h, c))
        a = self.d3(x)
        v = self.d2(x)
        return a, v, hc

    def loss(self, inputs, action_index, adv, targets, ep_old_ap, h, c):
        res = self.call(inputs, h, c)
        error = res[1][:, 0] - targets
        L = tf.reduce_sum(tf.square(error))

        adv = tf.dtypes.cast(tf.stop_gradient(adv), tf.float32)
        batch_size = inputs.shape[0]
        all_act_prob = res[0]
        selected_prob = tf.reduce_sum(action_index * all_act_prob, axis=1)
        old_prob = tf.reduce_sum(action_index * ep_old_ap, axis=1)

        r = selected_prob / (old_prob + 1e-6)

        H = -tf.reduce_sum(all_act_prob * tf.math.log(all_act_prob + 1e-6))

        Lclip = tf.reduce_sum(
            tf.minimum(
                tf.multiply(r, adv),
                tf.multiply(
                    tf.clip_by_value(
                        r,
                        1 - clip_epsilon,
                        1 + clip_epsilon
                    ),
                    adv
                )
            )
        )

        return -(Lclip - VFcoeff * L + beta * H) / batch_size, Lclip, H, L

    def total_grad(self, ep_stack_obs, ep_as, adv, realv, ep_old_ap, total_h, total_c):
        with tf.GradientTape() as tape:
            loss_value, Lclip, H, L = self.loss(ep_stack_obs, ep_as, adv, realv, ep_old_ap, total_h, total_c)
            self.total_index += 1

            if self.total_index % recode_span == 1:
                self.record('total loss', loss_value * len(ep_as))
                self.record('Lclip', Lclip)
                self.record('H', H)
                self.record('c loss', L)

            if self.total_index % save_span == 0:
                self.save_weights(self.weight_dir + str(self.total_index), save_format='tf')
        return tape.gradient(loss_value, self.trainable_weights), loss_value

    def record(self, name, data, step=None):
        if not step:
            step = self.total_index
        tf.summary.scalar(name, data=data, step=step)


def test1():
    m = CNNModel()
    m.call(np.random.random((batch_size, IMG_H, IMG_W, k)).astype(np.float32))
    m.summary()

    inputs = np.random.random((1, IMG_H, IMG_W, k)).astype(np.float32)
    a, c = m(inputs)

    inputs = np.random.random((1, IMG_H, IMG_W, k)).astype(np.float32)
    s = time.time()
    a, c = m(inputs)
    print(time.time() - s)

    inputs = np.random.random((32, IMG_H, IMG_W, k)).astype(np.float32)
    s = time.time()
    a, c = m(inputs)
    print(time.time() - s)


if __name__ == '__main__':
    test1()
