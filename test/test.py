import tensorflow as tf
from tensorflow.python.keras.layers import *
import numpy as np

# b = CuDNNLSTM(units=10)

# lstm_cell = tf.keras.layers.LSTMCell(10)
#
# h = tf.convert_to_tensor(np.random.random((12,10)).astype(np.float32))
# c = tf.convert_to_tensor(np.random.random((12,10)).astype(np.float32))
# inputs = np.random.random((12,10)).astype(np.float32)
#
# output,state = lstm_cell(inputs=inputs,states=[h,c])

# use_RNN = tf.keras.layers.SimpleRNN(30)



# print(1)



a = 1
def f():
    global a
    a = 2

f()
print(a)