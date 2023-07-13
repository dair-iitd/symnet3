import tensorflow as tf
# from pandas.conftest import axis
# from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Dropout

from spektral.layers import ops
from spektral.layers.convolutional.gcn import GraphConv
from spektral.layers.ops import modes

import math


class LocalNbrPool(tf.keras.layers.Layer):
    def __init__(self):
        super(LocalNbrPool, self).__init__()

    def call(self, inputs):
        """
        All Adj will be used as it is and expected to be modified in the calling function. Code is ported from Spektral's GAT.
        :param inputs: [X, A] with shapes= [batch, N, F] and [batch, N, N]
        :return: output of shape [batch, N, F']
        """
        X = inputs[0]
        A = inputs[1]

        num_nodes = A.shape[-1]
        A = tf.cast(tf.where(A==0, -1e10, 0.0), "float64")
        A = tf.expand_dims(A, axis=-1)
        X = [tf.reduce_max(X + A[:, :, i], axis=-2) for i in range(num_nodes)]
        # X = [tf.reduce_max(tf.multiply(X, A[:, :, i]), axis=-2) for i in range(num_nodes)]
        X = tf.stack(X, axis=-2)

        # X = [tf.reduce_max(tf.multiply(X, A[:, :, i]), axis=-2) for i in range(num_nodes)]
        # X = tf.stack(X, axis=-2)

        return X