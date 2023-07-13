import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Dropout

from spektral.layers.convolutional.gcn import GraphConv
import numpy as np

class GraphAttentionDistance(GraphConv):
    # This is the Influence Layer defined in the paper. 
    def __init__(self,
                 channels,
                 attn_heads=1,
                 concat_heads=True,
                 dropout_rate=0.5,
                 return_attn_coef=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 num_edge_types=None,
                 **kwargs):
        super().__init__(channels,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.num_edge_types = num_edge_types

        if concat_heads:
            # Output will have shape (..., attention_heads * channels)
            self.output_dim = self.channels * self.attn_heads
        else:
            # Output will have shape (..., channels)
            self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_dim, self.attn_heads, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.attn_kernel_self = self.add_weight(
            name='attn_kernel_self',
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        self.all_pair_kernel = self.add_weight(
            name='all_pair_kernel',
            shape=[2*self.channels+1, self.attn_heads, self.channels],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        self.attn_kernel_neighs = self.add_weight(
            name='attn_kernel_neigh',
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        if self.num_edge_types:
            self.attn_kernel_edges = self.add_weight(
                name='attn_kernel_edges',
                shape=[self.attn_heads, self.num_edge_types],
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias'
            )

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]
        distance_mat = inputs[2]
        distance_mask = inputs[3]
        add_self_loops = inputs[4]
        remove_attn = inputs[5]
        training = inputs[6]

        output, attn_coef = self._call_dense(X, A, distance_mat, distance_mask, add_self_loops, remove_attn, training=training)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_dense(self, X, A, distance_mat, distance_mask, add_self_loops=True, remove_attn=False, training=True):
        # Combination of all nodes
        Xl_Xr_concat = tf.concat([tf.tile(tf.expand_dims(X, 1), [1, tf.shape(X)[1], 1, 1]), tf.tile(tf.expand_dims(X, 2), [1, 1, tf.shape(X)[1], 1])], axis=3)
        Xl_Xr_dist_concat = tf.concat([Xl_Xr_concat, distance_mat], axis=3)
        
        X = tf.einsum("BNI , IHO -> BNHO", X, self.kernel)
        attn_for_self = tf.einsum("BNHI , IHO -> BNHO", X, self.attn_kernel_self)
        attn_for_neighs = tf.einsum("BNHI , IHO -> BNHO", X, self.attn_kernel_neighs)
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)
        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.einsum("BNHM -> BNMH", attn_coef)
        attn_coef_dist = tf.einsum("BNMI , HI -> BNMH", distance_mat, self.attn_kernel_edges)
        attn_coef = attn_coef + attn_coef_dist
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

        # Attention only b/w fluent nodes
        mask = -10e9 * (1 - distance_mask)[:,:,:,None]
        attn_coef += mask

        attn_coef = tf.nn.softmax(attn_coef, axis=2)
        attn_coef_drop = self.dropout(attn_coef, training=training)
        
        # Aggregation over all nodes
        output = tf.einsum("BNMH , BNMD -> BNHD", attn_coef_drop, distance_mat)
        output = tf.reshape(output, [output.shape[0], output.shape[1], output.shape[2]*output.shape[3]])
        
        return output, attn_coef_drop

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][:-1] + (self.output_dim,)
        return output_shape

    def get_config(self):
        config = {
            'attn_heads': self.attn_heads,
            'concat_heads': self.concat_heads,
            'dropout_rate': self.dropout_rate,
            'attn_kernel_initializer': initializers.serialize(self.attn_kernel_initializer),
            'attn_kernel_regularizer': regularizers.serialize(self.attn_kernel_regularizer),
            'attn_kernel_constraint': constraints.serialize(self.attn_kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def preprocess(A):
        return A
