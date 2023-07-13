from builtins import dict

import tensorflow as tf
from tensorflow.keras import backend as K, initializers, regularizers, constraints
from spektral.layers import GraphAttention, TopKPool, MinCutPool, GraphAttentionDistance
from spektral.layers.convolutional.gnn_cnn_style import GNNCNNStyle

# Todo try kernel regularizer, dropout, vary num_attn_heads, attn_kernel_regularizer (SEE ORIGINAL CODE FOR )
# https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gat.py

# Graph Unet reference original author code in pytorch:
# https://github.com/HongyangGao/Graph-U-Nets/blob/master/ops.py

class GATConvLayer(tf.keras.layers.Layer):
    def __init__(self, channels, filter_size, attn_heads, activation, dropout_rate=0, initializer=None, conv_type="GAT", num_edge_types=None, use_shared_gat=False):
        super(GATConvLayer, self).__init__()

        self.channels = channels
        self.filter_size = filter_size
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.conv_type = conv_type
        self.num_edge_types = num_edge_types
        self.use_shared_gat = use_shared_gat

        if initializer is None:
            self.initializer = tf.keras.initializers.GlorotUniform()
            # self.initializer = tf.ones_initializer()
        else:
            self.initializer = initializer

        self.gat_layers = []
        if self.filter_size>1 and self.use_shared_gat:
            first_gat = GraphAttention(channels=self.channels, attn_heads=self.attn_heads, concat_heads=True, dropout_rate=self.dropout_rate, activation=None, use_bias=False, kernel_initializer=self.initializer, num_edge_types=self.num_edge_types)
            second_gat = GraphAttention(channels=self.channels, attn_heads=self.attn_heads, concat_heads=True, dropout_rate=self.dropout_rate, activation=None, use_bias=False, kernel_initializer=self.initializer, num_edge_types=self.num_edge_types)

            self.gat_layers.append(first_gat)
            for i in range(filter_size - 2):
                self.gat_layers.append(second_gat)
            # Final hidden to output layer, notice params
            self.gat_layers.append(GraphAttention(channels=self.channels, attn_heads=self.attn_heads, concat_heads=False, dropout_rate=self.dropout_rate, activation=self.activation, use_bias=False , kernel_initializer=self.initializer, num_edge_types=self.num_edge_types))
        else:
            for i in range(filter_size - 1):
                self.gat_layers.append(GraphAttention(channels=self.channels, attn_heads=self.attn_heads, concat_heads=True, dropout_rate=self.dropout_rate, activation=None, use_bias=False, kernel_initializer=self.initializer, num_edge_types=self.num_edge_types))
            # Final hidden to output layer, notice params
            self.gat_layers.append(GraphAttention(channels=self.channels, attn_heads=self.attn_heads, concat_heads=False, dropout_rate=self.dropout_rate, activation=self.activation, use_bias=False , kernel_initializer=self.initializer, num_edge_types=self.num_edge_types))

    def call(self, X, A, use_self_loops_in_all_adj=True, remove_attn=False, training=True):
        if self.num_edge_types:
            A = tf.transpose(A, (1, 2, 3, 0))
        for layer in self.gat_layers:
            X = layer([X, A, use_self_loops_in_all_adj, remove_attn, training])
        return X

class GATConvLayerDistance(tf.keras.layers.Layer):
    def __init__(self, channels, filter_size, attn_heads, activation, dropout_rate=0, initializer=None, conv_type="GAT", num_edge_types=None, use_shared_gat=False, concat_last_gat=True, return_attn_coef=False):
        super(GATConvLayerDistance, self).__init__()

        self.channels = channels
        self.filter_size = filter_size
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.conv_type = conv_type
        self.num_edge_types = num_edge_types
        self.use_shared_gat = use_shared_gat
        self.concat_last_gat = concat_last_gat

        if initializer is None:
            self.initializer = tf.keras.initializers.GlorotUniform()
            # self.initializer = tf.ones_initializer()
        else:
            self.initializer = initializer
        self.return_attn_coef = return_attn_coef
        self.gat_layers = []
        if self.filter_size>1 and self.use_shared_gat:
            first_gat = GraphAttentionDistance(channels=self.channels, attn_heads=self.attn_heads, concat_heads=True, dropout_rate=self.dropout_rate, activation=None, use_bias=False, kernel_initializer=self.initializer, num_edge_types=self.num_edge_types)
            second_gat = GraphAttentionDistance(channels=self.channels, attn_heads=self.attn_heads, concat_heads=True, dropout_rate=self.dropout_rate, activation=None, use_bias=False, kernel_initializer=self.initializer, num_edge_types=self.num_edge_types)

            self.gat_layers.append(first_gat)
            for i in range(filter_size - 2):
                self.gat_layers.append(second_gat)
            # Final hidden to output layer, notice params
            self.gat_layers.append(GraphAttentionDistance(channels=self.channels, attn_heads=self.attn_heads, concat_heads=self.concat_last_gat, dropout_rate=self.dropout_rate, activation=self.activation, use_bias=False , kernel_initializer=self.initializer, num_edge_types=self.num_edge_types))
        else:
            for i in range(filter_size - 1):
                self.gat_layers.append(GraphAttentionDistance(channels=self.channels, attn_heads=self.attn_heads, concat_heads=True, dropout_rate=self.dropout_rate, activation=None, use_bias=False, kernel_initializer=self.initializer, num_edge_types=self.num_edge_types))
            # Final hidden to output layer, notice params
            self.gat_layers.append(GraphAttentionDistance(channels=self.channels, attn_heads=self.attn_heads, concat_heads=self.concat_last_gat, dropout_rate=self.dropout_rate, activation=self.activation, use_bias=False , kernel_initializer=self.initializer, num_edge_types=self.num_edge_types, return_attn_coef=self.return_attn_coef))

    def call(self, X, A, distance_mat, use_self_loops_in_all_adj=True, remove_attn=False, training=True, beta=1):
        if self.num_edge_types:
            A = tf.transpose(A, (1, 2, 3, 0))
            distance_mat = tf.transpose(distance_mat, (1, 2, 3, 0))
        for layer in self.gat_layers:
            X = layer([X, A, distance_mat, use_self_loops_in_all_adj, remove_attn, training, beta])
        return X


class GraphPool(tf.keras.layers.Layer):
    def __init__(self, k_value, sigmoid_gating=False, kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,):
        super(GraphPool, self).__init__()
        self.k_value = k_value
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        self.F = input_shape[0][-1]
        self.N = input_shape[0][0]
        self.kernel = self.add_weight(shape=(self.F, 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        super().build(input_shape)

    def compute_scores(self, X):
        # a = K.l2_normalize(self.kernel)
        # b = K.dot(X, a)
        # return b
        return K.dot(X, K.l2_normalize(self.kernel))

    def call(self, inputs):
        print("Droupout training flag not implemented. Implement it similar to GraphAttentionLayer")
        exit(-1)
        X, A, training = inputs

        # 1. Get score
        y = self.compute_scores(X)
        y = tf.squeeze(y, axis=[-1])

        # 2. Get top k
        values, idx = tf.math.top_k(y, int(self.k_value))

        # Multiply X and y to make layer differentiable
        y = self.gating_op(y)
        y = tf.tile(tf.expand_dims(y, axis=-1), [1, 1, self.F])
        X = X * y

        # 3. Get features of pooled nodes only
        idx = tf.expand_dims(idx, -1)
        new_X = tf.gather_nd(X, idx, batch_dims=1)

        # 4. Create new Adj Matrix
        new_A = GraphPool.reduce_adj_mat(A, idx)
        return new_X, new_A, idx

    @staticmethod
    def reduce_adj_mat(A, idx):
        """
        Fetches the specific rows and cols of adj mat
        :param A: [batch, rows, col]
        :param idx: [batch, rows_to_keep, 1]
        :return: [batch, len(idx), len(idx)]
        """
        # Fetch rows
        A = tf.gather_nd(A, idx, batch_dims=1)
        # Fetch columns
        A = tf.transpose(tf.gather_nd(tf.transpose(A, perm=[0, 2, 1]), idx, batch_dims=1), perm=[0, 2, 1])
        return A

# Pool class using spektal code
# class GraphPool(tf.keras.layers.Layer):
#     def __init__(self, k_ratio, sigmoid_gating=False, pool_type="topk"):
#         super(GraphPool, self).__init__()
#         self.k_ratio = k_ratio
#         self.activation = activation
#         self.sigmoid_gating = sigmoid_gating
#         self.pool_type = pool_type
#         self.pool_layer = None
#         if self.pool_type == "topk":
#             self.pool_layer = TopKPool(self.k_ratio, sigmoid_gating=self.sigmoid_gating, return_mask=True)
#         else:
#             print("unet.GraphPool: Incorrect params pool_type = " + self.pool_type)
#             exit(-1)
#
#     def call(self, X, A):
#         if self.pool_type == "topk":
#             return self.pool_layer([X, A])


# Todo Check if it is trainable or even required to be trainable
class GraphUnpool(tf.keras.layers.Layer):
    def __init__(self):
        super(GraphUnpool, self).__init__()

    def convert_indices(self, idx, output_graph_size):
        """
        Shape will be (batch_size,perm,1) (Want shape [batch_size*perm,1])
        """
        batch_size = idx.shape[0]
        idx = tf.transpose(tf.reshape(idx, [idx.shape[0], idx.shape[1]]))
        idx = tf.transpose(idx + tf.range(batch_size) * output_graph_size)
        return tf.reshape(idx, [-1, 1])

    def call(self, X, A, idx):
        """
        Up samples layer
        :param A: Adjacency Matrix; Should be of new shape
        :param X: Input to upsample. X is a variable.
        :param idx: idx has to be of same shape as of new_X (i.e. [A.shape[0], X.shape[1]])
        :return:
        """
        batch_size = X.shape[0]
        num_nodes_old = X.shape[1]
        feature_dims = X.shape[2]
        num_nodes_new = A.shape[1]

        X_flatten = tf.reshape(X, [batch_size * num_nodes_old, feature_dims])
        converted_indices = self.convert_indices(idx, num_nodes_new)
        new_X = tf.reshape(tf.scatter_nd(converted_indices, X_flatten, shape=[batch_size*num_nodes_new, feature_dims]), [batch_size, num_nodes_new, feature_dims])
        return new_X, A

class TransitionModelDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super(TransitionModelDecoder, self).__init__()

        self.up_gats = []
        self.unpools = []

    def add_upsample_layer(self, channels, filter_size, attn_heads, activation, dropout_rate):
        self.up_gats.append(GATConvLayer(channels, filter_size, attn_heads, activation, dropout_rate))
        self.unpools.append(GraphUnpool())

    def add_final_gat(self, out_dim, out_filters, attn_heads, activation, dropout_rate):
        self.end_gat = GATConvLayer(out_dim, out_filters, attn_heads, activation, dropout_rate)

    def call(self, X, orig_X, l_n, indices_list, adj_ms, down_outs, action):
        # Up Sample
        for i in range(l_n):
            up_i = l_n - i - 1
            idx, A = indices_list[up_i], adj_ms[up_i]
            X, A = self.unpools[i](X, A, idx)
            X = self.up_gats[i](X, A)
            # Skip connections
            X = tf.add(X, down_outs[up_i])

        # End Layer
        X = tf.concat([X, orig_X], -1)
        X = self.end_gat(X, A)
        return X

class GraphUnet(tf.keras.layers.Layer):
    def __init__(self, channel_l, filter_size_l, attn_heads_l, k_l, is_k_ratio, global_embed_units,
                 out_dim, out_filters, expand_nbrhood, add_self_loops, dropout_rate, activation, add_tm_decoder, add_skip=True,
                 pool_type="top_k"):
        """
            :param channel_l: list of channels for each layer
            :param filter_size_l: list of filter size for each layer
            :param attn_heads_l: list of num of attn heads for each layer
            :param k_l: list of number of ks to be picked in each downsampling layer. [7, 3]
            :param is_k_ratio: bool; Whether k_list is a list of ratios or exact k values
            :param global_embed_units: list of dense layer units to get global embedding; Last element represents global_dim
            :param out_dim: final output feature dimension
            :param out_filters: num of filters to be used on final output GAT
            :param expand_nbrhood: int; how much to expand nbrhood in each iteration of down sample
            :param add_self_loops: int; Add self loops after pooling
            :param dropout_rate: dropout value
            :param activation: String; activation; to be used for all layers
            :param add_tm_decoder: bool; whether to add a transition model decoder or not
            :param add_skip: list of bools; whether to add skip connections or not at a particular level. [True True, False] means at level [start_gat, 7, NOT 3].
            :param pool_type: string of pooling type; top_k or mincut
        """

        super(GraphUnet, self).__init__()

        # start gat + unet down + up samples
        try:
            assert len(k_l)%2 == 0 and len(channel_l)%2 == 1 and len(filter_size_l)%2 == 1 and len(attn_heads_l)%2 == 1
            assert len(add_skip) == len(k_l) + 1
        except AssertionError as error:
            print("unet: num of channel, attn_heads, filter_sizes lists should be even and num of elements in k_l shpuld be odd")
            exit(-1)

        self.channel_l = channel_l
        self.filter_size_l =  filter_size_l
        self.attn_heads_l = attn_heads_l
        self.k_list = k_l
        self.is_k_ratio = is_k_ratio
        self.global_embed_units = global_embed_units
        self.out_dim = out_dim
        self.out_filters = out_filters
        self.add_self_loops = add_self_loops
        self.expand_nbrhood = expand_nbrhood
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.add_tm_decoder = add_tm_decoder
        self.add_skip = add_skip
        self.pool_type = pool_type

        self.down_gats = []
        self.up_gats = []
        self.pools = []
        self.unpools = []
        self.l_n = len(k_l)

        if self.add_tm_decoder:
            self.tm_decoder = TransitionModelDecoder()

        self.start_gat = GATConvLayer(self.channel_l[0], self.filter_size_l[0], self.attn_heads_l[0], activation=self.activation, dropout_rate=self.dropout_rate)
        # GAT to convert input to hidden dim
        for i in range(1, len(channel_l)):
            self.down_gats.append(GATConvLayer(self.channel_l[i], self.filter_size_l[i], self.attn_heads_l[i], activation=self.activation, dropout_rate=self.dropout_rate))
            if self.pool_type == "top_k":
                self.pools.append(GraphPool(k_l[i - 1]))
            elif self.pool_type == "soft_mincut":
                self.pools.append(MinCutPool(k_l[i - 1], activation=self.activation))
            # self.pools.append(GraphPool(k_l[i-1]))

            self.up_gats.append(GATConvLayer(self.channel_l[i], self.filter_size_l[i], self.attn_heads_l[i], activation=self.activation, dropout_rate=self.dropout_rate))
            self.unpools.append(GraphUnpool())

            if self.add_tm_decoder:
                self.tm_decoder.add_upsample_layer(self.channel_l[i], self.filter_size_l[i], self.attn_heads_l[i], activation=self.activation, dropout_rate=self.dropout_rate)

        self.end_gat = GATConvLayer(self.out_dim, self.out_filters, self.attn_heads_l[i], activation=self.activation, dropout_rate=self.dropout_rate)
        if self.add_tm_decoder:
            self.tm_decoder.add_final_gat(self.out_dim, self.out_filters, self.attn_heads_l[i], activation=self.activation, dropout_rate=self.dropout_rate)

        # using fully connected layer to get global embedding (ge)
        if self.global_embed_units is not None:
            self.ge_layers = []
            for i in range(len(self.global_embed_units)):
                self.ge_layers.append(tf.keras.layers.Dense(units=self.global_embed_units[i], activation=self.activation))

    def expand_neighborhood(self, A, nbrhood=1, return_bin=True, add_self_loops=True):
        """
        :param A: adj mat [batch, node, node]
        :param nbrhood: number of hops to expand
        :param return_bin: return a binary matrix
        :param add_self_loops: bool; whether to add self loops or not
        """
        if add_self_loops:
            res = tf.eye(A.shape[1], dtype=A.dtype, batch_shape=[A.shape[0]])
        else:
            res = tf.zeros(A.shape)
        res += A
        res = tf.clip_by_value(res, 0, 1)
        for i in range(nbrhood):
            res = res + tf.linalg.matmul(res, res)

        if return_bin:
            return tf.clip_by_value(res, 0, 1)
        return res

    def call(self, inputs, A, pred_next_state):
        """
        :param inputs: [X, graph_features], X=[batch, num_nodes, F1], gf=[batch, F2]
        :param A: [batch, num_nodes, num_nodes]
        :param pred_next_state: bool; Predict the next state or not
        :return:
        """
        X, graph_features, action = inputs

        batch_size = X.shape[0]
        adj_ms = []
        indices_list = []
        down_outs = []

        # 1 Start GAT
        X = self.start_gat(X, A)
        orig_X = X

        # 2 Down Sample
        for i in range(self.l_n):
            # Pass through GAT
            X = self.down_gats[i](X, A)
            adj_ms.append(A)
            down_outs.append(X)
            if self.expand_nbrhood:
                A = self.expand_neighborhood(A, self.expand_nbrhood, self.add_self_loops)

            # Down Sample
            X, A, idx = self.pools[i]([X, A])
            indices_list.append(idx)

        X_middle = X
        # 3 Global Embedding
        global_embed = None
        if self.global_embed_units is not None:
            # Flatten X
            global_embed = tf.reshape(X, [batch_size, -1])
            # We can concat all nodes to get input for graph FCNN coz are
            # fixing number of nodes in previous layer rather than using ratio
            global_embed = tf.concat([global_embed, graph_features], axis=-1)
            # pass through FCNN
            for layer in self.ge_layers:
                global_embed = layer(global_embed)

        # 4 Up Sample
        for i in range(self.l_n):
            up_i = self.l_n - i - 1
            idx, A = indices_list[up_i], adj_ms[up_i]
            X, A = self.unpools[up_i](X, A, idx)
            X = self.up_gats[up_i](X, A)

            # Skip connections
            if self.add_skip[i+1]:  # i+1 coz 1st one is for start GAT
                X = tf.add(X, down_outs[up_i])

        # 5 End Layer
        if self.add_skip[0]:
            X = tf.concat([X, orig_X], -1)
        X = self.end_gat(X, A)

        # 6 Get next state from transition decoder
        next_state = None
        if self.add_tm_decoder and pred_next_state:
            next_state = self.tm_decoder(X_middle, orig_X, self.l_n, indices_list, adj_ms, down_outs, action)
        return X, global_embed, next_state

    @staticmethod
    def contains_isolated_node(A, idx=None):
        """
        Batchwise checks if there is an isolated node in graph
        :param A: [batch_size, num_nodes, num_nodes]
        :param idx: [batch_size, num_nodes, 1]
        :return: [batch_size] boolean
        """
        if idx is not None:
            # Fetch only part of adj mat at idx
            A = GraphPool.reduce_adj_mat(A, idx)
        A = A + tf.transpose(A, perm=[0, 2, 1])
        A = tf.reduce_sum(A, axis=-1)
        return tf.math.reduce_any(A==0, axis=-1)

    @staticmethod
    def convert_k_to_ratio(k_l, N):
        k_list = k_l[:]
        k_list.insert(0, N)
        res = []
        for i in range(1, len(k_list)):
            res.append(k_list[i]/k_list[i-1])
        return res

    def get_encoder_trainable_variables(self):
        variables = self.start_gat.trainable_variables
        for g in self.down_gats:
            variables += g.trainable_variables
        for p in self.pools:
            variables += p.trainable_variables
        return variables

    def get_decoder_trainable_variables(self):
        variables = []
        for g in self.up_gats:
            variables += g.trainable_variables
        variables += self.end_gat
        return variables

    def get_ge_trainable_variables(self):
        variables = []
        for l in self.ge_layers:
            variables += l.trainable_variables
        return variables


if __name__ == "__main__":
    channel_l = [6, 6, 6]
    filter_size_l = [2, 2, 2]
    attn_heads_l = [4, 4, 4]
    k_l = [3, 2]
    is_k_ratio = False
    global_embed_units = [5, 6]
    out_dim =5
    out_filters = 2
    expand_nbrhood = 1
    add_self_loops = True
    dropout_rate = 0
    activation = tf.nn.relu
    add_tm_decoder = False

    batch_size = 2

    X = tf.Variable(tf.random.uniform([batch_size, 5, 20]))
    gf = tf.Variable(tf.random.uniform([batch_size, 10]))
    A = tf.random.uniform([batch_size, 5, 5], maxval=2, dtype="int32")
    A = tf.cast(A, dtype="float64")

    # if not is_k_ratio:
    #     k_l = GraphUnet.convert_k_to_ratio(k_l, A.shape[0])

    u = GraphUnet(channel_l, filter_size_l, attn_heads_l, k_l, is_k_ratio, global_embed_units, out_dim, out_filters,
                  expand_nbrhood, add_self_loops, dropout_rate, activation, add_tm_decoder)

    print(X)
    if add_tm_decoder:
        X = u([X, gf, 1], A)
    else:
        X = u([X, gf], A)
    print(X)
    print("Test 3")

    # gp = GraphPool1(k_l[0], True)
    # gp([X, A])
    # print("Done")