import tensorflow as tf
from symnet3.unet import GATConvLayer, GATConvLayerDistance
from symnet3.action_decoder import ActionDecoder
import numpy as np
import pdb
use_self_loops_in_all_adj = True
remove_attn = False

symnet_params = {"channels", "num_postprocess", "num_preprocess", "attn_heads", "dropout_rate", "activation", "conv_type", "num_edge_types", "use_shared_gat"}
gat_params = {"channels", "attn_heads", "dropout_rate", "activation", "conv_type", "num_edge_types", "use_shared_gat"}

class SymNet3(tf.keras.Model):
    def __init__(self, general_params, se_params, ad_params, ge_params, tm_params, env_instance_wrapper=None):

        super(SymNet3, self).__init__()

        self.general_params = general_params
        self.se_params = se_params
        self.ad_params = ad_params
        self.ge_params = ge_params
        self.tm_params = tm_params
        self.tfm_params = self.ge_params["tfm_params"]

        self.out_deg = self.general_params["add_out_deg"]
        self.in_deg = self.general_params["add_in_deg"]
        self.bet_cen = self.general_params["add_bet_cen"]
        self.dist_leaves = self.general_params["add_dist_leaves"]
        self.use_bidir_edges = self.general_params["use_bidir_edges"]
        self.make_grid = self.general_params["make_grid"]

        self.use_distance_mat = se_params["use_distance_mat"]
        self.se_type = self.se_params["type"]
        self.se_count = self.se_params["num_se"]
        self.se_params["num_edge_types"] = None
        self.use_edge_types = se_params["use_edge_types"]
        self.preprocess_gat = se_params["use_preprocess_layer"]

        if self.use_distance_mat:
            arg_dict = dict((k, self.se_params[k]) for k in gat_params)
            arg_dict["num_edge_types"] = 1 # len(env_instance_wrapper.envs[0].instance_parser.dbn_edge_types_to_idx)
            arg_dict["filter_size"] = 1
            arg_dict["concat_last_gat"] = False
            arg_dict["attn_heads"] = se_params["num_dist_attn_heads"]
            arg_dict["return_attn_coef"] = True
            self.gat_distance_mat = GATConvLayerDistance(**arg_dict)
            print("Built gat_distance_mat")

        if self.use_edge_types:
            self.se_params["num_edge_types"] = self.se_params["num_se"]
        
        if self.preprocess_gat:
            self.se_list_preprocess = self.get_state_encoder(self.se_params["num_preprocess"], env_instance_wrapper)
            self.se_list_postprocess = self.get_state_encoder(self.se_params["num_postprocess"], env_instance_wrapper)
        else:
            self.se_list_postprocess = self.get_state_encoder(self.se_params["num_postprocess"], env_instance_wrapper)
        
        
        self.final_node_embedder = tf.keras.layers.Dense(units=self.se_params["out_dim"], activation=self.se_params["activation"])

        self.ge_type = self.ge_params["type"]
        
        self.num_action_dim = self.se_params["out_dim"]
        self.action_decoders = self.get_action_decoder()
        self.value_decoders = self.get_action_decoder()


    def get_node_feature_dim(self, env_instance_wrapper):
        state, _ = env_instance_wrapper.envs[0].reset()
        adjacency_matrix, node_features, graph_features, action_details = self.get_parsed_state([state], 0, env_instance_wrapper)
        return node_features.shape[-1]

    def get_ckpt_parts(self):
        ckpt_parts = {}
        ckpt_parts["se_list"] = self.se_list_postprocess
        ckpt_parts["final_node_embedder"] = self.final_node_embedder
        ckpt_parts["next_state_projection"] = self.next_state_projection
        ckpt_parts["reward_projection"] = self.reward_projection
        ckpt_parts["action_decoders"] = self.action_decoders
        return ckpt_parts

    def init_network(self, env_wrapper, instance):
        initial_state, _ = env_wrapper.envs[instance].reset()  # Initial state
        self.policy_prediction(states=[initial_state], instance=instance, env_wrapper=env_wrapper)

    def get_state_encoder(self, num_se, env_instance_wrapper):
        se_list = []
        if self.use_edge_types:
            args = dict((k, self.se_params[k]) for k in gat_params)
            args['filter_size'] = num_se
            print(f"Building a GAT with depth {num_se}")
            se_list.append(GATConvLayer(**args))
        else:
            for _ in range(num_se):
                se_list.append(GATConvLayer(**dict((k, self.se_params[k]) for k in symnet_params)))
        return se_list

    def get_action_decoder(self):
        action_decoders = []
        for _ in range(self.ad_params["num_action_templates"]):
            action_decoders.append(ActionDecoder(self.ad_params))
        return action_decoders

    def get_parsed_state(self, states, instance, env_wrapper):
        adjacency_matrix, node_features, graph_features = env_wrapper.get_parsed_state(states, instance)
        action_details = env_wrapper.get_action_details(instance)

        if self.use_bidir_edges:
            adjacency_matrix = adjacency_matrix + tf.transpose(adjacency_matrix, perm=[0, 1, 3, 2])
            adjacency_matrix = tf.clip_by_value(adjacency_matrix, 0, 1)

        return adjacency_matrix, node_features, graph_features, action_details


    def policy_prediction_helper(self, batch_size, adjacency_matrix_full, env_wrapper, instance, graph_features, action_details, se_embed_l, sample, training, prune_actions):
        node_embed = tf.concat(se_embed_l, axis=-1)
        final_node_embedding = self.final_node_embedder(node_embed)
        global_embed = tf.reshape(tf.concat([tf.reduce_max(final_node_embedding, axis=1), graph_features], axis=1), [batch_size, -1])
        if self.ge_type == "deep_global_pool":
            A = tf.reduce_max(adjacency_matrix_full, 0)
            global_embed_pool = self.global_embedder_net(final_node_embedding, A)
            global_embed = tf.concat([global_embed, global_embed_pool], axis=-1)

        action_scores = [0 for i in range(len(action_details))]  # Score of each action
        action_affects = env_wrapper.envs[instance].instance_parser.action_affects
        remove_dbn = env_wrapper.envs[instance].instance_parser.remove_dbn
        for i in range(len(action_details)):
            action_template = action_details[i][0]
            input_nodes = list(action_details[i][1])
            arg_nodes = action_details[i][2]
            global_embed_temp = global_embed
            
            if len(arg_nodes) == 0:  # Unparametrized action
                action_scores[i] = self.action_decoders[action_template]([global_embed_temp, None, training])
            else:
                if len(input_nodes) > 0:
                    temp_embedding_list = [  # Select embeddings of nodes used
                        tf.reshape(final_node_embedding[:, inp, :], [batch_size, self.num_action_dim]) for inp in input_nodes]
                    node_state_embedding_concat = tf.concat(temp_embedding_list, axis=1)  # Concat embeddings of all affected nodes
                    node_state_embedding_reshape = tf.reshape(node_state_embedding_concat, [batch_size, len(input_nodes), self.num_action_dim])
                    node_state_embedding_pooled = tf.reshape(tf.reduce_max(node_state_embedding_reshape, axis=1), [batch_size, self.num_action_dim])  # Max Pool
                    arg_embedding_list = [tf.reshape(final_node_embedding[:, inp, :], [batch_size, self.num_action_dim]) for inp in arg_nodes]
                    node_state_embedding_pooled = tf.concat(arg_embedding_list + [node_state_embedding_pooled], axis=1)
                    action_scores[i] = self.action_decoders[action_template]([node_state_embedding_pooled, global_embed_temp, training])
                else:
                    if remove_dbn:
                        arg_embedding_list = [tf.reshape(final_node_embedding[:, inp, :], [batch_size, self.num_action_dim]) for inp in arg_nodes]
                        action_scores[i] = self.action_decoders[action_template]([tf.concat(arg_embedding_list, axis=1), global_embed_temp, training])
                    else:
                        gnd_action_affects = action_affects[action_template]
                        # Wildfire case; Treat as NOOP
                        if gnd_action_affects:
                            # IF wildfire
                            action_template = action_details[0][0]
                            action_scores[i] = self.action_decoders[action_template]([global_embed_temp, None, training])
                        else:
                            arg_embedding_list = [tf.reshape(final_node_embedding[:, inp, :], [batch_size, self.num_action_dim]) for inp in arg_nodes]
                            action_scores[i] = self.action_decoders[action_template]([tf.concat(arg_embedding_list + [tf.zeros([batch_size, self.num_action_dim], tf.float64)], axis= 1), global_embed_temp, training])
        action_scores = tf.concat(action_scores, axis=-1)

        if sample:
            logits = tf.nn.log_softmax(action_scores)
            if prune_actions:
                # Get the actions you want to keep
                masks = env_wrapper.get_prune_mask(states, instance)
                masks = logits.dtype.min * (1.0 - masks)
                logits += masks
            return tf.random.categorical(logits=logits, num_samples=batch_size, dtype=tf.int32)  # Return sampled actions
        else:
            if prune_actions:
                # Get the actions you want to keep
                masks = env_wrapper.get_prune_mask(states, instance)
                masks = -10e9 * (1.0 - masks)
                action_scores += masks
            probs = tf.nn.softmax(action_scores)  # Expected shape is (batch_size,num_actions)
            return probs

    def policy_prediction(self, states, instance, env_wrapper, sample=False, action=None, plot_graph=False, file_name=None, action_taken=None, training=True, prune_actions=False, return_attn_coef=False, return_node_emb=False):
        adjacency_matrix, node_features, graph_features, action_details = self.get_parsed_state(states, instance, env_wrapper)
        adjacency_matrix = np.transpose(adjacency_matrix, [0, 1, 3, 2])
        batch_size = node_features.shape[0]
        num_nodes = node_features.shape[1]
        if self.preprocess_gat:
            se_embed_l = []
            if self.use_edge_types:
                for i, se in enumerate(self.se_list_preprocess):
                    # res = se(node_features, adjacency_matrix[i], use_self_loops_in_all_adj, remove_attn) INITIAL ERROR
                    res = se(node_features, adjacency_matrix, use_self_loops_in_all_adj, remove_attn)
                    se_embed_l.append(res)
                node_features = tf.concat(se_embed_l, axis=-1)
            else:
                for i, se in enumerate(self.se_list_preprocess):
                    res = se(node_features, adjacency_matrix[i], use_self_loops_in_all_adj, remove_attn) # INITIAL ERROR
                    # res = se(node_features, adjacency_matrix, use_self_loops_in_all_adj, remove_attn)
                    se_embed_l.append(res)
                node_features = tf.concat(se_embed_l, axis=-1)
            
            # print(len(se_embed_l), "Preprocess:", node_features.shape)
            
        if self.use_distance_mat:
            d = np.max(adjacency_matrix.astype("int32"), 0)
            d = env_wrapper.get_distance_mat(d, instance)
            mask = env_wrapper.get_distance_mask(instance)[None,:] # A 2D mask 
            mask = tf.repeat(mask, d.shape[0], axis=0)
            d = np.transpose(d, [1, 0, 2, 3])
            adjacency_matrix_fc = np.ones_like(d)
            distance_features, dist_attn_coef = self.gat_distance_mat(node_features, adjacency_matrix_fc, d, mask, use_self_loops_in_all_adj, remove_attn, beta=1.0)
            node_features = tf.concat([node_features, distance_features], axis=-1)
        
        se_embed_l = []
        if self.use_edge_types:
            res = self.se_list_postprocess[0](node_features, adjacency_matrix, use_self_loops_in_all_adj, remove_attn)
            se_embed_l.append(res)
        else:
            for i, se in enumerate(self.se_list_postprocess):
                res = se(node_features, adjacency_matrix[i], use_self_loops_in_all_adj, remove_attn)
                se_embed_l.append(res)
        if return_attn_coef:
            return self.policy_prediction_helper(batch_size, adjacency_matrix, env_wrapper, instance, graph_features, action_details, se_embed_l, training=training, sample=sample, prune_actions=prune_actions), dist_attn_coef
        elif return_node_emb:
            return self.policy_prediction_helper(batch_size, adjacency_matrix, env_wrapper, instance, graph_features, action_details, se_embed_l, training=training, sample=sample, prune_actions=prune_actions), se_embed_l
        else:
            return self.policy_prediction_helper(batch_size, adjacency_matrix, env_wrapper, instance, graph_features, action_details, se_embed_l, training=training, sample=sample, prune_actions=prune_actions)
