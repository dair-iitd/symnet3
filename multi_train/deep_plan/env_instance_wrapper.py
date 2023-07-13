from helper import get_adj_mat_from_list
import numpy as np
import tensorflow as tf
import my_config
import networkx as nx
import re

class EnvInstanceWrapper:
	def __init__(self, envs):
		self.envs = envs
		self.fluent_feature_dims = None
		self.nonfluent_feature_dims = None
		self.adjacency_mat = [None] * len(self.envs)
		self.extended_adjacency_mat = [None] * len(self.envs)
		self.nf_features = [None] * len(self.envs)
		self.action_details = [None] * len(self.envs)
		self.extended_action_details = [None] * len(self.envs)
		self.num_action_nodes = [None] * len(self.envs)
		self.fill_env_meta_data()  # Fetch meta data and nf_features for all envs
		self.modify_adj()  # Expand nbrhood and add bias for all envs

		# For Structural Features
		self.struct_features_all = None
		self.num_nodes = [self.adjacency_mat[i][0].shape[0] for i in range(len(envs))]
		self.nx_graphs = []     # List of nx graphs of each env
		self.out_degree = []    # List of list of out_degrees of each node in each env.
		self.in_degree = []     # List of list of in_degrees of each node in each env.
		self.bet_cent = []      # List of list of between_degrees of each node in each env.
		self.dist_leaves = []   # List of list of mean_distance from leaves of each node in each env.
		self.cached_distances = {}
		self.cached_masks = {}

	def fill_env_meta_data(self):
		for i in range(len(self.envs)):
			self.fluent_feature_dims, self.nonfluent_feature_dims = self.envs[i].get_feature_dims()
			self.nf_features[i] = self.envs[i].get_nf_features()
			self.action_details[i] = self.envs[i].get_action_details()
			self.extended_action_details[i] = self.envs[i].get_extended_action_details()
			self.num_action_nodes[i] = self.envs[i].get_num_action_nodes()

	def modify_adj(self):
		for i in range(len(self.envs)):
			adjacency_list = self.envs[i].get_adjacency_list()
			self.adjacency_mat[i] = [get_adj_mat_from_list(aj) for aj in adjacency_list]
			if my_config.fc_adjacency:
				self.adjacency_mat[i].append(self.find_distance_from_adj(self.adjacency_mat[i]))

	def get_distance_mask(self, instance):
		if instance not in self.cached_masks:
			n = len(self.envs[instance].instance_parser.node_dict)
			m = len(self.envs[instance].instance_parser.fluent_nodes)
			mask = np.eye(n)
			mask += np.pad(np.ones((m, m)), ((0, n-m), (0, n-m)))
			mask = np.clip(mask, 0, 1)
			self.cached_masks[instance] = mask
		return self.cached_masks[instance]

	def get_distance_mat(self, adjacency_mat, instance, dbn_distance=True):
		batch_size = adjacency_mat.shape[0]
		if instance in self.cached_distances: # Cache during training/testing to avoid re-computing
			distance_mat = self.cached_distances[instance]
		else:
			# Compute shortest distance on the first adjacency
			graph = nx.from_dict_of_lists(self.envs[instance].instance_parser.adjacency_lists[0], create_using=nx.DiGraph)
			distance_dict = dict(nx.all_pairs_shortest_path_length(graph))
			num_nodes = len(self.envs[instance].instance_parser.node_dict)
			distance_mat = np.zeros((1, num_nodes, num_nodes))
			for i in range(distance_mat.shape[1]):
				for j in range(distance_mat.shape[2]):
					distance_mat[0][i][j] = distance_dict[i].get(j, -1)
			distance_mat = distance_mat/max(np.max(distance_mat), 1)
			self.cached_distances[instance] = distance_mat
		
		distance_mat = np.stack([distance_mat for _ in range(batch_size)])
		return distance_mat

	def get_processed_adj_mat(self, i, batch_size):
		return np.array([[self.adjacency_mat[i][j] for batch in range(batch_size)] for j in range(len(self.adjacency_mat[i]))]).astype(np.float32)

	def get_processed_extended_adj_mat(self,i,batch_size):
		return np.array([[self.extended_adjacency_mat[i][j] for batch in range(batch_size)] for j in range(len(self.extended_adjacency_mat[i]))]).astype(np.float32)

	def get_processed_input(self, states, i,nf=True):
		def state2feature(state):
			feature_arr = self.envs[i].get_fluent_features(state)
			if nf == True:
				feature_arr = np.hstack((feature_arr, self.nf_features[i]))
			
			if my_config.use_type_encoding:
				feature_arr = np.hstack((feature_arr, self.envs[i].instance_parser.type_encoding))
			return feature_arr
		features = np.array(list(map(state2feature, states))).astype(np.float32)
		return features

	def get_processed_graph_input(self, states, i):
		def state2feature(state):
			feature_arr = self.envs[i].get_graph_fluent_features(state)
			return feature_arr
		features = np.array(list(map(state2feature, states))).astype(np.float32)
		return features

	def get_action_details(self,instance):
		return self.action_details[instance]

	def get_extended_action_details(self,instance):
		return self.extended_action_details[instance]

	def get_parsed_state(self, states, instance,extended=False):
		adj_preprocessed_mat = self.get_processed_adj_mat(instance,len(states))
		input_features_preprocessed = self.get_processed_input(states, instance)
		graph_input_features_preprocessed = self.get_processed_graph_input(states, instance)
		return adj_preprocessed_mat, input_features_preprocessed, graph_input_features_preprocessed

	def get_attr(self,instance,batch_size,name):
		return [self.envs[instance].get_attr(name) for _ in range(batch_size)]

	def get_node_dict(self, instance):
		return self.envs[instance].get_node_dict()
	
	def get_num_to_action(self, instance):
		return self.envs[instance].get_num_to_action()
	
	def get_max_reward(self, instance):
		return self.envs[instance].get_max_reward()
	
	def get_min_reward(self, instance):
		return self.envs[instance].get_min_reward()