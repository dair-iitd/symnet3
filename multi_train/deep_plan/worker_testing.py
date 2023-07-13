import collections
import numpy as np
import tensorflow as tf
import time

from model_factory import ModelFactory
from env_instance_wrapper import EnvInstanceWrapper
import numpy as np

import my_config as my_config

class Worker(object):
	def __init__(self, worker_id, envs, global_network, domain, instances,
		model_factory, lock, policy_monitor=None, summary_writer=None):
		self.worker_id = worker_id
		self.domain = domain
		self.instances = instances
		self.model_factory = model_factory
		self.global_network = global_network
		self.global_counter = model_factory.global_counter
		self.summary_writer = summary_writer
		self.envs = envs
		self.current_instance = 0
		self.state = None
		self.env_instance_wrapper = EnvInstanceWrapper(envs)
		self.local_network = model_factory.create_network(self.env_instance_wrapper)
		self.local_network.init_network(self.env_instance_wrapper,0)
		self.lock = lock
		self.policy_monitor = policy_monitor

		# ! Only for academic advising
		self.degree_completed = False
		self.steps_taken = 0

		# For reward prediction
		self.action_noop = 0

		if self.policy_monitor is not None:
			self.global_network.init_network(self.env_instance_wrapper, 0)
			self.policy_monitor.network_copy.init_network(self.env_instance_wrapper, 0)
			self.policy_monitor.copy_params()
		self.copy_params()

		# To train using a dataset
		self.save_transitions = self.train_from_dataset = False

		self.masks = [None, None]
		self.masks = [None, None]

	def copy_params(self, acquire_lock=True):
		if not acquire_lock:
			ModelFactory.copy_params(self.local_network.trainable_variables, self.global_network.trainable_variables)
			return
		self.lock.acquire()
		ModelFactory.copy_params(self.local_network.trainable_variables, self.global_network.trainable_variables)
		self.lock.release()

	def evaluate(self, num_episodes=5, save_model=True, get_random = False, plot_graph=False, file_name=None, test_envs=None, get_attn_map=False, get_node_emb=False):
		self.policy_monitor.copy_params()
		if get_attn_map:
			_,_,eval_time,total_rewards, attn_maps, save_path = self.policy_monitor.eval_once(num_episodes, save_model, get_random, plot_graph, file_name,test_envs=test_envs, get_attn_map=get_attn_map, get_node_emb=get_node_emb)
			return total_rewards, eval_time, attn_maps
		elif get_node_emb:
			_,_,eval_time,total_rewards, node_embs, save_path = self.policy_monitor.eval_once(num_episodes, save_model, get_random, plot_graph, file_name,test_envs=test_envs, get_attn_map=get_attn_map, get_node_emb=get_node_emb)
			return total_rewards, eval_time, node_embs
		else:
			_,_,eval_time,total_rewards, save_path = self.policy_monitor.eval_once(num_episodes, save_model, get_random, plot_graph, file_name,test_envs=test_envs, get_attn_map=get_attn_map, get_node_emb=get_node_emb)
			return total_rewards, eval_time
		