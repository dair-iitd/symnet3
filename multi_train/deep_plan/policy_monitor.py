import os, json
import numpy as np
import time
import math

import tensorflow as tf
from env_instance_wrapper import EnvInstanceWrapper
from model_factory import ModelFactory
import helper
from tqdm import tqdm

class PolicyMonitor(object):
	def __init__(self, envs, network, domain, instances, summary_writer, model_factory,network_copy):
		self.domain = domain
		self.instances = instances
		self.envs = envs
		self.network = network
		self.summary_writer = summary_writer
		self.env_instance_wrapper = EnvInstanceWrapper(envs)
		self.model_factory = model_factory
		self.network_copy = network_copy
		self.eval_step = 1

	def copy_params(self):
		ModelFactory.copy_params(self.network_copy.trainable_variables, self.network.trainable_variables)

	def eval_once(self, num_episodes=5, save_model=True, get_random=False, plot_graph=False, file_name=None,
				verbose=False, test_envs=None, expert=None, meta_logging=False, get_attn_map=False, get_node_emb=False):

		verbose = False
		start = time.time()
		mean_total_rewards = []
		std_total_rewards = []
		std_error_rewards = []
		mean_episode_lengths = []
		previous_action = 0
		total_rewards = []
		image_name = None
		if plot_graph:
			file_name = os.path.abspath(os.path.join(file_name, "graphs"))
		save_path = None
		if save_model:
			save_path = self.model_factory.save_ckpt()
			time.sleep(5)

		if test_envs is not None:
			self.envs = test_envs

		attn_maps, node_embs = [], []
		for i in range(len(self.envs)):
			if verbose:
				print("env = %d" % (i))
			rewards_i = []
			episode_lengths_i = []

			#  Update file name for plotting
			if plot_graph:
				dir_name = os.path.abspath(os.path.join(file_name, "{}".format(self.envs[i].instance)))
				os.makedirs(dir_name, exist_ok=True)

			state_action_cache = {}  # Caches actions so that a forward pass is not done each time
			for j in tqdm(range(num_episodes), desc=f'Env {self.envs[i].instance}'):
				initial_state, done = self.envs[i].reset()
				state = initial_state
				episode_reward = 0.0
				episode_length = 0
				print("----------------------------\n\n") if verbose else None
				while not done:

					#  Update file name for plotting
					if plot_graph:
						image_name = os.path.abspath(os.path.join(dir_name, "{}.png".format(episode_length)))

					if not get_random:
						state_str = np.array_str(state)
						if (state_str not in state_action_cache) or get_attn_map:
							if get_attn_map:
								logits, attn = self.network_copy.policy_prediction([state], i, self.env_instance_wrapper,
																		plot_graph=plot_graph,
																		file_name=image_name, action_taken=previous_action,
																		training=False, return_attn_coef=True)
								attn_maps.append((state, attn.numpy()))
								action = tf.argmax(tf.reshape(logits, [-1])).numpy()
							elif get_node_emb:
								logits, node_emb = self.network_copy.policy_prediction([state], i, self.env_instance_wrapper,
																		plot_graph=plot_graph,
																		file_name=image_name, action_taken=previous_action,
																		training=False, return_attn_coef=False, return_node_emb=True)
								node_embs.append((state, node_emb))
								action = tf.argmax(tf.reshape(logits, [-1])).numpy()
							else:
								action = tf.argmax(tf.reshape(
									self.network_copy.policy_prediction([state], i, self.env_instance_wrapper,
																		plot_graph=plot_graph,
																		file_name=image_name, action_taken=previous_action,
																		training=False), [-1])).numpy()
							
							state_action_cache[state_str] = action
						else:
							action = state_action_cache[state_str]
					else:
						action = np.random.randint(0, self.envs[i].get_num_actions())

					# Update previous action for plotting
					previous_action = action
					next_state, reward, done, _ = self.envs[i].test_step(action)

					if verbose:
						state_repr = self.envs[i].get_state_repr(state)
						print("Episode: ", episode_length)
						if type(state_repr) == dict:
							print("State:\n", json.dumps(state_repr, sort_keys=True, indent=4))
						else:
							print("State:")
							print(state_repr)
						if action == 0:
							print("Action taken: noop")
						else:
							print("Action taken:", self.env_instance_wrapper.get_num_to_action(i)[action])
						print("Reward:", reward)
						print("--------------------------------------------------------------\n\n")

					episode_reward += reward
					episode_length += 1
					state = next_state


				rewards_i.append(episode_reward)
				episode_lengths_i.append(episode_length)
			mean_total_reward = np.mean(rewards_i)
			mean_episode_length = np.mean(episode_lengths_i)
			std_total_reward = np.std(rewards_i)
			mean_total_rewards.append(mean_total_reward)
			mean_episode_lengths.append(mean_episode_length)
			std_total_rewards.append(std_total_reward)
			std_error_rewards.append(std_total_reward / math.sqrt(num_episodes))
			total_rewards.append(rewards_i)
			print("Instance:", i, "Mean reward:", mean_total_reward)

		end = time.time() - start


		str_to_print = ",".join([str(mr) for mr in mean_total_rewards]) + "\n"
		if meta_logging:
			helper.write_content(helper.meta_logging_file, str_to_print)

		print("\n==============")
		print("std_total_rewards = " + str(std_error_rewards))
		print("==============")
		print(str_to_print)
		if get_attn_map:
			return mean_total_rewards, mean_episode_lengths, end, total_rewards, attn_maps, save_path
		elif get_node_emb:
			return mean_total_rewards, mean_episode_lengths, end, total_rewards, node_embs, save_path
		else:
			return mean_total_rewards, mean_episode_lengths, end, total_rewards, save_path
