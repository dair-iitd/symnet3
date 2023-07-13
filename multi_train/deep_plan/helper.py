import os
import numpy as np
import shutil
import sys
import my_config
import symnet3_config
from datetime import datetime

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
gym_path = os.path.abspath(os.path.join(curr_dir_path, "../.."))
if gym_path not in sys.path:
	sys.path = [gym_path] + sys.path
import gym

def get_instance_names():
	train_instances,test_instances = [],[]
	for instance_num in my_config.train_instance.strip().split(","):
		instance = "{}".format(instance_num)
		train_instances.append(instance)
	for instance_num in my_config.test_instance.strip().split(","):
		instance = "{}".format(instance_num)
		test_instances.append(instance)
	instances = []
	instances.extend(train_instances)
	instances.extend(test_instances)
	return train_instances,len(train_instances),test_instances,len(test_instances),instances

def make_envs(instances):
	envs = []
	for instance in instances:
		env_name = "RDDL-{}{}-v1".format(my_config.domain, instance)
		env = gym.make(env_name)
		envs.append(env)
	return envs

def get_env_metadata(envs_):
	num_nodes_list = []
	num_valid_actions_list = []
	num_graph_fluent_list = []
	num_adjacency_list = []
	for env_ in envs_:
		num_nodes_list.append(env_.get_num_nodes())
		num_valid_actions_list.append(env_.get_num_actions())
		num_graph_fluent_list.append(env_.get_num_graph_fluents())
		num_adjacency_list.append(env_.get_num_adjacency_list())
	return num_nodes_list, num_valid_actions_list, num_graph_fluent_list, num_adjacency_list

def get_model_dir():
	MODEL_DIR = my_config.model_dir
	model_suffix = f"{my_config.domain}_{my_config.exp_description}"
	MODEL_DIR = os.path.abspath(os.path.join(MODEL_DIR, model_suffix))
	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
	if not os.path.exists(CHECKPOINT_DIR):
		os.makedirs(CHECKPOINT_DIR)
	train_summary_path = os.path.join(MODEL_DIR, "train_summaries")
	val_summary_path = os.path.join(MODEL_DIR, "val_summaries")

	# init_meta_logging
	shutil.copy(os.path.abspath("my_config.py"), MODEL_DIR)
	shutil.copy(os.path.abspath("policy_monitor.py"), MODEL_DIR)
	shutil.copy(os.path.abspath("networks/symnet3/symnet3.py"), MODEL_DIR)

	os.makedirs(os.path.join(MODEL_DIR, "instances"), exist_ok=True)

	str_to_print = str(datetime.now()) + "\n"
	str_to_print += my_config.test_instance + "\n"
	global meta_logging_file
	meta_logging_file = os.path.join(MODEL_DIR, "meta_logging.csv")
	write_content(meta_logging_file, str_to_print)

	return MODEL_DIR, CHECKPOINT_DIR, train_summary_path, val_summary_path

def failsafe():
	print("================================================================")
	print("domain = " + my_config.domain)
	print("model_dir = " + my_config.model_dir)
	print("train_instance = " + my_config.train_instance)
	print("test_instance = " + my_config.test_instance)
	print("t_max = " + str(my_config.t_max))
	print()
	print("symnet3_params = " + str(symnet3_config.ge_params))
	print("================================================================")
	input()

def create_modelfactory_args(**kwargs):
	args = {}
	for key,value in kwargs.items():
		args[key] = value
	if "policynet_optim" not in args.keys():
		args["policynet_optim"] = None
	args["grad_clip_value"] = my_config.grad_clip_value

	return args

def add_network_args(args, env, MODEL_DIR, copy_config=True):
	args["general_params"] = symnet3_config.general_params
	args["se_params"] = symnet3_config.se_params
	args["ad_params"] = symnet3_config.ad_params
	args["ge_params"] = symnet3_config.ge_params
	args["tm_params"] = symnet3_config.tm_params

	args["se_params"]["num_se"] = env.get_num_adjacency_list()
	if my_config.fc_adjacency:
		args["se_params"]["num_se"] += 1
	args["ad_params"]["num_action_templates"] = env.get_num_type_actions()

	if copy_config:
		shutil.copy(os.path.abspath("symnet3_config.py"), MODEL_DIR)

def get_adj_mat_from_list(adjacency_list):
	l = len(adjacency_list)
	adj_mat = np.array(np.zeros((l, l), dtype=np.int32), dtype=np.int32)
	for key, value in adjacency_list.items():
		for val in value:
			adj_mat[key, val] = 1
	for i in range(l):
		adj_mat[i][i] = 1
	return adj_mat

def write_content(file_path, content):
	with open(file_path, 'a') as f:
		f.write(content)

# ===============================================================
# For Testing
# ===============================================================

def get_test_model_dir():
	MODEL_DIR = my_config.model_dir
	model_suffix = f"{my_config.domain}_{my_config.exp_description}"
	MODEL_DIR = os.path.abspath(os.path.join(MODEL_DIR, model_suffix))

	CHECKPOINT_DIR = os.path.abspath(os.path.join(my_config.trained_model_path, "checkpoints"))

	if not os.path.exists(CHECKPOINT_DIR):
		print("Incorrect CHECKPOINT_DIR :\n" + CHECKPOINT_DIR)
		exit(-1)
	train_summary_path = os.path.join(MODEL_DIR, "train_summaries")
	val_summary_path = os.path.join(MODEL_DIR, "val_summaries")

	return MODEL_DIR, CHECKPOINT_DIR, train_summary_path, val_summary_path
