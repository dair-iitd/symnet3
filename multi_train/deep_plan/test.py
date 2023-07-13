import argparse
import sys
import os
import threading
import multiprocessing
import my_config
import csv
from time import time
import multiprocessing
import pandas as pd
import numpy as np
from env_instance_wrapper import EnvInstanceWrapper
from multiprocessing import Manager

lock = threading.Lock()

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
gym_path = os.path.abspath(os.path.join(curr_dir_path, "../.."))
if gym_path not in sys.path:
    sys.path = [gym_path] + sys.path
parser_path = os.path.abspath(os.path.join(curr_dir_path, "../../utils"))
if parser_path not in sys.path:
    sys.path = [parser_path] + sys.path
network_path = os.path.abspath(os.path.join(curr_dir_path, "networks"))
if network_path not in sys.path:
    sys.path = [network_path] + sys.path

from policy_monitor import PolicyMonitor
from worker_testing import Worker
import helper
from model_factory import ModelFactory


def create_workers(NUM_WORKERS, network, instances, N_train_instances, train_summary_writer, val_summary_writer, model_factory, env_instance_wrapper_all):
    network_copy = model_factory.create_network(env_instance_wrapper_all)
    pe = PolicyMonitor(
        envs=helper.make_envs(instances),
        network=network,
        domain=my_config.domain,
        instances=instances,
        summary_writer=val_summary_writer,
        model_factory=model_factory,
        network_copy=network_copy)
    workers = []
    for worker_id in range(NUM_WORKERS):
        worker_summary_writer = None
        policy_monitor = None
        if worker_id == 0:
            worker_summary_writer = train_summary_writer
            policy_monitor = pe
        worker = Worker(
            worker_id=worker_id,
            envs=helper.make_envs(instances[:N_train_instances]),
            global_network=network,
            domain=my_config.domain,
            instances=instances[:N_train_instances],
            model_factory=model_factory,
            lock=lock,
            policy_monitor=policy_monitor,
            summary_writer=worker_summary_writer)
        workers.append(worker)
    return workers



def evaluate(test_instance, num_episodes, process_index, output_dict):
    is_human_eval = False
    get_random_policy = False
    plot_graph = False
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.set_floatx('float64')

    envs_ = helper.make_envs([test_instance])
    MODEL_DIR, CHECKPOINT_DIR, train_summary_path, val_summary_path = helper.get_test_model_dir()

    NUM_WORKERS = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Don't use GPU for inference
    train_summary_writer = None
    val_summary_writer = None

    policynet_optim = tf.keras.optimizers.RMSprop(lr=my_config.lr, rho=0.99, momentum=0.0, epsilon=1e-6)

    env_instance_wrapper_all = EnvInstanceWrapper(envs_)
    args = helper.create_modelfactory_args(policynet_optim=policynet_optim, instances=[test_instance], env_instance_wrapper=env_instance_wrapper_all)
    helper.add_network_args(args, envs_[0], MODEL_DIR, copy_config=False)

    model_factory = ModelFactory(args)
    network = model_factory.create_network(env_instance_wrapper_all)
    for e in envs_:
        e.close()


    # Create CheckpointManager
    # ckpt_parts = network.get_ckpt_parts()
    ckpt_parts = {}
    ckpt_parts["network"] = network
    ckpt_parts["policynet_optim"] = policynet_optim
    ckpt = tf.train.Checkpoint(**ckpt_parts)
    ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(my_config.trained_model_path, 'checkpoints'), 2000)
    model_factory.set_ckpt_metadata(ckpt, ckpt_manager)

    workers = create_workers(NUM_WORKERS, network, [test_instance],
                             1, train_summary_writer, val_summary_writer,
                             model_factory,env_instance_wrapper_all)
    if not get_random_policy and not is_human_eval:
        model_factory.load_ckpt(my_config.exact_checkpoint)

    policy_monitor = workers[0]
    
    if is_human_eval:
        policy_monitor.evaluate_human(num_episodes)
        return

    print("In process:", process_index)
    total_rewards, _ = policy_monitor.evaluate(num_episodes=num_episodes,
                                               save_model=False, get_random=get_random_policy,
                                               plot_graph=plot_graph, file_name=my_config.trained_model_path)

    sys.stdout = sys.__stdout__
    print("Rewards:", total_rewards)
    output_dict[process_index] = total_rewards
   

def test():
    train_instances, N_train_instances, test_instances, N_test_instances, instances = helper.get_instance_names()
    print(test_instances)
    # Control variables
    get_random_policy = False
    num_episodes = my_config.num_testing_episodes
    
    # print(f"Launching {num_threads} processes")
    manager = Manager()
    rewards_all_instances = []
    output_dict = manager.dict()

    with multiprocessing.Pool(my_config.num_threads) as pool:
        print(pool._processes)
        results = pool.starmap(evaluate, [(inst, num_episodes, i, output_dict) for i, inst in enumerate(test_instances)])


    print(output_dict, len(test_instances))
    for i in range(len(test_instances)):
        rewards_all_instances.append(output_dict[i])

    
    total_rewards = np.array(rewards_all_instances)

    csv_file = os.path.abspath(os.path.join(my_config.trained_model_path, "results.csv"))
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Instance", "Mean Rewards", "Standard deviation", "Rewards(per episode)"])

    for i in range(len(test_instances)):
        current_rewards = total_rewards[i]
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([test_instances[i], np.mean(current_rewards), np.std(current_rewards) / (num_episodes ** 0.5),
                            current_rewards])

if __name__ == '__main__':

    my_config.trained_model_path = os.path.join(my_config.model_dir, f'{my_config.domain}_{my_config.exp_description}')
    my_config.train_instance = ""

    if my_config.setting == "ippc":
        if my_config.domain == 'navigation':
            my_config.test_instance = ",".join([str(910+i) for i in range(40)])
        else:
            my_config.test_instance = ",".join([str(1110+i) for i in range(40)])
    else:
        my_config.test_instance = ",".join([str(3200+i) for i in range(0, 200)])
        if my_config.domain == 'navigation':
            my_config.test_instance = ",".join([str(2200+i) for i in range(200)])
	
    ptr = open(f'{my_config.trained_model_path}/meta_logging.csv')
    ckpt = 1
    best_ckpt, best_rew = 1, -1000000
    for line in ptr.readlines()[2:]:
        if my_config.mode == "ippc":
            toks = line.split(",")[:10] # 10 val instances in ippc
        else:
            toks = line.split(",")[:100] # 100 val instances in lr
        rew = np.mean([float(x) for x in toks])
        if rew > best_rew:
            best_rew = rew
            best_ckpt = ckpt
        ckpt += 1
        
    my_config.exact_checkpoint = str(best_ckpt)
    print("Exact checkpoint:", my_config.exact_checkpoint)
    test()
