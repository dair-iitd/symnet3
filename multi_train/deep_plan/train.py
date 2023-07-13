import sys
import os
import multiprocessing
import my_config
from env_instance_wrapper import EnvInstanceWrapper
import time
import pickle
import pdb
import random
import numpy as np

lock = multiprocessing.Lock()

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
gym_path = os.path.abspath(os.path.join(curr_dir_path, "../.."))
if gym_path not in sys.path:
    sys.path = [gym_path] + sys.path
parser_path = os.path.abspath(os.path.join(curr_dir_path, "../../utils"))
if parser_path not in sys.path:
    sys.path = [parser_path] + sys.path
network_path = os.path.abspath(os.path.join(curr_dir_path,"networks"))
if network_path not in sys.path:
    sys.path = [network_path] + sys.path

import tensorflow as tf
from policy_monitor import PolicyMonitor
import helper
from model_factory import ModelFactory
from supervised_dataset import SupervisedDataset

load_saved_dataset = False
# @tf.function
def train_step(network, x, y, instance, env_instance_wrapper, loss_fn, optimizer, grad_clip_value, step, multiplier=0.0):
    if my_config.add_aux_loss:
        with tf.GradientTape() as policynet_tape:
            policynet_pred, dist_attn_coef = network.policy_prediction(x, instance, env_instance_wrapper, return_attn_coef=True)
            policynet_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y, policynet_pred)
            if my_config.use_fluent_for_kl:
                random_node = tf.constant(random.randint(0, len(env_instance_wrapper.envs[instance].instance_parser.fluent_nodes)-1), dtype=tf.int32)
            else:
                random_node = tf.constant(random.randint(0, len(env_instance_wrapper.envs[instance].instance_parser.node_dict)-1), dtype=tf.int32)
            
            random_heads = tf.constant(random.sample(list(np.arange(0, dist_attn_coef.shape[3])), k=2), dtype=tf.int32)
            attn_coef0 = dist_attn_coef[:,random_node,:,random_heads[0]] # shape-> BXN
            attn_coef1 = dist_attn_coef[:,random_node,:,random_heads[1]] # shape-> BXN

            aux_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(attn_coef0, attn_coef1)
            total_loss = policynet_loss - multiplier * aux_loss

        grads_of_policynet = policynet_tape.gradient(total_loss, network.trainable_variables)
        grads_of_policynet = tf.clip_by_global_norm(grads_of_policynet, grad_clip_value)

        optimizer.apply_gradients(zip(grads_of_policynet[0], network.trainable_variables))
        return policynet_loss, aux_loss

    else:
        with tf.GradientTape() as policynet_tape:
            policynet_pred = network.policy_prediction(x, instance, env_instance_wrapper)
            policynet_loss = loss_fn(y, policynet_pred)
        grads_of_policynet = policynet_tape.gradient(policynet_loss, network.trainable_variables)
        grads_of_policynet = tf.clip_by_global_norm(grads_of_policynet, grad_clip_value)

        optimizer.apply_gradients(zip(grads_of_policynet[0], network.trainable_variables))
        return policynet_loss

def train():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.set_floatx('float64')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    train_instances, N_train_instances, test_instances, N_test_instances, instances = helper.get_instance_names()
    envs_ = helper.make_envs(instances)
    num_nodes_list, num_valid_actions_list, num_graph_fluent_list, num_adjacency_list = helper.get_env_metadata(envs_)
    MODEL_DIR, CHECKPOINT_DIR, train_summary_path, val_summary_path = helper.get_model_dir()
    NUM_WORKERS = 1
    train_summary_writer = None
    val_summary_writer = None

    policynet_optim = tf.keras.optimizers.Adam(lr=my_config.lr)

    args = helper.create_modelfactory_args(policynet_optim=policynet_optim)
    helper.add_network_args(args, envs_[0],MODEL_DIR)

    model_factory = ModelFactory(args)
    env_instance_wrapper = EnvInstanceWrapper(envs_[:N_train_instances])

    network = model_factory.create_network(env_instance_wrapper)
    network.init_network(env_instance_wrapper, 0)

    # Create policy_monitor
    network_copy = model_factory.create_network(env_instance_wrapper)
    pe = PolicyMonitor(
        envs=helper.make_envs(test_instances),
        network=network,
        domain=my_config.domain,
        instances=instances,
        summary_writer=val_summary_writer,
        model_factory=model_factory,
        network_copy=network_copy)

    network.init_network(env_instance_wrapper, 0)
    pe.network_copy.init_network(env_instance_wrapper, 0)
    pe.copy_params()

    for e in envs_:
        e.close()

    # Create CheckpointManager
    ckpt_parts = {}
    ckpt_parts["network"] = network
    ckpt_parts["policynet_optim"] = policynet_optim
    ckpt = tf.train.Checkpoint(**ckpt_parts)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, 2000)
    model_factory.set_ckpt_metadata(ckpt, ckpt_manager)

    if my_config.use_pretrained:
        model_factory.load_ckpt(ckpt_num=None)


    # SUPERVISED TRAINING STARTS
    # Training dataset
    dataset_folder = my_config.trajectory_dataset_folder
    batch_size = my_config.batch_size
    dataset_ob = SupervisedDataset(train_instances, env_instance_wrapper, dataset_folder, batch_size, num_episodes=None)
    # Loss Function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    grad_clip_value = model_factory.grad_clip_value
    step = 0
    best_val_reward = -float('inf')
    best_ckpt = ""
    for epoch in range(my_config.train_epochs):
        print("\n\n------Start of epoch %d" % (epoch,))
        start_time = time.time()
        random.shuffle(dataset_ob.instance_order)
        for ins in dataset_ob.instance_order:
            instance, states, actions = dataset_ob.dataset[ins]
            total_size = len(states)
            cur_loc= 0
            while cur_loc < total_size:
                if cur_loc + batch_size < total_size:
                    x = states[cur_loc: cur_loc + batch_size]
                    y = actions[cur_loc: cur_loc + batch_size]
                    cur_loc += batch_size
                else:
                    x = states[cur_loc:]
                    y = actions[cur_loc:]
                    cur_loc = total_size+1

                if my_config.add_aux_loss:
                    if my_config.decay_aux_loss:
                        if step < 2000:
                            multiplier = 0.1
                        elif 2000 <= step < 3000:
                            multiplier = 0.1 * (3000 - step) / 1000
                        else:
                            multiplier = 0
                    else:
                        multiplier = 0.1
                    
                    loss_value, aux_value = train_step(network, x, y, instance, env_instance_wrapper, loss_fn, model_factory.policynet_optim, grad_clip_value, step, multiplier=multiplier)
                else:
                    loss_value = train_step(network, x, y, instance, env_instance_wrapper, loss_fn, model_factory.policynet_optim, grad_clip_value, step)
                step += 1
            if my_config.add_aux_loss:
                print("Instance %d \t| Total steps: %d | Imitation loss: %.4f | KL loss: %.4f | KL multiplier: %.4f" % (ins, step-1, float(loss_value), float(aux_value), multiplier))
            else:
                print("Instance %d \t| Total steps: %d | Imitation loss: %.4f" % (ins, step-1, float(loss_value)))
            
        # Validation
        if epoch%my_config.ckpt_freq == my_config.ckpt_freq-1:
            pe.copy_params()
            _, _, eval_time, total_rewards, save_path = pe.eval_once(meta_logging=True, num_episodes=my_config.num_validation_episodes)
            if np.mean(total_rewards) <= best_val_reward:
                os.remove(save_path + ".index")
                os.remove(save_path + ".data-00000-of-00001")
                print(f"This checkpoint has a reward of {np.mean(total_rewards)}, best is {best_val_reward}, deleting this.")
            else:
                if best_ckpt != "":
                    os.remove(best_ckpt + ".index")
                    os.remove(best_ckpt + ".data-00000-of-00001")
                best_ckpt = save_path
                print(f"This checkpoint has a reward of {np.mean(total_rewards)}, best was {best_val_reward}, deleting previous.")
                
            best_val_reward = max(best_val_reward, np.mean(total_rewards))
                


if __name__ == '__main__':
    
    if my_config.setting == "ippc":
        my_config.train_instance = ",".join(str(900+i) for i in range(20))   
        my_config.test_instance = ",".join(str(1100+i) for i in range(10))

        # Some training files weren't created properly which is why they've been excluded from training
        if my_config.domain == 'navigation':
            my_config.train_instance = ",".join([str(700+i) for i in range(200) if 700+i not in [705,798]])
            my_config.test_instance = ",".join(str(900+i) for i in range(10))
        
        if my_config.domain == 'triangle_tireworld':
            my_config.train_instance = ",".join(str(1000+i) for i in range(100)) # Generator doesn't have enough diversity in parameters, only need a 100 instances
            my_config.test_instance = ",".join(str(1100+i) for i in range(10))
    
    elif my_config.setting == "lr":
        if my_config.domain == 'recon':
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2177])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))
        if my_config.domain == 'academic_advising_prob':
            # These instances were rejected because their score is worse than a no-op policy
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2100,2119,2126,2130,2143,2150,2183,2185,2189,2194,2203,2205,2242,2258,2283,2290,2296,2329,2336,2344,2351,2417,2432,2472,2482,2488,2505,2508,2523,2530,2540,2569,2586,2589,2597,2605,2613,2623,2640,2686,2730,2747,2778,2780,2869,2889,2961,2994,2998,3023,3042,3081,3095])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))
        if my_config.domain == 'pizza_delivery_windy':
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2692])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))        
        if my_config.domain == 'navigation':
            my_config.train_instance = ",".join([str(1100+i) for i in range(1000) if 1100+i not in [1333,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979]])
            my_config.test_instance = ",".join([str(2100+i) for i in range(100)])
        if my_config.domain == 'stochastic_wall':
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2806,2835,2837,3069,3080])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))
        if my_config.domain == 'corridor':
            my_config.train_instance = ",".join(str(2100+i) for i in range(1000) if 2100+i not in [2211])
            my_config.test_instance = ",".join(str(3100+i) for i in range(100))

    
    print("Domain: ", my_config.domain)
    print("Model dir: ", my_config.model_dir)
    print(my_config.train_instance, my_config.test_instance)
    train()
