import helper
import my_config
import numpy as np

# Set these variables to run a random agent on these instances.
domain = 'navigation'
num_episodes = 30
instances = [str(i) for i in range(910, 950)]


my_config.domain = domain
envs = helper.make_envs(instances)
writer = open(f'result_scripts/random_results/{domain}_random.csv', 'a')
writer.write("Instance,Mean Reward\n")
writer.close()

for i, inst in enumerate(instances):
    rewards_i = []
    for j in range(num_episodes):
        initial_state, done = envs[i].reset()
        state = initial_state
        episode_length = 0
        episode_reward = 0
        while not done:
            action = np.random.randint(0, envs[i].get_num_actions())
            previous_action = action
            next_state, reward, done, _ = envs[i].test_step(action)

            episode_reward += reward
            state = next_state

        rewards_i.append(episode_reward)

    mean_total_reward = np.mean(rewards_i)
    print(inst, mean_total_reward)
    writer = open(f'result_scripts/random_results/{domain}_random.csv', 'a')
    writer.write(f"{inst},{mean_total_reward}\n")
    writer.close()

