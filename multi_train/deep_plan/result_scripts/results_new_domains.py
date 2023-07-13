import os
import pandas as pd
import numpy as np
import json

model_categories = ['prost', 'symnet2', 'symnet3']
def get_score(domain, models, instances):
    raw_scores = {m: {i: None for i in instances} for m in models}
    for model in models:
        if model != 'random' and model != 'prost':
            ptr = open(f'/scratch/cse/dual/cs5180404/expt/icaps2023_after/new_domains/{model}/{domain}10x10-15x15-symnet3prost/testing_200.csv')
        elif model == 'random':
            ptr = open(f'random_results/{domain}_random.csv')
        elif model == 'prost':
            ptr = open(f'prost_results/{domain}_prost.csv')
        
        for line in ptr.readlines():
            toks = line.split(",")
            if toks[0] in instances:
                instance = toks[0]
                reward = float(toks[1])
                raw_scores[model][instance] = reward
        ptr.close()
            
    alpha_scores = {m: {i: None for i in instances} for m in models}
    for inst in instances:
        rewards_of_instance = [raw_scores[m][inst] for m in models]
        max_reward_of_instance = max(rewards_of_instance+[raw_scores['random'][inst]])
        min_reward_of_instance = min(rewards_of_instance+[raw_scores['random'][inst]])
        random_reward_of_instance = raw_scores['random'][inst]
        
        if random_reward_of_instance == max_reward_of_instance and random_reward_of_instance != min_reward_of_instance:
            print("WARNING:", inst)
        for model in models:
            # alpha_scores[model][inst] = max(0, (raw_scores[model][inst] - random_reward_of_instance) / (max_reward_of_instance - random_reward_of_instance + 1e-5))
            # alpha_scores[model][inst] = (raw_scores[model][inst] - min_reward_of_instance) / (max_reward_of_instance - min_reward_of_instance + 1e-5)
            # if random_reward_of_instance == max_reward_of_instance:
            #     print(model, inst, raw_scores[model][inst], random_reward_of_instance)
            alpha_scores[model][inst] = (raw_scores[model][inst] - random_reward_of_instance) / (max_reward_of_instance - random_reward_of_instance+1e-5)
            # alpha_scores[model][inst] = (raw_scores[model][inst] - min_reward_of_instance) / (max_reward_of_instance - min_reward_of_instance + 1e-5)
    # print(alpha_scores['run1_no_distance'])
    model_scores = {m: None for m in models}
    for model in models:
        model_scores[model] = np.mean([alpha_scores[model][i] for i in alpha_scores[model].keys()])
    
    print(domain)
    print(json.dumps(model_scores, indent=4))       
        
    model_agg_scores = {m: [] for m in model_categories}
    for model in models:
        for cat in model_categories:
            if cat in model:
                model_agg_scores[cat].append(model_scores[model])
                
    # print(json.dumps(model_agg_scores, indent=4))
    for cat in model_categories:
        model_agg_scores[cat] = np.mean(model_agg_scores[cat])
        
    print(json.dumps(model_agg_scores, indent=4))
    # print(model_agg_scores)
    print("-------------------------------------\n\n") 


domain = 'wall'
models = ['symnet2_run1', 'symnet2_run2', 'symnet2_run3', 'symnet3_run1', 'symnet3_run2', 'symnet3_run3', 'prost', 'random']
instances = [str(3200+i) for i in range(200)]
get_score(domain, models, instances)


domain = 'stochastic_navigation'
models = ['symnet2_run1', 'symnet2_run2', 'symnet2_run3', 'symnet3_run1', 'symnet3_run2', 'symnet3_run3', 'prost', 'random']
instances = [str(3200+i) for i in range(50)]
get_score(domain, models, instances)


