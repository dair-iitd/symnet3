import os
import pandas as pd
import numpy as np
import json

model_categories = ['prost', 'no_distance', 'dbn_dist_kl_correct_masking']
model_categories = ['prost', 'no_dist', 'kl', 'no_kl']

def get_score(domain, models, instances):
    raw_scores = {m: {i: None for i in instances} for m in models}
    for model in models:
        if model != 'random' and model != 'prost':
            ptr = open(f'/scratch/cse/dual/cs5180404/expt/icaps2023_after/more_data_cdac/{model}/{domain}10x10-15x15-symnet3prost/testing_200.csv')
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
            if model.startswith(cat):
                model_agg_scores[cat].append(model_scores[model])
                
    # print(json.dumps(model_agg_scores, indent=4))
    for cat in model_categories:
        if len(model_agg_scores[cat]) > 1:
             # ci = st.norm.interval(alpha=0.95, loc=np.mean(model_agg_scores[cat]), scale=st.sem(model_agg_scores[cat]))
             # model_agg_scores[cat] = str(np.mean(model_agg_scores[cat])) + " +/- " + str(ci[1]-np.mean(model_agg_scores[cat]))
             model_agg_scores[cat] = str(round(np.mean(model_agg_scores[cat]), 2)) + " +/- " + str(round(np.std(model_agg_scores[cat]), 2))
        else:
             model_agg_scores[cat] = str(round(np.mean(model_agg_scores[cat]), 2))
        
    print(json.dumps(model_agg_scores, indent=4))
    
    with open('more_data_results_full.csv', 'a') as output_file:
        scores = ",".join(str(round(model_scores[model], 2)) for model in models)
        # scores = ",".join(str(model_agg_scores[t]) for t in model_categories)
        output_file.write(f"{domain},{scores}\n")
    # print(model_agg_scores)
    print("-------------------------------------\n\n") 


models = ['no_dist_run1', 'no_dist_run2', 'no_dist_run3', 'kl_run1', 'kl_run2', 'kl_run3', 'no_kl_run1', 'no_kl_run2', 'no_kl_run3', 'prost', 'random']
with open('more_data_results_full.csv', 'a') as output_file:
        headers = ",".join(model for model in models)
        output_file.write(f"domain,{headers}\n")


domain = 'recon'
instances = [str(3200+i) for i in range(200)]
get_score(domain, models, instances)


domain = 'pizza_delivery_windy'
instances = [str(3200+i) for i in range(200)]
get_score(domain, models, instances)


domain = 'navigation'
instances = [str(2200+i) for i in range(200)]
get_score(domain, models, instances)


domain = 'stochastic_wall'
instances = [str(3200+i) for i in range(200)]
get_score(domain, models, instances)


domain = 'academic_advising_prob'
instances = [str(3200+i) for i in range(200)]
get_score(domain, models, instances)


domain = 'corridor'
instances = [str(3200+i) for i in range(200)]
get_score(domain, models, instances)

