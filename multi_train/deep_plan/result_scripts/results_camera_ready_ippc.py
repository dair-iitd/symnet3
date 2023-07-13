import os
import pandas as pd
import numpy as np
import json

model_categories = ['prost', 'no_dist', 'no_kl', 'kl_decay', 'kl', 'random']
#model_categories = ['prost', 'no_dist', 'no_kl', 'kl', 'random']
#model_categories = ['prost', 'no_dist', 'no_kl', 'kl_decay', 'random']
output_filename = 'result_csvs/ippc_camera_ready_results.csv'
std_dev = False
best3 = False
take_max = True
all_entries = True
def get_score(domain, models, instances):
    raw_scores = {m: {i: None for i in instances} for m in models}
    for model in models:
        if model != 'random' and model != 'prost':
            if domain == 'navigation':
                ptr = open(f'/scratch/cse/dual/cs5180404/expt/uai2023_camera_ready_final/ippc/{model}/{domain}10x10-15x15-symnet3prost/testing_200.csv')
            else:
                ptr = open(f'/scratch/cse/dual/cs5180404/expt/uai2023_camera_ready/ippc/{model}/{domain}10x10-15x15-symnet3prost/testing_200.csv')
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
    
    # print(raw_scores)
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
    
    if all_entries:
        with open(output_filename, 'a') as output_file:
            scores = ",".join(str(round(model_scores[t], 2)) for t in models)
            output_file.write(f"{domain},{scores}\n")             
        return   

    model_agg_scores = {m: [] for m in model_categories}
    for model in models:
        if model in ['prost', 'random']:
            model_agg_scores[model].append(model_scores[model])
        else:
            model_agg_scores[model[:model.index("_run")]].append(model_scores[model])
    if best3:
        for k in model_agg_scores:
            if k not in ['prost', 'random']:
                model_agg_scores[k] = list(reversed(sorted(model_agg_scores[k])))[:3]
    if take_max:
        for k in model_agg_scores:
            if k not in ['prost', 'random']:
                model_agg_scores[k] = [max(model_agg_scores[k])]

    # print(json.dumps(model_agg_scores, indent=4))
    for cat in model_categories:
        if len(model_agg_scores[cat]) > 1:
             # ci = st.norm.interval(alpha=0.95, loc=np.mean(model_agg_scores[cat]), scale=st.sem(model_agg_scores[cat]))
             # model_agg_scores[cat] = str(np.mean(model_agg_scores[cat])) + " +/- " + str(ci[1]-np.mean(model_agg_scores[cat]))
             if std_dev:
                 model_agg_scores[cat] = str(round(np.mean(model_agg_scores[cat]), 2)) + " +/- " + str(round(np.std(model_agg_scores[cat]), 2))
             else:
                 model_agg_scores[cat] = str(round(np.mean(model_agg_scores[cat]), 2))
        else:
             model_agg_scores[cat] = str(round(np.mean(model_agg_scores[cat]), 2))
        
    print(json.dumps(model_agg_scores, indent=4))
    
    with open(output_filename, 'a') as output_file:
        scores = ",".join(str(model_agg_scores[t]) for t in model_categories)
        output_file.write(f"{domain},{scores}\n")
    # print(model_agg_scores)
    print("-------------------------------------\n\n") 

models = ['no_dist_run1', 'no_dist_run2', 'no_dist_run3', 'no_dist_run4', 'no_dist_run5',
          'kl_decay_run1', 'kl_decay_run2', 'kl_decay_run3', 'kl_decay_run4', 'kl_decay_run5',
          'no_kl_run1', 'no_kl_run2', 'no_kl_run3', 'no_kl_run4', 'no_kl_run5',
          'kl_run1', 'kl_run2', 'kl_run3', 'kl_run4', 'kl_run5',
          'prost', 'random']

if all_entries:
    with open(output_filename, 'w') as output_file:
        types = ",".join(models)
        output_file.write(f"Domain,{types}\n")
else:
    with open(output_filename, 'w') as output_file:
        types = ",".join(model_categories)
        output_file.write(f"domain,{types}\n")

domain = 'tamarisk'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'traffic'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'sysadmin'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'skill_teaching'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'navigation'
instances = [str(910+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'triangle_tireworld'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'recon'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'elevators'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'academic_advising'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'crossing_traffic'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'game_of_life'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)

domain = 'wildfire'
instances = [str(1110+i) for i in range(40)]
get_score(domain, models, instances)
