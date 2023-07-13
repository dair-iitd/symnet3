config_file = __file__
# ==============================================
# General settings for flags
# ==============================================

'''
IPPC domains:
academic_advising(Acad), navigation(Nav), game_of_life(GoL), crossing_traffic(CT), wildfire(Wild), triangle_tireworld(TT), 
elevators(Elev), sysadmin(Sys), traffic(Traf), recon(Recon), skill_teaching(Skill), tamarisk(Tam)

LR domains:
academic_advising_prob(EAcad), navigation(DNav), corridor(StNav), recon(SRecon), pizza_delivery_windy(Pizza), stochastic_wall(StNav)
'''

setting = "ippc" # Set to ippc to run on ippc domains and lr to run on lr domains
exp_description = "standard" # suffix for model folder
domain = "navigation" # Domain to be tested on
model_dir =  "models/" # Path to model

num_validation_episodes = 5 # Validation episodes for each epoch
num_testing_episodes = 30 # Testing episodes when model has been trained

train_instance = "" # These two flags are set automatically when you run train.py
test_instance = ""

trajectory_dataset_folder = "../../data/datasets/IPPC_2014"
max_transitions_per_instance = 300
last_in_dataset = False # Setting to true might lead to better results since PROST learns while executing

batch_size = 32
train_epochs = 10
ckpt_freq = 1
lr = 0.001
use_pretrained = True
grad_clip_value = 5.0

# Best not to change these. 
add_separate_adj = False
remove_dbn = False
add_edge_type = True
merged_model = True
fc_adjacency = False
split_dbn = False
use_type_encoding = True
use_fluent_for_kl = True
    
mode = "no_kl"

if mode == 'no_dist': # SymNet2.0
    add_aux_loss = False
    decay_aux_loss = False
elif mode == "kl": # SymNet3.0+KL
    add_aux_loss = True
    decay_aux_loss = False
elif mode == "no_kl": # SymNet3.0-KL 
    add_aux_loss = False
    decay_aux_loss = False
elif mode == "kl_decay": # SymNet3.0+KL_{decay}
    add_aux_loss = True
    decay_aux_loss = True