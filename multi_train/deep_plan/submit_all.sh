#!/bin/sh
# declare -a runs=("no_dist_run1" "no_dist_run2" "no_dist_run3" "no_dist_run4" "no_dist_run5")
# declare -a runs=("no_dist_run1" "no_dist_run2" "no_dist_run3")
# declare -a runs=("no_dist_run4" "no_dist_run5")
# declare -a runs=("no_kl_run1" "no_kl_run2" "no_kl_run3")
# declare -a runs=("no_kl_run4" "no_kl_run5")
# declare -a runs=("kl_run1" "kl_run2" "kl_run3")
# declare -a runs=("kl_run4" "kl_run5")
# declare -a runs=("kl_run1_64" "kl_run2_64" "kl_run3_64" "kl_run4_64" "kl_run5_64")
# declare -a runs=("kl_decay_run1_64" "kl_decay_run2_64" "kl_decay_run3_64" "kl_decay_run4_64" "kl_decay_run5_64")
# declare -a runs=("no_kl_run1_64" "no_kl_run2_64" "no_kl_run3_64" "no_kl_run4_64" "no_kl_run5_64")
# declare -a runs=("no_dist_run1_64" "no_dist_run2_64" "no_dist_run3_64" "no_dist_run4_64" "no_dist_run5_64")
# declare -a runs=("kl_run1_v100_64" "kl_run2_v100_64" "kl_run3_v100_64" "kl_run4_v100_64" "kl_run5_v100_64")
# declare -a runs=("no_kl_run4" "no_kl_run5")
declare -a runs=("kl_decay_run1" "kl_decay_run2" "kl_decay_run3" "kl_run1" "kl_run2" "kl_run3" "no_kl_run1" "no_kl_run2" "no_kl_run3")
# declare -a runs=("kl_decay_run1" "no_kl_run2" "no_kl_run3")
# declare -a runs=("kl_decay_run4" "kl_decay_run5" "kl_run4" "kl_run5" "no_kl_run4" "no_kl_run5")
# declare -a runs=("no_dist_run1")
# declare -a domains=("academic_advising" "navigation" "skill_teaching" "tamarisk" "traffic" "sysadmin" "recon" "crossing_traffic" "elevators" "game_of_life" "triangle_tireworld" "wildfire")
# declare -a domains=("academic_advising" "navigation" "skill_teaching" "tamarisk" "traffic" "sysadmin" "recon" "crossing_traffic" "elevators")
# declare -a domains=("elevators" "game_of_life" "triangle_tireworld" "wildfire")
# declare -a domains=("triangle_tireworld" "game_of_life" "wildfire")
# declare -a domains=("academic_advising" "skill_teaching" "tamarisk" "traffic" "sysadmin" "recon" "crossing_traffic" "elevators" "game_of_life" "triangle_tireworld" "wildfire")

declare -a domains=("recon" "navigation" "corridor" "stochastic_wall" "academic_advising_prob" "pizza_delivery_windy")
declare -a domains=("navigation" "corridor" "stochastic_wall" "academic_advising_prob")
declare -a domains=("recon" "pizza_delivery_windy")

# declare -a domains=("pizza_delivery_windy" "stochastic_wall")
# declare -a domains=("pizza_delivery_windy" "corridor")
# declare -a domains=("navigation")
# declare -a domains=("academic_advising")
# declare -a domains=("triangle_tireworld")
# declare -a domains=("recon" "pizza_delivery_grid" "navigation")
# declare -a domains=("stochastic_navigation")
# declare -a domains=("pizza_delivery_grid")
# declare -a domains=("wall" "stochastic_navigation")
# declare -a domains=("tamarisk" "game_of_life" "traffic")
# declare -a domains=("game_of_life" "wildfire")
# declare -a domains=("tamarisk" "crossing_traffic")
# declare -a domains=("academic_advising")
# declare -a domains=("academic_advising" "skill_teaching" "game_of_life")
# declare -a domains=("skill_teaching" "academic_advising")
# declare -a domains=("skill_teaching")
# declare -a domains=("tamarisk" "wildfire" "sysadmin")
# declare -a domains=("triangle_tireworld")
# declare -a domains=("recon")

log_folder="/home/cse/dual/cs5180404/scratch/logs/uai2023_camera_ready_final/lr/"
mode="test"

for i in "${domains[@]}"
do
    for j in "${runs[@]}"
    do
        mkdir $log_folder""$j"/"
        if [ "$mode" = "train" ]; then
            qsub -N $i -o $log_folder""$j"/"$i"_"$mode.o -e $log_folder""$j"/"$i"_"$mode.e -v 'domain='"$i"', model_dir=/home/cse/dual/cs5180404/scratch/expt/uai2023_camera_ready_final/lr/'"$j/"', train_instances="1,2,3", val_instances="4"' submit.sh
        fi
        
        if [ "$mode" = "test" ]; then
            qsub -N $i -o $log_folder""$j"/"$i"_"$mode.o -e $log_folder""$j"/"$i"_"$mode.e -v 'domain='"$i"', trained_model_path=/home/cse/dual/cs5180404/scratch/expt/uai2023_camera_ready_final/lr/'"$j/$i"'10x10-15x15-symnet3prost, num_threads=12, val_instance="1", test_instance="1"' submit.sh
        fi
    done
done
