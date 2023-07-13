#!/bin/sh

#PBS -P darpa.ml.cse
#PBS -q high
#PBS -lselect=1:ncpus=12
#PBS -lwalltime=06:00:00

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo $PBS_JOBNAME
echo "==============================="

source /home/apps/anaconda3_2018/4.6.9/etc/profile.d/conda.sh
module load apps/graphviz/2.38.0/intel
module load apps/anaconda/3
module load apps/tensorflow/2.5/gpu/python3.6
module load compiler/gcc/11.2.0
module load compiler/java/JDK/13.0.2
cd /home/cse/dual/cs5180404/scratch/symnet3/multi_train/deep_plan

DOMAIN=$PBS_JOBNAME

# python -u train_supervised.py --domain $domain --model_dir $model_dir --train_instances $train_instances --val_instances $val_instances
# python -u validate.py --domain $domain --trained_model_path $trained_model_path --num_threads $num_threads --val_instance $val_instance
python -u test.py --domain $domain --trained_model_path $trained_model_path --num_threads $num_threads --test_instance $test_instance

