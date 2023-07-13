#!/bin/bash

for i in $(seq 11 30);
do
	./new_instance.sh rddl/domains/sysadmin_mdp.rddl rddl/domains/sysadmin_inst_mdp__$i.rddl sysadmin_inst_mdp__$i $i
done

for i in $(seq 1 5);
do
	./new_instance.sh rddl/domains/sysadmin_mdp.rddl rddl/domains/sysadmin_inst_mdp__10_$i.rddl sysadmin_inst_mdp__10_$i 10_$i
done

for i in $(seq 1 5);
do
	./new_instance.sh rddl/domains/game_of_life_mdp.rddl rddl/domains/game_of_life_inst_mdp__5_$i.rddl game_of_life_inst_mdp__5_$i $i
done

for i in $(seq 1 5);
do
	./new_instance.sh rddl/domains/game_of_life_mdp.rddl rddl/domains/game_of_life_inst_mdp__10_$i.rddl game_of_life_inst_mdp__10_$i $i
done

for i in $(seq 1 5);
do
	./new_instance.sh rddl/domains/wildfire_mdp.rddl rddl/domains/wildfire_inst_mdp__5_$i.rddl wildfire_inst_mdp__5_$i $i
done

for i in $(seq 1 5);
do
	./new_instance.sh rddl/domains/wildfire_mdp.rddl rddl/domains/wildfire_inst_mdp__10_$i.rddl wildfire_inst_mdp__10_$i $i
done