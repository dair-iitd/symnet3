!/bin/sh

# declare -a domains=("academic_advising" "crossing_traffic" "game_of_life" "navigation" "skill_teaching" "sysadmin" "tamarisk" "traffic" "wildfire" "recon" "triangle_tireworld" "elevators")

declare -a domains=("academic_advising")
# declare -a instances=("701" "702" "703" "704" "705" "706" "707" "708" "709" "710" "711" "712")
# declare -a instances=("821" "822" "823" "824" "825" "826" "827" "828" "828" "829" "830" "831" "832" "833" "834" "835" "836" "837" "838" "839" "840" "841" "842" "843" "844")
# declare -a instances=("1000" "1001" "1002" "1003" "1004" "1005" "1006" "1007" "1008" "1009" "1010" "1011" "1013" "1014")
# declare -a instances=("1001" "1002" "1003" "1005" "1007" "1008" "1009" "1010" "1011" "1012" "1014")
# declare -a instances=("1001" "1002" "1005" "1006" "1007" "1013" "1014")
declare -a instances=($(seq 900 950))

for i in "${domains[@]}"
do
	for j in "${instances[@]}"
	# for j in {10001..10084}
	do
    	# echo ./new_instance.sh "rddl/domains/"$i"_mdp.rddl" "rddl/domains/"$i"_inst_mdp__"$j".rddl" $i"_inst_mdp__"$j $j
    	./new_instance.sh "rddl/domains/"$i"_mdp.rddl" "rddl/domains/"$i"_inst_mdp__"$j".rddl" $i"_inst_mdp__"$j $j
    done
done
