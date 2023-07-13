
#rddl/domains/sysadmin_mdp.rddl
#rddl/domains/sysadmin_inst_mdp__900.rddl
#sysadmin_inst_mdp__900
#900

echo "relative_path_domain_file relative_path_instance_file name_of_instance instance_number"
echo "./new_instance.sh rddl/domains/sysadmin_mdp.rddl rddl/domains/sysadmin_inst_mdp__900.rddl sysadmin_inst_mdp__900 900"

# Copy the instance file
(
    # Wait for lock on /var/lock/.myscript.exclusivelock (fd 200) for 10 seconds
    flock -s -x -w 300 200

    # Do stuff
    cp ../$2 ../gym/envs/rddl/$2

    # Create a dbn file
    mkdir temp_merger_$3
    cat ../$1 ../$2 > ./temp_merger_$3/temp.rddl

    ./run rddl.viz.RDDL2Graph ./temp_merger_$3/temp.rddl $3

    # Copy the dbn file
    cd ..
    cp ./rddlsim-master/tmp_rddl_graphviz.dot ./rddl/dbn/$3.dot
    cp ./rddl/dbn/$3.dot ./gym/envs/rddl/rddl/dbn/

    # Create a dot file
    ./rddl/lib/rddl-parser $1 $2 .

    # Copy dot ile
    cp $3 ./rddl/parsed/$3
    cp $3 ./gym/envs/rddl/rddl/parsed/$3

    # Copy env ile
    # cp ./rddl/lib/clibxx.so ./rddl/lib/clibxx$4.so 
    # cp ./rddl/lib/clibxx.so ./gym/envs/rddl/rddl/lib/clibxx$4.so 
    # chmod +x ./gym/envs/rddl/rddl/lib/clibxx$4.so

    # rm ./rddlsim-master/tmp_rddl_graphviz.dot
    rm $3


) 200>./.myscript.exclusivelock
