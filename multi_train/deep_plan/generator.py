import subprocess, os, sys, random, shutil
import generator_config
from transition_dataset_builder import generate_dataset
import helper
# This file is for generating new domains using the generators
class Generator():
    def __init__(self):
        self.script_args = generator_config.script_args
        self.script_name = generator_config.script_name
        
        self.generator_dir = '../../data/datasets/rddlsim/src/'
        self.script_path = 'rddl/competition/generators/'
        self.cp = '.:rddl/competition/generators/commons-math3-3.6.1.jar'
        self.symnet_folder = '../../'
     
    def print_args(self, domain, args):
        arg_str = []
        for i, arg_type in enumerate(self.script_args[domain]):
            arg_str.append(f'{arg_type["param_name"]} : {args[i]}')
        print(f"Generating RDDL with arguments-{','.join(arg_str)}")
        
    def generate_instance(self, domain, name='21', verbose=False):
        curr_dir = os.getcwd()
        output_dir = os.path.join(self.symnet_folder, "rddl/domains/")
        script_name = os.path.join(self.script_path, self.script_name[domain])
        cp = self.cp
        instance_name = f'{domain}_inst_mdp__{name}'
        if domain in ['academic_advising_chain', 'academic_advising_prob', 'pizza_delivery', 'pizza_delivery_grid', 'pizza_delivery_windy', 'wall', 'stochastic_navigation', 'stochastic_wall', 'corridor']:
            command = f'python {script_name}'
        else:
            command = f'java -cp {cp} {script_name}'

        while True:
            args = []
            for arg_types in self.script_args[domain]:
                if arg_types['param_name'] == 'output-dir':
                    args.append(output_dir)
                elif arg_types['param_name'] == 'instance-name':
                    args.append(instance_name)
                elif arg_types['type'] == int:
                    args.append(str(random.randint(arg_types['min'], arg_types['max'])))
                elif arg_types['type'] == float:
                    args.append(str(random.uniform(arg_types['min'], arg_types['max'])))
                elif arg_types['type'] == str:
                    args.append(arg_types['value'])
            # CONSTRAINTS
            if domain == 'academic_advising_prob':
                # FOR TRAINING
                num_courses_per_level_idx = 3
                num_levels_idx = 2
                num_courses = int(args[num_courses_per_level_idx]) * int(args[num_levels_idx])
                if num_courses >= 20:
                    print(f"Courses too high({num_courses})! Retrying...")
                else:
                    break
            else:
                break
        self.print_args(domain, args)
        args = " ".join(args)
        command = " ".join([command, args])
        
        os.chdir(self.generator_dir)
        
        out, err = subprocess.DEVNULL, subprocess.DEVNULL
        if verbose:
            out, err = None, None
        
        try:
            process = subprocess.Popen(command, shell=True, stdout=out, stderr=err)
            exitcode = process.wait()
        except:
            exitcode = -1
        finally:
            os.chdir(curr_dir)

        return exitcode