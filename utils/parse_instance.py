import copy
import os, pdb
import re
import numpy as np
import sys
import functools
from functools import partial
from copy import deepcopy
sys.path.append("../multi_train/deep_plan/")
import my_config
from tabulate import tabulate
import json

class InstanceParser(object):
    def __init__(self, domain, instance):
        self.domain = domain
        self.instance = instance
        curr_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.domain_file = os.path.abspath(os.path.join(curr_dir_path,"../rddl/domains/{}_mdp.rddl".format(self.domain)))
        self.instance_file = os.path.abspath(os.path.join(curr_dir_path, "../rddl/domains/{}_inst_mdp__{}.rddl".format(self.domain, self.instance.replace('.', '_'))))
        self.parsed_instance_file = os.path.abspath(os.path.join(curr_dir_path, "../rddl/parsed/{}_inst_mdp__{}".format(self.domain, self.instance.replace('.', '_'))))
        self.dot_file = os.path.abspath(os.path.join(curr_dir_path, "../rddl/dbn/{}_inst_mdp__{}.dot".format(self.domain, self.instance.replace('.', '_'))))
        # Read domain description
        try:
            with open(self.domain_file) as f:
                self.domain_file_str = f.readlines()
        except UnicodeDecodeError as e:
            with open(self.domain_file, encoding="ISO-8859-1") as f:
                self.domain_file_str = f.readlines()
        # Read instance description
        with open(self.instance_file) as f:
            self.instance_file_str = f.read()
        # Contains information about action preconditions, and hashing
        with open(self.parsed_instance_file) as f:
            self.parsed_instance_file_str = f.read()
        # DBN instance
        with open(self.dot_file) as f:
            self.dot_instance_file_str = f.read()
        # parameters from dbn
        self.color = {  # Refer to the paper for an example of a dbn
            'initial_state': 'lightblue',  # Initial State visualized in this color in dbn
            'final_state': 'gold1',  # Final State Visualized in this color in the dbn
            'action': 'olivedrab1'  # Action visualized in this color in the dbn
        }

        self.object_names = set()  # contains name of nodes from dbn
        self.para_state_names = set()  # contains name of state templates
        self.para_action_names = set()  # contains name of action templates
        self.state_object_names = set()  # contains name of nodes(states) form dbn
        self.para_state_of_objects = {}  # contains fluent state template for an object
        self.action_object_names = set()  # contains name of nodes(actions) form dbn
        self.para_action_of_objects = {}  # contains action template for an object
        self.para_state_connections = set()  # contains tuple of (state object,state object) connections in objects
        self.para_action_connections = set()  # contains tuple of (state object,action object) connections in objects
        self.unpara_action_connections = set()
        self.unpara_state_names = set()  # contains name of state templates (unparameterized)
        self.unpara_action_names = set()  # contains name of action templates (unparameterized)
        self.unpara_fluents = set() # contains the name of unparameterised state fluents

        self.parse_dot_file()
        self.formulae = {}
        self.zero_nf_names = []  # Non-fluents with zero parameters
        self.unary_nf_names = []  # Non-fluents with one parameters
        self.multiple_nf_names = []
        self.objects_in_domain_file = []  # types of objects in domain file
        self.object_name_to_type = {}  # object -> type
        self.type_of_nodes_in_graph = set()  # types of state nodes
        self.para_state_of_objects_nf_values = {}  # contains vector of nf values for nodes
        self.para_state_of_objects_nf_values_names = {}  # ?
        self.transition_unpara_nf_values = []
        self.reward_unpara_nf_values = []
        self.unpara_nf_values = []
        self.para_state_of_objects_numeric_nf_values = None
        self.para_state_of_objects_numeric_nf_values_names = set()
        for k in self.para_state_of_objects.keys():
            self.para_state_of_objects_nf_values[k] = []  #
            self.para_state_of_objects_nf_values_names[k] = []

        self.remove_dbn = my_config.remove_dbn
        self.unary_nf_names_gnd = set()
        self.multiple_nf_names_gnd = set()
        self.objects_in_instance_file_gnd = None
        self.max_arity = 0
        self.action_affects = {}
        self.parse_domain_file()
        self.objects_in_instance_file_gnd = set(self.object_name_to_type.keys())
        self.parse_instance_file()
        self.build_graph()
        self.objects_in_instance_file_to_num_gnd = {i:self.node_dict[i] for i in sorted(self.objects_in_instance_file_gnd)}

        # Node type encoding
        self.type_encoding = []
        for i in range(len(self.node_dict)):
            self.type_encoding.append([0 for _ in range(len(self.type_of_nodes_in_graph))])
        for key, item in self.node_dict.items():
            pr = key.split(",")
            sr = ""
            for c in pr:
                sr += self.object_name_to_type[c] +","
            sr = sr[:-1]
            for k, t in enumerate(sorted(self.type_of_nodes_in_graph)):
                if t == sr:
                    self.type_encoding[item][k] = 1
                else:
                    self.type_encoding[item][k] = 0

    def parse_instance_file(self):
        psr = self.parsed_instance_file_str[
            self.parsed_instance_file_str.find("#####TASK##### Here") +
            len("#####TASK##### Here"):self.parsed_instance_file_str.find(
                "#####ACTION FLUENTS#####")].strip().split(
            "## initial state\n")[1]
        psr = psr.strip().split("\n")
        self.initial_state = list(map(int, psr[0].strip().split(' '))) # Extract the initial state of the variables from the instance
        self.action_to_num = {} # mapping form action to numbers
        action_str = self.parsed_instance_file_str[
            self.parsed_instance_file_str.find("#####ACTION FLUENTS#####") +
            len("#####ACTION FLUENTS#####"):self.parsed_instance_file_str.find(
                "#####DET STATE FLUENTS AND CPFS#####")].strip().split(
                    "## index\n")[1:]
        action_str = [i.strip().split('\n') for i in action_str] # Extract information about all actions from the instance file
        for ac in action_str:
            self.action_to_num[ac[2].replace(' ', '')] = int(ac[0]) + 1 # Constrct the mapping ({'move-east': 1, 'move-north': 2, 'move-south': 3, 'move-west': 4})
        self.num_to_action = {v: k for k, v in self.action_to_num.items()} # Inverse dict
        self.state_to_num = {} # mapping form state to numbers
        det_str = self.parsed_instance_file_str[
                self.parsed_instance_file_str.find(
                    "#####DET STATE FLUENTS AND CPFS#####") +
                len("#####DET STATE FLUENTS AND CPFS#####"):
                self.parsed_instance_file_str.find(
                    "#####PROB STATE FLUENTS AND CPFS#####")].strip().split(
            "## index\n")[1:]
        self.det_str = [i.strip().split('\n') for i in det_str]  # Deterministic state fluents
        for ac in self.det_str:
            self.state_to_num[ac[2].replace(' ', '')] = int(ac[11])  # Construct the mapping from states to nums
        self.prob_str = self.parsed_instance_file_str[
                        self.parsed_instance_file_str.find(
                            "#####PROB STATE FLUENTS AND CPFS#####") +
                        len("#####PROB STATE FLUENTS AND CPFS#####"):
                        self.parsed_instance_file_str.find("#####REWARD#####")].strip().split(
            "## index\n")[1:]
        self.prob_str = [i.strip().split('\n') for i in self.prob_str]  # Probability Transitions
        for ac in self.prob_str:
            self.state_to_num[ac[2].replace(' ', '')] = int(ac[11])
        self.num_to_state = {v: k for k, v in self.state_to_num.items()}  # Inverse mapping

        # Getting reward formula
        self.reward_str = self.parsed_instance_file_str[
                        self.parsed_instance_file_str.find(
                            "#####REWARD#####\n## formula") +
                        len("#####REWARD#####\n## formula"):
                        self.parsed_instance_file_str.find("## min\n")].strip()

        self.reward_min = float(self.parsed_instance_file_str[
                        self.parsed_instance_file_str.find(
                            "## min\n") +
                        len("## min\n"):
                        self.parsed_instance_file_str.find("## max\n")].strip())

        self.reward_max = float(self.parsed_instance_file_str[
                        self.parsed_instance_file_str.find(
                            "## max\n") +
                        len("## max\n"):
                        self.parsed_instance_file_str.find("## independent from actions\n")].strip())
        return

    def parse_domain_file(self):  # Parse the domain file
        for line in self.domain_file_str:  # For all lines in the domain definition file
            line = line.strip()
            line = line[:line.find('//')].strip()
            if 'object' in line:  # Append possible objects in a list
                self.objects_in_domain_file.append(line[:line.find(':')].strip())
            elif 'non-fluent' in line:
                values = line[line.find('{') + 1:line.find('}')].split(',')  # Characteristics of the non-fluent
                name = line[:line.find(':')].strip()  # Name of the non-fluent (CONNECTED, REBOOT PROB)
                default = values[-1][values[-1].find("=") + 1:].strip()  # Default Value of the non-fluent

                # Vishal Start: To count max arity of any non-fluent
                if name.count("(") >= 1:
                    cur_arity = name.count(",") + 1
                    if self.max_arity < cur_arity:
                        self.max_arity = cur_arity
                # Vishal End
                type_val = None
                if values[-2].strip() == 'real':
                    default = float(default)
                    type_val = float
                elif values[-2].strip() == 'bool':
                    if default == 'false':
                        default = 0
                    else:
                        default = 1
                    type_val = bool
                if name.count(',') >= 1:  # If the non-fluent depends on more than two parameters
                    parameters = name[name.find('(') + 1:name.find(')')].strip().replace(' ', '')
                    self.multiple_nf_names.append((name[:name.find('(')].strip(), type_val, default, parameters))
                elif name.count('(') >= 1:  # If the non-fluent is unary
                    parameters = name[name.find('(') + 1:name.find(')')].strip().replace(' ', '')
                    self.unary_nf_names.append((name[:name.find('(')].strip(), type_val, default, parameters))
                else:  # If the non-fluent is non parametrized
                    self.zero_nf_names.append((name.strip(), type_val, default, None))
            elif 'action-fluent' in line:
                pass
            elif 'state-fluent' in line:
                name = line[:line.find(':')].strip()  # Name of the state-fluent (CONNECTED, REBOOT PROB)
                if name.count("(") >= 1:
                    cur_arity = name.count(",") + 1
                    if self.max_arity < cur_arity:
                        self.max_arity = cur_arity

        for ob_type in self.objects_in_domain_file:  # Object types (Computer in sysadmin)
            c = re.findall('{0} : {{.*?}}'.format(ob_type), self.instance_file_str)[0]
            temp = c[c.find("{") + 1:c.find("}")].split(',')
            temp = [i.strip() for i in temp]  # Find all objects of that type
            for ob_name in temp:  # Add them to this dict
                self.object_name_to_type[ob_name] = ob_type

            # Added by Vishal
            self.type_of_nodes_in_graph.add(ob_type)

        for ob_name in self.object_names:  # Types of nodes in graph (Say an object is a computer and bird, and one object is computer - two types of nodes in the graph)
            temp = ob_name.split(',')
            temp = [self.object_name_to_type[i] for i in temp]
            self.type_of_nodes_in_graph.add(','.join(temp))

        # Added by Vishal
        for nf in self.multiple_nf_names:
            self.type_of_nodes_in_graph.add(nf[-1])


        self.domain_file_str = functools.reduce(lambda a, b: a + b, self.domain_file_str)
        cpfs_index = self.domain_file_str.find("cpfs")
        cpfs_end = self.domain_file_str[cpfs_index:].find("};") + cpfs_index
        cpfs = self.domain_file_str[cpfs_index:cpfs_end]
        reward = self.domain_file_str[cpfs_end:]
        for nf in sorted(self.zero_nf_names):
            pr = re.findall('{} = [-+]?[0-9]*\.?[0-9]+'.format(nf[0]), self.instance_file_str)
            if nf[0] not in cpfs:
                if len(pr) == 0:
                    self.reward_unpara_nf_values.append(float(nf[2]))
                    self.unpara_nf_values.append(float(nf[2]))
                else:
                    self.reward_unpara_nf_values.append(float(pr[0].split()[2]))
                    self.unpara_nf_values.append(float(pr[0].split()[2]))
            elif nf[0] not in reward:
                if len(pr) == 0:
                    self.transition_unpara_nf_values.append(float(nf[2]))
                    self.unpara_nf_values.append(float(nf[2]))
                else:
                    self.transition_unpara_nf_values.append(float(pr[0].split()[2]))
                    self.unpara_nf_values.append(float(pr[0].split()[2]))
            else:
                if len(pr) == 0:
                    self.transition_unpara_nf_values.append(float(nf[2]))
                    self.reward_unpara_nf_values.append(float(nf[2]))
                    self.unpara_nf_values.append(float(pr[0].split()[2]))
                else:
                    self.transition_unpara_nf_values.append(float(pr[0].split()[2]))
                    self.reward_unpara_nf_values.append(float(pr[0].split()[2]))
                    self.unpara_nf_values.append(float(pr[0].split()[2]))
        del cpfs, reward

        for k, nf in enumerate(sorted(self.unary_nf_names)):  # For all unary non-fluents
            temp_flag = False
            for key in self.type_of_nodes_in_graph:  # Non-fluent depends on an object which is present in the types of nodes (Always true??)
                if nf[-1] in key:
                    temp_flag = True
            if not temp_flag:
                continue
            if nf[1] is not bool:  # Only for boolean non-fluents
                continue
            pr = re.findall('{}\(.*?;'.format(nf[0]), self.instance_file_str) # Check what object has this boolean non-fluent set to true

            for key in self.para_state_of_objects_nf_values.keys():  # Add this non-fluent to each objects nf values (Boolean nf so for all objects)
                self.para_state_of_objects_nf_values[key].append(nf[2])
            for key in self.para_state_of_objects_nf_values_names.keys():
                self.para_state_of_objects_nf_values_names[key].append(nf[0])

            for c in pr:
                node = c[c.find("(") + 1:c.find(")")]
                node = node.strip().replace(' ', '')  # Node is the object for which this boolean is set to true
                self.unary_nf_names_gnd.add(node)
                for key in self.para_state_of_objects_nf_values.keys():
                    if node in key:  # If object in some node for graph
                        self.para_state_of_objects_nf_values[key][-1] = 1 - nf[2]  # Set it to true

        for k, nf in enumerate(sorted(self.multiple_nf_names)):  # For all multiple non fluents (Repeat same thing as above)
            if nf[-1] not in self.type_of_nodes_in_graph:
                continue
            if nf[1] is not bool:
                continue
            pr = re.findall('{}\(.*?;'.format(nf[0]), self.instance_file_str)
            for key in self.para_state_of_objects_nf_values.keys():
                self.para_state_of_objects_nf_values[key].append(nf[2])
            for key in self.para_state_of_objects_nf_values_names.keys():
                self.para_state_of_objects_nf_values_names[key].append(nf[0])
            for c in pr:
                node = c[c.find("(") + 1:c.find(")")]
                node = node.strip().replace(' ', '')
                self.multiple_nf_names_gnd.add(node)
                if node in self.para_state_of_objects_nf_values.keys():
                    self.para_state_of_objects_nf_values[node][-1] = 1 - nf[2]
        return

    def parse_dot_file(self):  # Parse graph visualization dot files
        file_str = self.dot_instance_file_str.strip().split(';')
        for k, line in enumerate(file_str):  # For all lines in file_str
            if self.color['final_state'] in line:  # Identify line as defining final state
                c = re.findall('\".*?\"', line)[0][1:-1]
                if '$' in c:
                    obj = c[c.find('$'):c.find(')')].replace('$', '').replace(' ',
                                                                            '')  # Object associated with that node (c1,c2 in sysadmin) ((x1,x2) in navigation)
                    state_var = c[:c.find('\'')].replace('$',
                                                        '')  # The predicate variable (running in sysadmin, robot-at in navigation)
                    self.object_names.add(obj)
                    self.para_state_names.add(state_var)
                    if obj not in self.para_state_of_objects:
                        self.para_state_of_objects[obj] = set({state_var})  # Set state of say c3 to running (sysadmin)
                    else:
                        self.para_state_of_objects[obj].add(
                            state_var)  # Add all possible variables over c3 (running, on, off)
                else:
                    self.unpara_state_names.add(c.replace('\'', ''))  # Unparametrized state variables (Which domains?)
            elif self.color['action'] in line:  # Identify line as defining an action
                c = line[:line.find('[color')].strip().replace('\"', '')
                if '$' in c:
                    obj = c[c.find('$'):c.find(')')].replace('$', '').replace(' ',
                                                                            '')  # Variable over which action is parametrized (c2 in sysadmin)
                    action_var = c[:c.find('(')].replace('$', '')  # Action var (reboot in sysadmin)
                    self.object_names.add(obj)
                    self.para_action_names.add(action_var)
                    if obj not in self.para_action_of_objects:
                        self.para_action_of_objects[obj] = set({action_var})  # Set action of say c3 to reboot
                    else:
                        self.para_action_of_objects[obj].add(action_var)
                else:
                    self.unpara_action_names.add(c.replace('$',
                                                        ''))  # Unparametrized Action variables (For instance a reset action which resets the whole instance)
            elif '->' in line:  # Defines a edge in the DBN (In the bipartite graph)
                sp = line.split('->')
                f = sp[0].strip()
                t = sp[1].strip()

                from_var = f[f.find('\"') + 1:f.find('(')].replace('$',
                                                                '')  # State of the node from which edge starts (running)
                from_obj = ""
                if f.find('(') != -1:
                    from_obj = f[f.find('(') + 1:f.find(')')].replace('$', '').replace(' ',
                                                                                    '')  # Parameteers from which edge starts (c1)
                to_var = t[t.find('\"') + 1:t.find('(')].replace('$', '').replace('\'', '')
                to_obj = ""
                if t.find('(') != -1:
                    to_obj = t[t.find('(') + 1:t.find(')')].replace('$', '').replace(' ', '')
                else:
                    continue

                if from_obj == "":
                    if from_var in self.unpara_action_names:
                        self.unpara_action_connections.add((from_var, to_obj))
                    continue
                if from_var in self.para_state_names:
                    self.para_state_connections.add((from_obj, to_obj))  # Add connections between two states
                elif from_var in self.para_action_names:
                    self.para_action_connections.add(
                        (to_obj, from_obj, from_var))  # Add connections between the action to the state

        self.state_object_names = set(self.para_state_of_objects.keys())  # Obvious
        self.action_object_names = set(self.para_action_of_objects.keys())  # Obvious

    def build_graph(self):
        # 1. Create nodes of the instance graph
        self.node_dict = {}  # Node in the dbn to index dict
        self.extended_node_dict = {}  # Extended node_dict
        for k, obj in enumerate(sorted(self.state_object_names)):
            self.node_dict[obj] = k
            self.extended_node_dict[obj] = k
        for k, obj in enumerate(sorted(self.action_object_names - self.state_object_names)):
            self.extended_node_dict[obj] = k + len(self.state_object_names)
        self.fluent_nodes = list(range(len(self.node_dict)))
        # Vishal Start: Add new Nodes for non-fluents and gnd objects
        offset = len(self.node_dict)
        k = 0
        for obj in sorted(self.objects_in_instance_file_gnd):
            if obj not in self.node_dict:
                self.node_dict[obj] = k + offset
                k += 1
        # No need to add unary as individual objects
        offset = len(self.node_dict)
        k = 0
        for obj in sorted(self.multiple_nf_names_gnd):
            if obj not in self.node_dict:
                self.node_dict[obj] = k + offset
                k += 1

        for key in sorted(self.node_dict.keys()):
            self.para_state_of_objects_nf_values[key] = []
            self.para_state_of_objects_nf_values_names[key] = []
        for k, nf in enumerate(sorted(self.unary_nf_names)):  # For all unary non-fluents
            temp_flag = False
            for key in self.type_of_nodes_in_graph:  # Non-fluent depends on an object which is present in the types of nodes (Always true??)
                if nf[-1] in key:
                    temp_flag = True
            if not temp_flag:
                continue
            if nf[1] is bool:  # Only for boolean non-fluents
                pr = re.findall('{}\(.*?;'.format(nf[0]), self.instance_file_str)  # Check what object has this boolean non-fluent set to true

                for key in self.para_state_of_objects_nf_values.keys():  # Add this non-fluent to each objects nf values (Boolean nf so for all objects)
                    self.para_state_of_objects_nf_values[key].append(nf[2])
                for key in self.para_state_of_objects_nf_values_names.keys():
                    self.para_state_of_objects_nf_values_names[key].append(nf[0])

                for c in pr:
                    node = c[c.find("(") + 1:c.find(")")]
                    self.para_state_of_objects_nf_values[node][-1] = 1 - nf[2]  # Set it to true
            else:
                pr = re.findall('{}\(.*\) = [-+]?[0-9]*\.?[0-9]+'.format(nf[0]), self.instance_file_str)

                for key in self.para_state_of_objects_nf_values.keys():  # Add this non-fluent to each objects nf values (Boolean nf so for all objects)
                    self.para_state_of_objects_nf_values[key].append(nf[2])
                for key in self.para_state_of_objects_nf_values_names.keys():
                    self.para_state_of_objects_nf_values_names[key].append(nf[0])

                for c in pr:
                    node = c[c.find("(") + 1:c.find(")")]
                    self.para_state_of_objects_nf_values[node][-1] = float(c.split("=")[-1])

        for k, nf in enumerate(sorted(self.multiple_nf_names)):  # For all multiple non fluents (Repeat same thing as above)
            if nf[-1] not in self.type_of_nodes_in_graph:
                print("Parse_instance.build_graph: Should not happen: all keys should be there")
                exit(-1)
            if nf[1] is bool:
                pr = re.findall('{}\(.*?;'.format(nf[0]), self.instance_file_str)
                for key in self.para_state_of_objects_nf_values.keys():
                    self.para_state_of_objects_nf_values[key].append(nf[2])
                for key in self.para_state_of_objects_nf_values_names.keys():
                    self.para_state_of_objects_nf_values_names[key].append(nf[0])
                for c in pr:
                    node = c[c.find("(") + 1:c.find(")")].replace(" ", "")
                    self.para_state_of_objects_nf_values[node][-1] = 1 - nf[2]
            else:
                pr = re.findall('{}\(.*?;'.format(nf[0]), self.instance_file_str)
                for key in self.para_state_of_objects_nf_values.keys():
                    self.para_state_of_objects_nf_values[key].append(nf[2])
                for key in self.para_state_of_objects_nf_values_names.keys():
                    self.para_state_of_objects_nf_values_names[key].append(nf[0])
                for c in pr:
                    node = c[c.find("(") + 1:c.find(")")]
                    self.para_state_of_objects_nf_values[node][-1] = float(c.split("=")[-1].strip(";"))
        # Vishal End: Add non-fluent values again due to adding new nodes.


        self.num_graph_action = 1  # The no operation action (directly acts on the graph)
        self.num_parameter_actions = 0
        self.nf_features = [None] * len(self.node_dict)  # NF Feaures for each node
        # for key, item in self.para_state_of_objects_nf_values.items():
        for key, item in self.node_dict.items():
            self.nf_features[item] = self.para_state_of_objects_nf_values[key]  # Assign features for each node
        # Add unparameterized non fluents to all nodes
        for i in range(len(self.unpara_nf_values)):
            for j in range(len(self.nf_features)):
                self.nf_features[j].append(self.unpara_nf_values[i])

        self.fluent_state_dict = {}
        for i, sn in enumerate(sorted(self.para_state_names)):
            self.fluent_state_dict[sn] = i  # State to index
        self.graph_f_features = []
        try:
            self.nonfluent_feature_dims = len(self.nf_features[0])
        except IndexError as e:
            self.nonfluent_feature_dims = 0
        self.num_graph_action = len(self.unpara_action_names) + 1  # No-operation + unparametrized actions
        self.action_template_to_num = {}  # Action template to number mapping
        for k, tp in enumerate(sorted(self.unpara_action_names)):
            self.action_template_to_num[tp] = k + 1
        for k, tp in enumerate(sorted(self.para_action_names)):
            self.action_template_to_num[tp] = k + self.num_graph_action

        self.num_to_action_template = {}
        for k, tp in enumerate(sorted(self.unpara_action_names)):
            self.num_to_action_template[k+1] = tp
        for k, tp in enumerate(sorted(self.para_action_names)):
            self.num_to_action_template[k+self.num_graph_action] = tp

        self.detailed_action = {}

        # To see if any of the action of this type affects something
        self.action_affects = {self.action_template_to_num[i]:False for i in self.action_template_to_num.keys()}
        
        for st, ob, ac in sorted(self.para_action_connections):  # ToObj, FromObj,FromVar
            try:
                action_num = self.action_to_num[ac + '(' + ob + ')']
            except KeyError as e:
                continue
            template_num = self.action_template_to_num[ac]
            node_num = self.node_dict[st]
            self.action_affects[template_num] = True
            if not my_config.remove_dbn:
                if action_num not in self.detailed_action.keys():
                    self.detailed_action[action_num] = (template_num, set([node_num]), [])  # Details about that action (Action_num, what template, which node it infolunces)
                else:
                    self.detailed_action[action_num][1].add(node_num)
            else:
                if action_num not in self.detailed_action.keys():
                    self.detailed_action[action_num] = (template_num, set(), [])

            if len(self.detailed_action[action_num][2]) == 0:
                for o in ob.split(","):
                    self.detailed_action[action_num][2].append(self.node_dict[o])
        self.detailed_action[0] = (0, set(), [])  # Unparameter action details
        for k in sorted(self.action_to_num.keys()):
            action_num = self.action_to_num[k]
            if action_num not in self.detailed_action.keys():  # This action doesnt influnce any node
                self.detailed_action[action_num] = (self.action_template_to_num[k.split("(")[0]], set(), [])
                if "(" in k:
                    for o in k.split("(")[1].split(")")[0].split(","):
                        self.detailed_action[action_num][2].append(self.node_dict[o])
                else:
                    self.detailed_action[action_num] = (self.action_template_to_num[k], set(), [])
        # Counting the number of types of actions
        self.num_types_action = -1
        for action_template, _, _ in self.detailed_action.values():
            self.num_types_action = max(self.num_types_action, action_template)
        self.num_types_action += 1

        # Building adjacency lists
        self.build_adjacency_lists()
        if my_config.fc_adjacency:
            self.build_fc_adjacency_lists()

        self.extended_detailed_action = {}
        for k in sorted(self.action_to_num.keys()):
            if k.find('(') == -1:  # Unpara action
                self.extended_detailed_action[self.action_to_num[k]] = (
                self.action_template_to_num[k], -1, -1)  # Unpara actions
                continue
            action_num = self.action_to_num[k]
            action_template = k.split('(')[0]
            action_obj = k.split('(')[1].split(')')[0]
            if action_obj in self.node_dict:
                self.extended_detailed_action[action_num] = (
                self.action_template_to_num[action_template], self.extended_node_dict[action_obj],
                self.node_dict[action_obj])
            else:
                self.extended_detailed_action[action_num] = (
                self.action_template_to_num[action_template], self.extended_node_dict[action_obj], -1)
        self.extended_detailed_action[0] = (0, -1)

        print("Graph built")

    def get_num_to_action(self):
        return self.num_to_action

    def build_fc_adjacency_lists(self):
        # pdb.set_trace()
        pass

    def build_adjacency_lists(self):
        extra_adj = 1
        if my_config.add_separate_adj:
            # Last 2 are extra adj. -2 is for DBN edges and -1 for (x,y) to x and y each
            extra_adj = 3
        if my_config.add_edge_type:
            extra_adj += self.max_arity
        self.adjacency_lists = [{} for _ in range(len(
            self.action_template_to_num.keys()) + extra_adj)]  # the kth adjacency list defines adjacency due to kth action template (Into different decoders?)
        self.num_nodes = len(self.node_dict)
        if not my_config.remove_dbn:
            for a, b in sorted(self.para_state_connections):
                if self.node_dict[a] not in self.adjacency_lists[0].keys():  # Add directed edges to the
                    self.adjacency_lists[0][self.node_dict[a]] = [self.node_dict[b]]
                else:
                    self.adjacency_lists[0][self.node_dict[a]].append(self.node_dict[b])

        # Keep DBN edges separately in -2 adj
        if my_config.add_separate_adj:
            for k in self.adjacency_lists[0].keys():
                self.adjacency_lists[-2][k] = self.adjacency_lists[0][k]

        # Vishal Start: Add new edges between new nodes for non-fluents and gnd objects
        if my_config.merged_model:
            for k, obj in enumerate(sorted(self.node_dict.keys())):
                for next_obj in obj.split(","):
                    if self.node_dict[obj] not in self.adjacency_lists[0]:
                        self.adjacency_lists[0][self.node_dict[obj]] = []

                    # if self.node_dict[next_obj] not in self.adjacency_lists[0]:
                    # 	self.adjacency_lists[0][self.node_dict[next_obj]] = [self.node_dict[obj]]
        else:
            for k, obj in enumerate(sorted(self.node_dict.keys())):
                for next_obj in obj.split(","):
                    if self.node_dict[obj] not in self.adjacency_lists[0]:
                        self.adjacency_lists[0][self.node_dict[obj]] = [self.node_dict[next_obj]]
                        # if my_config.add_separate_adj:
                        # 	self.adjacency_lists[-2][self.node_dict[obj]] = []
                    else:
                        if self.node_dict[next_obj] not in self.adjacency_lists[0][self.node_dict[obj]]:
                            self.adjacency_lists[0][self.node_dict[obj]].append(self.node_dict[next_obj])

                    if self.node_dict[next_obj] not in self.adjacency_lists[0]:
                        self.adjacency_lists[0][self.node_dict[next_obj]] = [self.node_dict[obj]]
                        # if my_config.add_separate_adj:
                        # 	self.adjacency_lists[-2][self.node_dict[obj]] = []
                    else:
                        if self.node_dict[obj] not in self.adjacency_lists[0][self.node_dict[next_obj]]:
                            self.adjacency_lists[0][self.node_dict[next_obj]].append(self.node_dict[obj])

        # For the case where we want to add edge types
        if my_config.add_edge_type:
            offset = len(self.adjacency_lists) - self.max_arity
            # Copy keys
            for zz in range(offset, len(self.adjacency_lists)):
                self.adjacency_lists[zz] = {k: [] for k in self.adjacency_lists[0].keys()}


            for k, obj in enumerate(sorted(self.node_dict.keys())):
                splits = obj.split(",")
                for zz in range(len(splits)):
                    next_obj = splits[zz]
                    if self.node_dict[obj] not in self.adjacency_lists[offset+zz]:
                        self.adjacency_lists[offset+zz][self.node_dict[obj]] = [self.node_dict[next_obj]]
                    else:
                        if self.node_dict[next_obj] not in self.adjacency_lists[offset+zz][self.node_dict[obj]]:
                            self.adjacency_lists[offset+zz][self.node_dict[obj]].append(self.node_dict[next_obj])

                    if self.node_dict[next_obj] not in self.adjacency_lists[offset+zz]:
                        self.adjacency_lists[offset+zz][self.node_dict[next_obj]] = [self.node_dict[obj]]
                    else:
                        if self.node_dict[obj] not in self.adjacency_lists[offset+zz][self.node_dict[next_obj]]:
                            self.adjacency_lists[offset+zz][self.node_dict[next_obj]].append(self.node_dict[obj])

                    
        if my_config.add_separate_adj:
            for k in self.adjacency_lists[0].keys():
                if k not in self.adjacency_lists[-2]:
                    self.adjacency_lists[-2][k] = []

        # Exact same loop as above but for last adj i.e. for (x,) to x and y each.
        if my_config.add_separate_adj:
            for k, obj in enumerate(sorted(self.node_dict.keys())):
                for next_obj in obj.split(","):
                    if self.node_dict[obj] not in self.adjacency_lists[-1]:
                        self.adjacency_lists[-1][self.node_dict[obj]] = [self.node_dict[next_obj]]
                    else:
                        if self.node_dict[next_obj] not in self.adjacency_lists[-1][self.node_dict[obj]]:
                            self.adjacency_lists[-1][self.node_dict[obj]].append(self.node_dict[next_obj])

                    if self.node_dict[next_obj] not in self.adjacency_lists[-1]:
                        self.adjacency_lists[-1][self.node_dict[next_obj]] = [self.node_dict[obj]]
                    else:
                        if self.node_dict[obj] not in self.adjacency_lists[-1][self.node_dict[next_obj]]:
                            self.adjacency_lists[-1][self.node_dict[next_obj]].append(self.node_dict[obj])


        for i in range(1, len(self.action_template_to_num.keys()) + 1):
            self.adjacency_lists[i] = {k: [] for k in self.adjacency_lists[0].keys()}

        self.extended_adjacency_lists = deepcopy(self.adjacency_lists)
        for strrr in [self.det_str, self.prob_str]:  # For state fluents in deterministic and probablistic
            for ac in strrr:
                state_var = ac[2].replace(' ', '')
                state_var_ob = state_var.split('(')[-1].replace('(', '').replace(')', '')
                formula = ac[9].strip()

                # If switch is not in formula add brackets
                flag = 0
                self.parse_formula(formula, state_var, flag)
                if 'switch' in formula:
                    formula = formula[7:-1].strip()
                    flag = 1

                brackets = []
                i = 0
                j = 0

                while (i < len(formula)):
                    j = i
                    if formula[i] == '(':
                        count = 0
                        while (j < len(formula)):
                            if (formula[j] == '('):
                                count += 1
                            elif formula[j] == ')':
                                count -= 1
                            if count == 0:
                                break
                            j += 1
                        if formula[j] == ')':
                            brackets.append(formula[i:j + 1])
                        i = j + 1
                    else:
                        i += 1
                dependencies = [re.findall('\$a\(\d+\).*?\$s\(\d+\)', bac) for bac in brackets if '$c(0)' not in bac]
                for ininin in dependencies:
                    for dep in ininin:
                        try:
                            ac_num, st_num = tuple(map(int, re.findall('\d+', dep)))
                        except ValueError:
                            continue
                        ac_num += 1
                        ac = self.num_to_action[ac_num].replace(' ', '')
                        st = self.num_to_state[st_num].replace(' ', '')
                        ac_temp = self.action_template_to_num[ac.split('(')[0]]
                        try:
                            st_ob = re.findall('\(.*?\)', st)[0][1:-1]
                        except IndexError:
                            continue
                        try:
                            if self.node_dict[state_var_ob] not in self.adjacency_lists[ac_temp]:
                                self.adjacency_lists[ac_temp][self.node_dict[state_var_ob]] = [self.node_dict[st_ob]]
                            else:
                                if self.node_dict[st_ob] not in self.adjacency_lists[ac_temp][
                                    self.node_dict[state_var_ob]]:
                                    self.adjacency_lists[ac_temp][self.node_dict[state_var_ob]].append(
                                        self.node_dict[st_ob])
                        except KeyError:
                            # Adding unparameterised state fluents
                            self.unpara_fluents.add(state_var_ob)
                            continue
        
        if my_config.split_dbn:
            dbn_edge_types = []
            for conn in self.para_state_connections:
                start, end = conn
                start_type = tuple([self.object_name_to_type[x] for x in start.split(",")])
                end_type = tuple([self.object_name_to_type[x] for x in end.split(",")])
                if (start_type, end_type) not in dbn_edge_types:
                    dbn_edge_types.append((start_type, end_type))
            dbn_edge_types = sorted(dbn_edge_types)
            self.dbn_edge_types_to_idx = {}
            for i, x in enumerate(dbn_edge_types):
                self.dbn_edge_types_to_idx[x] = i
            print("DBN Edge types:", self.dbn_edge_types_to_idx)            
            original_num_adj = len(self.adjacency_lists)
            for _ in range(len(self.dbn_edge_types_to_idx)):
                self.adjacency_lists.append({k: [] for k in self.adjacency_lists[0].keys()})

            for conn in self.para_state_connections:
                start, end = conn
                start_type = tuple([self.object_name_to_type[x] for x in start.split(",")])
                end_type = tuple([self.object_name_to_type[x] for x in end.split(",")])
                offset = self.dbn_edge_types_to_idx[(start_type, end_type)]
                self.adjacency_lists[original_num_adj+offset][self.node_dict[start]].append(self.node_dict[end])
                
            

        self.fluent_feature_dims = len(self.para_state_names) + len(self.unpara_fluents)

        for i in range(1, len(self.action_template_to_num.keys()) + 1):
            self.extended_adjacency_lists[i] = {k: [] for k in range(len(self.object_names))}
        for i in range(len(self.object_names)):
            if i not in self.extended_adjacency_lists[0].keys():
                self.extended_adjacency_lists[0][i] = []

        for a, b, c in sorted(self.para_action_connections):
            if self.extended_node_dict[b] not in self.extended_adjacency_lists[self.action_template_to_num[c]].keys():
                self.extended_adjacency_lists[self.action_template_to_num[c]][self.extended_node_dict[b]] = [
                    self.extended_node_dict[a]]
            else:
                self.extended_adjacency_lists[self.action_template_to_num[c]][self.extended_node_dict[b]].append(
                    self.extended_node_dict[a])

        if my_config.merged_model:
            pass
            # self.adjacency_lists = [self.adjacency_lists[0]] + self.adjacency_lists[-self.max_arity:]
        elif my_config.remove_dbn and my_config.add_edge_type:
            self.adjacency_lists = self.adjacency_lists[-self.max_arity:]
        elif my_config.use_only_first_adj:
            self.adjacency_lists = [self.adjacency_lists[0]]

    def get_adjacency_list(self):
        return self.adjacency_lists

    def get_extended_adjacency_list(self):
        return self.extended_adjacency_lists

    def get_num_adjacency_list(self):
        return len(self.adjacency_lists)

    def get_fluent_features(self, state):  # Given a vector of the state, build feature vector for each fluents
        f_features = np.array([[0 for i in range(self.fluent_feature_dims)] for _ in
                    range(len(self.node_dict))], dtype="float32")  # For all fluents for all possible vertices in the dbn

        if len(self.unpara_fluents) != 0:
            for (i, st) in enumerate(sorted(self.unpara_fluents)):
                f_features[:, -i - 1] = state[self.state_to_num[st]]

        for st in self.para_state_names:  # For each fluent
            for node in self.state_object_names:  # For each parameter of the fluent (a vertex in the graph - rememberd dbn)
                stn = st + '(' + node + ')'
                try:  # Assign features from the mapping of nodes and states to indices
                    f_features[self.node_dict[node]][self.fluent_state_dict[st]] = float(state[self.state_to_num[stn]])
                except KeyError as e:
                    f_features[self.node_dict[node]][self.fluent_state_dict[st]] = 0
                except:
                    pdb.set_trace()
                    print("something is wrong")
                    exit(0)

        return f_features

    def get_numeric_unary_nf_features(self, state):
        return self.para_state_of_objects_numeric_nf_values

    def get_graph_fluent_features(self, state):
        gf = []
        for key, item in self.state_to_num.items():
            if key in self.unpara_state_names:  # If unparametrized fluents , append it to the graph embedding
                gf.append(state[item])
        return gf

    def get_feature_dims(self):
        return self.fluent_feature_dims, self.nonfluent_feature_dims

    def get_num_actions(self):
        return len(list(self.detailed_action.keys()))

    def get_action_details(self):
        return self.detailed_action

    def get_extended_action_details(self):
        return self.extended_detailed_action

    def get_nf_features(self):
        return self.nf_features


    def get_num_action_nodes(self):  # Number of nodes corresponding to state variable tuples
        return len(self.extended_node_dict) - len(self.node_dict)

    def get_num_nodes(self):
        return self.num_nodes

    def get_num_graph_fluents(self):
        return len(list(self.unpara_state_names))

    def get_num_type_actions(self):
        return self.num_types_action

    def get_action_templates(self):
        return self.action_template_to_num.keys()

    def get_attr(self, attr):
        if attr not in self.__dict__.keys():
            raise KeyError("Wrong attr")
        return self.__dict__[attr]

    def get_state_repr(self, state):
        state = np.array(state)
        if self.domain == 'navigation':
            num_x = 0
            num_y = 0
            mapping = []
            for _, k in self.object_name_to_type.items():
                if k == 'xpos':
                    num_x += 1
                elif k == 'ypos':
                    num_y += 1
            r_at = -1
            for k in self.state_object_names:
                x, y = tuple(k.split(','))
                xn = int(x[1:])
                yn = int(y[1:])
                pos = self.state_to_num['robot-at({})'.format(k)]  # Robot At value
                val = state[pos]
                if val == 1:
                    r_at = (xn, yn)
                    # print("r_at", r_at)
                goal = self.para_state_of_objects_nf_values[k][5]  # Goal value
                if goal == 1 and val == 0:
                    val = -1
                    g_at = (xn, yn)
                    # print("g_at", g_at)
                elif goal == 1 and val == 1:
                    val = 2
                    g_at = (xn, yn)
                    # print("g_at", g_at)
                mapping.append((xn, yn, val))
            mapping.sort(key=lambda x: (-x[1], x[0]))
            state_repr = np.array([a[-1] for a in mapping]).reshape((num_y, num_x))
            return np.array_str(state_repr) + "\n"

        elif self.domain == 'crossing_traffic':
            num_x = 0
            num_y = 0
            mapping = []
            for _, k in self.object_name_to_type.items():
                if k == 'xpos':
                    num_x += 1
                elif k == 'ypos':
                    num_y += 1
            for k in self.state_object_names:
                x, y = tuple(k.split(','))
                xn = int(x[1:])
                yn = int(y[1:])
                pos = self.state_to_num['robot-at({})'.format(k)]  # Robot
                val = state[pos]
                try:
                    pos = self.state_to_num['obstacle-at({})'.format(k)]
                    val2 = state[pos]
                    if val2 == 1:
                        if val == 1:
                            val = 3  # Robot collide with obstacle
                        else:
                            val = 2  # Obstacle
                except Exception as _:
                    pass
                if val == 1 or val == 0:
                    goal = self.para_state_of_objects_nf_values[k][-1]  # Goal value
                    if goal == 1 and val == 0:
                        val = 4
                    if goal == 1 and val == 1:
                        val = 5
                mapping.append((xn, yn, val))
            mapping.sort(key=lambda x: (-x[1], x[0]))
            s = np.array([a[-1] for a in mapping]).reshape((num_y, num_x))
            return np.array_str(s) + "\n"

        elif self.domain == 'tamarisk':
            reach = {}
            for node in self.state_object_names:
                tamarisk = state[self.state_to_num['tamarisk-at(' + node + ')']]
                native = state[self.state_to_num['native-at(' + node + ')']]
                if tamarisk == 1 and native == 1:
                    val = "F"
                elif tamarisk == 1:
                    val = "T"
                elif native == 1:
                    val = "N"
                else:
                    val = 0
                reachNum = node[1]
                if reachNum not in reach.keys():
                    reach[reachNum] = dict()
                slotNum = node[3]
                reach[reachNum][slotNum] = val
            toReturn = ""
            for key in sorted(reach.keys()):
                for slot in sorted(reach[key].keys()):
                    toReturn = toReturn + str(reach[key][slot]) + " "
                toReturn = toReturn + "\n"
            return toReturn

        elif self.domain == 'academic_advising':
            state_repr = "Goal : "
            for key, item in self.para_state_of_objects_nf_values.items():
                if item[0] == 1:  # Goal course
                    state_repr = state_repr + key + ", "  # Added Goal Courses
            state_repr = state_repr + "\nTaken :"
            for node in self.state_object_names:
                if state[self.state_to_num['taken(' + node + ')']] == 1:
                    state_repr = state_repr + node + ", "
            state_repr = state_repr + "\nPassed :"
            for node in self.state_object_names:
                if state[self.state_to_num['passed(' + node + ')']] == 1:
                    state_repr = state_repr + node + ", "
            state_repr = state_repr + "\n"
            return state_repr

        elif self.domain == "game_of_life":
            num_x = 0
            num_y = 0
            mapping = []
            for _, k in self.object_name_to_type.items():
                if k == 'x_pos':
                    num_x += 1
                elif k == 'y_pos':
                    num_y += 1
            for k in self.state_object_names:
                x, y = tuple(k.split(','))
                xn = int(x[1:])
                yn = int(y[1:])
                pos = self.state_to_num['alive({})'.format(k)]  # Robot At value
                val = state[pos]
                mapping.append((xn, yn, val))
            mapping.sort(key=lambda x: (x[1], x[0]))
            s = np.array([a[-1] for a in mapping]).reshape((num_y, num_x))
            return tabulate(s, tablefmt='fancy_grid', stralign='center')

        elif self.domain == "sysadmin":
            count = dict()
            values = dict()
            for k in self.state_object_names:
                pos = self.state_to_num['running({})'.format(k)]
                values[int(k[1:])] = state[pos]
                count[int(k[1:])] = 0
            toReturn = ["", "", ""]
            for k in sorted(count.keys()):
                toReturn[0] += str(k) + " "
                toReturn[1] += str(values[k]) + " "
            return toReturn[0] + "\n" + toReturn[1] + "\n"

        elif self.domain == "wildfire":
            num_x = 0
            num_y = 0
            mapping = []
            fuel_mapping = []
            for _, k in self.object_name_to_type.items():
                if k == 'x_pos':
                    num_x += 1
                elif k == 'y_pos':
                    num_y += 1
            for k in self.state_object_names:
                x, y = tuple(k.split(','))
                xn = int(x[1:])
                yn = int(y[1:])
                pos = self.state_to_num['burning({})'.format(k)]
                val = state[pos]
                target = self.para_state_of_objects_nf_values[k][0]
                if target == 1:
                    if val == 1:
                        val = 2
                    else:
                        val = -1
                mapping.append((xn, yn, val))
                pos = self.state_to_num['out-of-fuel({})'.format(k)]
                val = 1 - int(state[pos])
                fuel_mapping.append((xn, yn, val))
            mapping.sort(key=lambda x: (x[1], x[0]))
            fuel_mapping.sort(key=lambda x: (x[1], x[0]))
            s = np.array([a[-1] for a in mapping]).reshape((num_y, num_x))
            fs = np.array([a[-1] for a in fuel_mapping]).reshape((num_y, num_x))
            return np.array_str(s) + "\nFuel\n" + np.array_str(fs)

        elif self.domain == "traffic":
            num_x, num_y = 0, 0
            cells = set()
            intersections = set()
            for k in sorted(self.state_object_names):
                ca_pos = k.find('ca')
                if ca_pos == -1:
                    ia_pos = k.find('ia')
                    a_pos = k.find('a', 2)
                    x = int(k[ia_pos + 2:a_pos])
                    y = int(k[a_pos + 1:])
                    intersections.add((x, y))
                else:
                    a_pos = k.find('a', 2)
                    x = int(k[ca_pos + 2:a_pos])
                    y = int(k[a_pos + 1:])
                    cells.add((x, y))
                if x > num_x:
                    num_x = x
                if y > num_y:
                    num_y = y
            toReturn = ""
            intersectionVals = {}
            for y in range(num_y):
                for x in range(num_x):
                    if (x + 1, y + 1) in cells:
                        pos = self.state_to_num['occupied({})'.format('ca' + str(x + 1) + 'a' + str(y + 1))]
                        if state[pos] == 1:
                            toReturn = toReturn + "O "
                        else:
                            toReturn = toReturn + "C "
                    elif (x + 1, y + 1) in intersections:
                        pos = self.state_to_num['light-signal1({})'.format('ia' + str(x + 1) + 'a' + str(y + 1))]
                        light1 = state[pos]
                        pos = self.state_to_num['light-signal2({})'.format('ia' + str(x + 1) + 'a' + str(y + 1))]
                        light2 = state[pos]
                        if light1 == 1 and light2 == 1:
                            intersectionVals[(x + 1, y + 1)] = "Red"
                        elif light1 == 0 and light2 == 0:
                            intersectionVals[(x + 1, y + 1)] = "Red"
                        elif light1 == 0 and light2 == 1:
                            intersectionVals[(x + 1, y + 1)] = "NS"
                        else:
                            intersectionVals[(x + 1, y + 1)] = "EW"
                        toReturn = toReturn + "X "
                    else:
                        toReturn = toReturn + "  "
                toReturn = toReturn + "\n"
            for key in sorted(intersectionVals.keys()):
                toReturn = toReturn + str(key) + " " + intersectionVals[key] + "\n"
            return toReturn

        elif self.domain == "triangle_tireworld":
            num_x, num_y = 0, 0
            locations = set()
            mapping = dict()
            spare_mapping = dict()
            for k in sorted(self.state_object_names):
                la_pos = k.find('la')
                a_pos = k.find('a', 2)
                x = int(k[la_pos + 2:a_pos])
                y = int(k[a_pos + 1:])
                locations.add((x, y))
                if x > num_x:
                    num_x = x
                if y > num_y:
                    num_y = y
                pos = self.state_to_num['vehicle-at({})'.format(k)]
                val = state[pos]
                target = self.para_state_of_objects_nf_values[k][0]
                if target == 1:
                    if val == 1:
                        val = 3
                    else:
                        val = 2
                mapping[(x, y)] = str(val)
                try:
                    pos = self.state_to_num['spare-in({})'.format(k)]
                    val = state[pos]
                    if val == 1:
                        spare_mapping[(x, y)] = "S"
                    else:
                        spare_mapping[(x, y)] = "0"
                except KeyError:
                    spare_mapping[(x, y)] = "0"
            toReturn = ""
            toReturnSpare = ""
            for i in range(num_y, 0, -1):
                for j in range(1, num_x + 1):
                    try:
                        toReturn = toReturn + mapping[(j, i)] + ' '
                        toReturnSpare = toReturnSpare + spare_mapping[(j, i)] + ' '
                    except KeyError:
                        toReturn = toReturn + "  "
                        toReturnSpare = toReturnSpare + "  "
                toReturn = toReturn + "\n"
                toReturnSpare = toReturnSpare + "\n"
            return toReturn + toReturnSpare + "HasSpare:" + str(
                state[self.state_to_num['hasspare']]) + "\nFlatTire:" + str(
                1 - state[self.state_to_num['not-flattire']])
        elif self.domain == "recon":
            num_x, num_y, num_objects = 0, 0, 0
            for node in self.node_dict:
                if re.match(r'^x((\d)+)$', node):
                    num_x += 1
                elif re.match(r'^y((\d)+)$', node):
                    num_y += 1
                elif re.match(r'^o((\d)+)$', node):
                    num_objects += 1
            
            cell_description = [[[] for _ in range(num_x)] for _ in range(num_y)]
            
            for i in range(num_y):
                for j in range(num_x):
                    if state[self.state_to_num[f'agentAt(a1,x{j},y{i})']]:
                        cell_description[i][j].append('a1')
                    
                    if f'x{j},y{i}' in self.para_state_of_objects_nf_values_names:
                        base_index = self.para_state_of_objects_nf_values_names[f'x{j},y{i}'].index('BASE')
                        hazard_index = self.para_state_of_objects_nf_values_names[f'x{j},y{i}'].index('HAZARD')
                        if self.para_state_of_objects_nf_values[f'x{j},y{i}'][base_index]:
                            cell_description[i][j].append('BASE')
                        if self.para_state_of_objects_nf_values[f'x{j},y{i}'][hazard_index]:
                            cell_description[i][j].append('HAZARD')
                    
            for o in range(num_objects):
                for i in range(num_y):
                    for j in range(num_x):
                        key = f'o{o},x{j},y{i}'
                        if key in self.para_state_of_objects_nf_values_names:
                            index = self.para_state_of_objects_nf_values_names[key].index('objAt')
                            if self.para_state_of_objects_nf_values[key][index]:
                                cell_description[i][j].append(f'o{o}')
            
            
            s = {}
            for key, index in self.state_to_num.items():
                s[key] = int(state[index])
            
            cell_description = [[','.join(cell_description[i][j]) for j in range(num_y)] for i in range(num_x)]
            cell_description.reverse()
            output = tabulate(cell_description, tablefmt='fancy_grid', stralign='center')
            output += "\n\n"

            objects = ['w1', 'l1']
            objects.extend([f'o{o}' for o in range(num_objects)])
            object_state = {o: [] for o in objects}
            for key, index in self.state_to_num.items():
                for o in objects:
                    suffix = f'({o})'
                    if suffix in key:
                        object_state[o].append(f'{key[:-len(suffix)]}:{state[index]}')

            object_state = {o: ','.join(object_state[o]) for o in object_state}
            object_state_str = json.dumps(object_state, indent=4)
            output += object_state_str
            return output
        else:
            s = {}
            for key, index in self.state_to_num.items():
                s[key] = int(state[index])
            return s

    def get_action_repr(self, action_probs=[], action=False):
        if len(action_probs) == 0:
            for key, val in self.action_to_num.items():
                if val == action:
                    return key
            return "noop"
        action_probs = np.around(action_probs, decimals=2)
        mapping = {'noop': action_probs[0]}
        max_action = None
        if action == False:
            action = np.argmax(action_probs, axis=0)
        for key, val in self.action_to_num.items():
            mapping[key] = action_probs[val]
            if val == action:
                max_action = key
        return str(mapping) + "\n" + str(max_action) + "\n"

    def get_embedding_repr(self, embeddings):  # Assuming a single embedding (batch_size = 1)
        embeddings = np.array(embeddings)
        embeddings = np.around(embeddings, decimals=2)[0]
        toReturn = ""
        for node in sorted(self.state_object_names):
            toReturn += node + " " + str(list(embeddings[self.node_dict[node]])) + "\n"
        return toReturn

    def get_node_dict(self):
        return self.node_dict

    def get_expected_next_state_cpt(self, state, action):
        # Make next state and call get_processed input to get next
        next_state = np.array(state, dtype=np.float32)
        num_actions = self.get_num_actions()
        actions = [0] * num_actions
        actions[action] = 1
        reward = self.reward_formula(state, actions)
        for (i, node) in enumerate(state):
            # next_state[i] = self.eval_formula(self.formulae[i], state, actions)
            next_state[i] = self.formulae[i](state, actions)

        return next_state, reward

    def get_transition_prob(self, state, action, next_state):
        prob = 1.0
        bernoulli_probs, _ = self.get_expected_next_state_cpt(state, action)

        for i, state_var in enumerate(next_state):
            if state_var == 1:
                prob *= (bernoulli_probs[i])
            else:
                prob *= (1 - bernoulli_probs[i])
        return prob

    def parse_formula(self, formula, state_var, flag):
        self.formulae[self.state_to_num[state_var]] = self.get_partial_formula(formula)
        self.reward_formula = self.get_partial_formula(self.reward_str)

    def represents_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def remove_last_occurrence(self, lst, element):
        """
        Removes the last occurrence of a given element in a list (modifies list in-place).
        Raises same exception than lst.index(element) if element can not be found.
        """
        try:
            del lst[len(lst) - lst[::-1].index(element) - 1]
            return True
        except:
            return False

    def st(self, x, state, actions):
        return state[int(x)]

    def act(self, x, state, actions):
        return actions[int(x)]

    def const(self, x, state, actions):
        return x

    def summation(self, operands, state, actions):
        ans = 0
        for o in operands:
            ans += o(state, actions)
        return ans

    def multiplication(self, operands, state, actions):
        ans = 1
        for o in operands:
            ans *= o(state, actions)
        return ans

    def subtraction(self, operands, state, actions):
        return operands[0](state, actions) - operands[1](state, actions)

    def division(self, operands, state, actions):
        return operands[0](state, actions) / operands[1](state, actions)

    def logicalAND(self, operands, state, actions):
        ans = True
        for o in operands:
            ans = ans and o(state, actions)
        return ans

    def logicalOR(self, operands, state, actions):
        ans = False
        for o in operands:
            ans = ans or o(state, actions)
        return ans

    def logicalNOT(self, operands, state, actions):
        return not (operands[0](state, actions))

    def bernoulli(self, operands, state, actions):
        return (operands[0](state, actions))

    def exponent(self, operands, state, actions):
        return np.exp(operands[0](state, actions))

    def equalTo(self, operands, state, actions):
        return operands[0](state, actions) == operands[1](state, actions)

    def greaterThanEqualTo(self, operands, state, actions):
        return operands[0](state, actions) >= operands[1](state, actions)

    def lessThanEqualTo(self, operands, state, actions):
        return operands[0](state, actions) <= operands[1](state, actions)

    def switch(self, operands, state, actions):
        for j in range(0, len(operands), 2):
            # If condition is true return value
            if operands[j](state, actions):
                return operands[j+1](state, actions)

    def get_partial_formula(self, str):
        # Add brackets as delimitters
        str = '(' + str + ')'
        stack = []

        # Constructing the current operation
        curr_op = ""
        # print(str)

        to_remove = False

        # Traverse in reverse order
        for c in str[::-1]:
            # print(stack)

            if c == ')':

                # If extra ')' is present
                if to_remove:
                    self.remove_last_occurrence(stack, ')')
                    to_remove = False

                # Push closing bracket
                curr_op = ""
                stack.append(c)
                continue

            elif c == '(' or c == ' ':

                # current operator
                curr_op = curr_op[::-1]
                # print(curr_op)

                # If float/int
                if self.represents_float(curr_op):
                    stack.append(float(curr_op))
                else:
                    if curr_op == "$s":
                        o1 = stack.pop()
                        # This is always a unary operator
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.st, o1))

                    elif curr_op == "$c":
                        o1 = stack.pop()
                        # This is always a unary operator
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.const, o1))

                    elif curr_op == "$a":
                        o1 = stack.pop()
                        # This is always a unary operator
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.act, int(o1) + 1))

                    elif curr_op == "+":
                        # print("Found +")
                        curr = stack.pop()
                        lst = []
                        # Evaluate until ')'
                        while curr != ')':
                            lst.append(curr)
                            curr = stack.pop()
                        stack.append(partial(self.summation, lst))

                    elif curr_op == "-":
                        # print("Found -")
                        o1 = stack.pop()
                        o2 = stack.pop()
                        # This is always a binary operator
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.subtraction, [o1, o2]))

                    elif curr_op == "/":
                        # print("Found /")
                        o1 = stack.pop()
                        o2 = stack.pop()
                        # This is always a binary operator
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.division, [o1, o2]))

                    elif curr_op == "*":
                        curr = stack.pop()
                        lst = []
                        # Evaluate until ')'
                        while curr != ')':
                            lst.append(curr)
                            curr = stack.pop()
                        stack.append(partial(self.multiplication, lst))

                    elif curr_op == "and":
                        # Apply until )
                        curr = stack.pop()
                        lst = []
                        # Evaluate until ')'
                        while curr != ')':
                            lst.append(curr)
                            curr = stack.pop()
                        stack.append(partial(self.logicalAND, lst))

                    elif curr_op == "or":
                        curr = stack.pop()
                        lst = []
                        # Evaluate until ')'
                        while curr != ')':
                            lst.append(curr)
                            curr = stack.pop()
                        stack.append(partial(self.logicalOR, lst))

                    elif curr_op == "~":
                        o1 = stack.pop()
                        # This is always a unary operator
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.logicalNOT, [o1]))

                    elif curr_op == "Bernoulli":
                        # print("Found Bernoulli")
                        o1 = stack.pop()
                        # This is always a unary operator
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.bernoulli, [o1]))

                    elif curr_op == "exp":
                        # print("Found exp")
                        # This is always a unary operator
                        o1 = stack.pop()
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.exponent, [o1]))

                    elif curr_op == "==":
                        # print("Found ==")
                        # This is always a binary operator
                        o1 = stack.pop()
                        o2 = stack.pop()
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.equalTo, [o1, o2]))

                    elif curr_op == ">=":
                        # print("Found >=")
                        # This is always a binary operator
                        o1 = stack.pop()
                        o2 = stack.pop()
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.greaterThanEqualTo, [o1, o2]))

                    elif curr_op == "<=":
                        # print("Found <=")
                        # This is always a binary operator
                        o1 = stack.pop()
                        o2 = stack.pop()
                        if stack[len(stack) - 1] == ')':
                            stack.pop()
                        stack.append(partial(self.lessThanEqualTo, [o1, o2]))

                    elif curr_op == "switch":
                        # print("Found switch")

                        # Getting the last ')'
                        idx = len(stack) - 1 - stack[::-1].index(')')

                        # Traversing conditions and values in reverse order
                        lst = stack[idx + 1:][::-1]

                        # Delete extra ')'
                        del stack[idx:]
                        stack.append(partial(self.switch, lst))

                    elif curr_op == ":":
                        pass
                    elif curr_op == "":
                        # If extra bracket is to be removed
                        if to_remove:
                            self.remove_last_occurrence(stack, ')')
                            to_remove = False
                    else:
                        # Operation not found
                        raise ValueError("Operation not found " + curr_op)
                curr_op = ""
                if c == '(':
                    # If extra ')' is present
                    to_remove = True
            else:
                curr_op += c
                # No extra ')' present
                to_remove = False
        # print(len(stack))
        return stack.pop()

    def get_max_reward(self):
        return self.reward_max

    def get_min_reward(self):
        return self.reward_min