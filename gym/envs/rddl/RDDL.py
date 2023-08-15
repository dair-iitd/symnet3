import sys
import os
import random
import ctypes
import numpy as np
import gym
import tempfile
import shutil

from gym import Env
from gym.utils import seeding

# For instance parser
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
parser_path = os.path.abspath(os.path.join(curr_dir_path, "../../../utils"))
if parser_path not in sys.path:
	sys.path = [parser_path] + sys.path

from parse_instance import InstanceParser


redirect = True
def redirect_stdout():
	sys.stdout.flush() # <--- important when redirecting to files
	newstdout = os.dup(1)
	devnull = os.open(os.devnull, os.O_WRONLY)
	os.dup2(devnull, 1)
	os.close(devnull)
	sys.stdout = os.fdopen(newstdout, 'w')

class RDDLEnv(Env):
	def __init__(self, domain, instance):
		self.domain = domain + '_mdp'
		self.problem = domain + '_inst_mdp__' + instance
		self.instance = instance

		self.instance_parser = InstanceParser(self.domain[:-4], self.instance)

		# Seed Random number generator
		self._seed()

		# f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), './rddl/parsed/',self.problem)))
		f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../rddl/parsed/',self.problem)))

		p = "##"  # Values of p are hard-coded in PROST. Should not be changed.
		for l in f:
			if (p == "## horizon\n"):
				h = int(l)
			elif (p == "## number of actions\n"
				and self.domain != 'academic_advising_mdp'):
				num_act = int(l)
			elif (p == "## number of action fluents\n"
				and self.domain == 'academic_advising_mdp'):
				num_act = int(l) + 1
			elif (p == "## number of det state fluents\n"):
				num_det = int(l)
			elif (p == "## number of prob state fluents\n"):
				num_prob = int(l)
			elif (p == "## initial state\n"):
				init = [int(i) for i in l.split()]
				break
			p = l
		f.close()

		# Problem parameters
		self.num_state_vars = num_det + num_prob  # number of state variables
		self.num_action_vars = self.instance_parser.get_num_actions()  # number of action variables
		self.initial_state = init
		self.state_type = type(self.initial_state)
		self.state = np.array(self.initial_state)  # current state
		self.horizon = h  # episode horizon
		self.tstep = 1  # current time step
		self.done = False  # end_of_episode flag

		# Set up RDDL Simulator clibxx.so
		qwwe = str(self.instance).split(".|_")
		
		path = tempfile.NamedTemporaryFile().name
		shutil.copy2(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../rddl/lib/clibxx.so')), path)
		self.rddlsim = ctypes.CDLL(path)

		# self.rddlsim = ctypes.CDLL(os.path.abspath(os.path.join(os.path.dirname(__file__),'./rddl/lib/clibxx{}.so'.format(qwwe[0]))))
		self.rddlsim.step.restype = ctypes.c_double

		# Better without the explicit encoding
		parsed_file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../rddl/parsed/', self.problem))
		parsed_file_name_byteobject = parsed_file_name.encode()
		parsed_file_name_ctype = ctypes.create_string_buffer(parsed_file_name_byteobject, len(parsed_file_name_byteobject))

		if redirect:
			sys.stdout.flush()
			_origstdout = sys.stdout
			_oldstdout_fno = os.dup(sys.stdout.fileno())
			redirect_stdout()
		self.rddlsim.parse(parsed_file_name_ctype.value)
		if redirect:
			sys.stdout = _origstdout
			sys.stdout.flush()
			os.dup2(_oldstdout_fno, 1)
		
	# Do not understand this yet. Almost all other sample environments have it, so we have it too.
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	# Take a real step in the environment. Current state changes.
	def _step(self, action_var):
		# Convert state and action to c-types
		s = self.state
		ss = s.tolist()
		sss = (ctypes.c_double * len(ss))(*ss)
		action = (ctypes.c_int)(action_var)

		# Call Simulator
		reward = self.rddlsim.step(sss, len(ss), action)
		self.state = np.array(sss, dtype=np.int8)

		# Advance time step
		self.tstep = self.tstep + 1
		if self.domain == "navigation_mdp":
			if self.tstep > self.horizon:
				self.done = True
		elif self.domain == "academic_advising_mdp":
			if self.tstep > self.horizon:
				self.done = True
		else:
			if self.tstep > self.horizon:
				self.done = True

		return self.state, reward, self.done, {}

	def test_step(self, action_var):

		# Convert state and action to c-types
		s = self.state
		ss = s.tolist()
		sss = (ctypes.c_double * len(ss))(*ss)
		action = (ctypes.c_int)(action_var)

		# Call Simulator
		reward = self.rddlsim.step(sss, len(ss), action)
		self.state = np.array(sss, dtype=np.int8)

		# Advance time step
		self.tstep = self.tstep + 1
		if self.domain == "navigation_mdp":
			if self.tstep > self.horizon:
				self.done = True
		elif self.domain == "academic_advising_mdp":
			if self.tstep > self.horizon:
				self.done = True
		else:
			if self.tstep > self.horizon:
				self.done = True

		# # Handle episode end in case of navigation
		# # Not able to check if robot's position is same as goal state
		# if self.domain == "navigation_mdp" and not(np.any(self.state)):
		#	 self.done = True

		return self.state, reward, self.done, {}

	# Take an imaginary step to get the next state and reward. Current state does not change.
	def pseudostep(self, curr_state, action_var):

		# Convert state and action to c-types
		s = np.array(curr_state)
		ss = s.tolist()
		sss = (ctypes.c_double * len(ss))(*ss)
		action = (ctypes.c_int)(action_var)

		# Call Simulator
		reward = self.rddlsim.step(sss, len(ss), action)
		next_state = np.array(sss, dtype=np.int8)

		return next_state, reward

	# Using Sampling
	def get_expected_step(self, state, action_var, num_samples=50):
		expected_rew = 0.0
		next_state_l = []
		for i in range(num_samples):
			next_state, reward = self.pseudostep(state, action_var)
			next_state_l.append(next_state)
			expected_rew += reward
		next_state = np.mean(np.stack(next_state_l), axis=0)

		return next_state, expected_rew/num_samples

	def _reset(self):
		self.state = np.array(self.initial_state)
		self.tstep = 1
		self.done = False
		return self.state, self.done

	def random_reset(self):
		self.state = self.random_state()
		self.tstep = 1
		self.done = False

		return self.state, self.done

	def initialize_using_state(self, state):
		self.state = state
		self.tstep = 1
		self.done = False

		return self.state, self.done

	def random_state(self):
		if self.domain == 'academic_advising_mdp':
			# (taken, passed)
			num_courses = self.get_num_nodes()
			valid_state_vars_values = [(0,0), (1,0), (1,1)]
			state_temp = np.zeros(self.num_state_vars)
			for i in range(num_courses):
				val = random.choice(valid_state_vars_values)
				state_temp[i] = val[0]
				state_temp[num_courses+i] = val[1]
			return state_temp
		else:
			raise Exception("Random initial state not defined for %s domain"%self.domain)
		# return np.random.choice(2, self.num_state_vars)

	def _set_state(self, state):
		self.state = state

	def _close(self):
		pass

	def get_fluent_features(self, state):
		return np.array(self.instance_parser.get_fluent_features(state))

	def get_numeric_unary_nf_features(self, state):
		return np.array(self.instance_parser.get_numeric_unary_nf_features(state))

	def get_action_details(self):
		return self.instance_parser.get_action_details()

	def get_extended_action_details(self):
		return self.instance_parser.get_extended_action_details()

	def get_nf_features(self):
		return np.array(self.instance_parser.get_nf_features())

	def get_numeric_para_unary_nf_features(self):
		return np.array(self.instance_parser.get_numeric_para_unary_nf_features())

	def get_adjacency_list(self):
		return self.instance_parser.get_adjacency_list()

	def get_extended_adjacency_list(self):
		return self.instance_parser.get_extended_adjacency_list()

	def get_num_adjacency_list(self):
		return self.instance_parser.get_num_adjacency_list()

	def get_feature_dims(self):
		return self.instance_parser.get_feature_dims()

	def get_num_action_nodes(self):
		return self.instance_parser.get_num_action_nodes()

	def get_num_actions(self):
		return self.instance_parser.get_num_actions()

	def get_action_templates(self):
		return self.instance_parser.get_action_templates()

	def get_graph_fluent_features(self, state):
		return np.array(self.instance_parser.get_graph_fluent_features(state))

	def get_num_nodes(self):
		return self.instance_parser.get_num_nodes()

	def get_num_graph_fluents(self):
		return self.instance_parser.get_num_graph_fluents()

	def get_num_type_actions(self):
		return self.instance_parser.get_num_type_actions()

	def get_graph_type(self):
		return self.instance_parser.graph_type

	def get_state_repr(self, state):
		return self.instance_parser.get_state_repr(state)

	def get_action_repr(self, action_probs, action):
		return self.instance_parser.get_action_repr(action_probs, action)

	def get_embedding_repr(self, embeddings):
		return self.instance_parser.get_embedding_repr(embeddings)

	def get_attr(self, attr):  # For getting any attribute in the instance parser class directly
		return self.instance_parser.get_attr(attr)
	
	def get_expected_next_state(self, state, action):
		return self.instance_parser.get_expected_next_state(state, action)

	def get_expected_step_cpt(self, state, action):
		return self.instance_parser.get_expected_next_state_cpt(state, action)

	def get_transition_prob(self, state, action, next_state):
		return self.instance_parser.get_transition_prob(state, action, next_state)

	def get_node_dict(self):
		return self.instance_parser.get_node_dict()

	def get_num_to_action(self):
		return self.instance_parser.get_num_to_action()

	def get_max_reward(self):
		return self.instance_parser.get_max_reward()

	def get_min_reward(self):
		return self.instance_parser.get_min_reward()

if __name__ == '__main__':
	env_name = "RDDL-{}{}-v1".format(sys.argv[1], sys.argv[2])
	env = gym.make(env_name)
	env.detailed_init()
	env.seed(0)
	NUM_EPISODES = 1
	for i in range(NUM_EPISODES):
		reward = 0  # epsiode reward
		rwd = 0  # step reward
		curr, done = env.reset()  # current state and end-of-episode flag
		while not done:
			action = random.randint(0, env.num_action_vars)  # choose a random action
			nxt, rwd, done, _ = env.step(action)  # next state and step reward
			print(('state: {}  action: {}  reward: {} next: {}'.format(curr, action, rwd, nxt)))
			curr = nxt
			reward += rwd
		print(('Episode Reward: {}'.format(reward)))
		print()

	env.close()
