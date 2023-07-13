import symnet3.symnet3 as symnet3
from itertools import count

symnet3_args = {"general_params", "se_params", "ad_params", "ge_params", "tm_params", "env_instance_wrapper"}

class ModelFactory:
	def __init__(self, args):
		self.args = args
		self.global_counter = count()
		self.policynet_optim = self.args["policynet_optim"]
		self.grad_clip_value = self.args["grad_clip_value"]

		self.ckpt = None
		self.ckpt_manager = None

		self.total_num_updates = 0.0
		self.total_examples = 0.0

		self.total_episodes = 0.0
		self.num_complete = 0.0
		self.avg_steps = 0.0

	def set_ckpt_metadata(self, ckpt, ckpt_manager):
		self.ckpt = ckpt
		self.ckpt_manager = ckpt_manager

	def create_network(self, env_instance_wrapper=None): #	 Creates a combined network
		self.args['env_instance_wrapper'] = env_instance_wrapper
		return symnet3.SymNet3(**dict((k, self.args[k]) for k in symnet3_args))

	@staticmethod
	def copy_params(target_variables, source_variables):
		for t, s in zip(target_variables, source_variables):
			t.assign(s)  # copies the variables of global model (g) into local model (l)

	def load_ckpt(self, ckpt_num=None):
		if self.ckpt_manager.latest_checkpoint:
			if ckpt_num:
				ckpt_path = self.ckpt_manager._checkpoint_prefix + "-" + ckpt_num
				print(("Loading model checkpoint: {}".format(ckpt_path)))
				self.ckpt.restore(ckpt_path)
			else:
				print(("Loading model checkpoint: {}".format(self.ckpt_manager.latest_checkpoint)))
				self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
		else:
			print("Training new model")

	def save_ckpt(self):
		ckpt_loc = self.ckpt_manager.save()
		print("Model saved at: " + ckpt_loc)
		return ckpt_loc

