import os, argparse

def create_dataset(domain, start_instance, num_instances, prost_log, save_folder):
	episodes = []
	transitions = []
	for i in range(1, num_instances+1):
		episodes = []
		f = open(os.path.join(prost_log, str(start_instance + i)+".result"))
		for line in f.readlines():
			
			if ">>> END OF ROUND" in line:
				episodes.append(transitions)
				transitions = []

			# Current state: | 1 0 1 0 0 0 0 1 1 
			if "Current state:" in line:
				res = line.split(":")[1].split("|")
				state = res[0].strip()+" "+res[1].strip()

			# Submitted action: set(x2, y2) 
			if "Submitted action:" in line:
				action = line.split(":")[1].strip().replace(" ", "")

			# Immediate reward: -1.000000
			if "Immediate reward:" in line:
				reward = line.split(":")[1].strip()
				transitions.append([str(start_instance+i), state, action, reward])


		f = open(os.path.join(save_folder, str(start_instance+i)+".csv"), "w")

		for ep in episodes:
			for t in ep:
				res = str(t[0]) + ":"
				res += ",".join(t[1].strip().split(" ")) + ":"
				res += str(t[2]) + ":"
				res += str(t[3])
				f.write(res+"\n")

		f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--domain", help="name of the domain")
	parser.add_argument("--start_instance", help="starting instance number", type=int)
	parser.add_argument("--num_instances", help="number of instances to build dataset for", type=int)
	parser.add_argument("--prost_log", help="path of prost logs")
	parser.add_argument("--save_folder", help="folder to save dataset")
	args = parser.parse_args()

	if not os.path.isdir(args.save_folder):
		os.mkdir(args.save_folder)
	
	create_dataset(
		domain=args.domain, 
		start_instance=args.start_instance,
		num_instances=args.num_instances,
		prost_log=args.prost_log,
		save_folder=args.save_folder
	)