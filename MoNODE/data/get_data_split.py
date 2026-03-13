from collections import defaultdict
import os 
import math 
import hashlib
import random 
import pandas as pd 

def gen_ecg_data(args): 

	split = args.split 
	anatomy = args.anatomy
	dataset = args.dataset
	root_dir = args.root_dir
	out_dir = args.out_dir

	data_paths = defaultdict(lambda: defaultdict(list))
	if dataset.lower() == 'medalcare_xl':
		subclasses = ['LAD_0.3', 'LAD_1.0', 'LCX_0.3_ant', 'LCX_0.3_post', 'LCX_1.0_ant', 'LCX_1.0_post', 'RCA_0.3', 'RCA_1.0']
		classes_dict = {
			'ventricular': subclasses + ['lbbb', 'rbbb'],
			'artrial': ['av_block', 'fam', 'iab', 'lae'],
			'normal': ['sinus']
		}

		total_cls = defaultdict(int)
		for cls_dir in os.listdir(root_dir):
			if cls_dir == 'mi':
				for subclass_dir in os.listdir(os.path.join(root_dir, cls_dir)):
					if subclass_dir in subclasses:
						cur_dir = os.path.join(root_dir, cls_dir, subclass_dir, split)
						for run_dir in os.listdir(cur_dir):
							cur_dir = os.path.join(cur_dir, run_dir)
							for data_file in os.listdir(cur_dir):
								data_path = os.path.join(cur_dir, data_file)
								if data_path.endswith('csv') and 'raw' in data_path:
									total_cls['mi'] += 1
									data_paths[subclass_dir][run_dir].append(os.path.join(cur_dir, data_path))
			else:
				cur_dir = os.path.join(root_dir, cls_dir, split)
				for run_dir in os.listdir(cur_dir):
					cur_dir = os.path.join(cur_dir, run_dir)
					for data_file in os.listdir(cur_dir):
						data_path = os.path.join(cur_dir, data_file)
						if data_path.endswith('csv') and 'raw' in data_path:
							total_cls[cls_dir] += 1
							data_paths[cls_dir][run_dir].append(os.path.join(cur_dir, data_path))
		
		out_dataset = {
			'data_path': [],
			'label': [],
			'hash': []
		}
		n_sinus = (total_cls['lbbb'] + total_cls['rbbb'] + total_cls['mi']) if anatomy == 'ventricular' \
			else (total_cls['av_block'] + total_cls['fam'] + total_cls['iab'] + total_cls['lae'])
		n_sinus = int(math.ceil(n_sinus * 1.25))
		n_sinus_class = len(classes_dict['ventricular']) + 1 if anatomy == 'atrial' else len(classes_dict['atrial']) + 1
		n_per_class = math.ceil(n_sinus / n_sinus_class)

		for cls, run in data_paths.items():
			if cls in classes_dict[anatomy]:
				for paths in run.values():
					for path in paths:
						out_dataset['data_path'].append(path)
						out_dataset['label'].append(cls)

						filename = path.split('/')[-1]
						out_dataset['hash'].append(hashlib.sha256(filename.encode()).hexdigest())
			else:
				n_runs = len(run)
				n_per_run = math.ceil(n_per_class / n_runs)
				for paths in run.values():
					random.shuffle(paths)
					for i in range(n_per_run):
						out_dataset['data_path'].append(paths[i])
						out_dataset['label'].append(cls)

						filename = paths[i].split('/')[-1]
						out_dataset['hash'].append(hashlib.sha256(filename.encode()).hexdigest())

		df = pd.DataFrame(out_dataset)
		df.to_csv(os.path.join(out_dir, f"{dataset.lower()}_{split}_{anatomy}.csv"), index=False)