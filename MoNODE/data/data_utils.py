import os
import yaml
import json
import torch
from   torch.utils import data
from data.data_gen import gen_sin_data, gen_lv_data, gen_rmnist_data, gen_bb_data, gen_mocap_data, gen_mocap_shift_data, gen_ecg_data
import pickle

from model.misc import io_utils

def _adjust_name(data_path, substr, insertion):
	idx = data_path.index(substr)
	return data_path[:idx] + '-' + insertion + data_path[idx:]


def load_data(args, device, dtype):
	if args.task in ['rot_mnist', 'rot_mnist_ou', 'mov_mnist', 'sin', 'lv', 'spiral', 'bb', 'mocap', 'mocap_shift', 'ecg'] :
		(trainset, valset, testset), params = __load_data(args, device, dtype, args.task)
	else:
		return ValueError(r'Invalid task {arg.task}')
	return trainset, valset, testset, params  #, N, T, D, data_settings


def __load_data(args, device, dtype, dataset=None):
	#load data parameters
	with open("data/config.yml", 'r') as stream:
		try:
			params = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	#create data folder/files
	if args.task == 'sin':
		#override config file
		if args.noise is not None:
			params[args.task]['noise'] = args.noise

		folder_path = os.path.join(args.data_root,args.task + str(params[args.task]['noise']))	     
	else:
		folder_path = os.path.join(args.data_root,args.task)

	io_utils.makedirs(folder_path)
	data_path_tr = os.path.join(folder_path,f'{dataset}-tr-data.pkl')
	data_path_vl = os.path.join(folder_path,f'{dataset}-vl-data.pkl')
	data_path_te = os.path.join(folder_path,f'{dataset}-te-data.pkl')

	data_path_ytr = os.path.join(folder_path,f'{dataset}-ytr-data.pkl')
	data_path_yvl = os.path.join(folder_path,f'{dataset}-yvl-data.pkl')
	data_path_yte = os.path.join(folder_path,f'{dataset}-yte-data.pkl')

	#adjust name if specifc configuration
	if dataset == 'bb':
		data_path_tr = _adjust_name(data_path_tr, '.pkl', str(params[dataset]['nballs']))
		data_path_vl = _adjust_name(data_path_vl, '.pkl', str(params[dataset]['nballs']))
		data_path_te = _adjust_name(data_path_te, '.pkl', str(params[dataset]['nballs']))
	elif dataset == 'ecg':
		data_path_tr = _adjust_name(data_path_tr, '.pkl', 
							  str(params[dataset]['type']) + 
							  str(params[dataset]['f']) + str(params[dataset]['dataset']) + 
							  str('150' if params[dataset]['qrs_only'] else params[dataset]['train']['T']) + 
							  str('QRS' if params[dataset]['qrs_only'] else '') + 
							  str('join_atrial' if params[dataset]['join_atrial'] else ''))
		data_path_vl = _adjust_name(data_path_vl, '.pkl', 
							  str(params[dataset]['type']) +
							    str(params[dataset]['f']) + 
								str(params[dataset]['dataset']) + 
								str('150' if params[dataset]['qrs_only'] else params[dataset]['valid']['T']) + 
								str('QRS' if params[dataset]['qrs_only'] else '') + 
								str('join_atrial' if params[dataset]['join_atrial'] else ''))
		data_path_te = _adjust_name(data_path_te, '.pkl', 
							  str(params[dataset]['type']) + 
							  str(params[dataset]['f']) + 
							  str(params[dataset]['dataset']) + 
							  str('150' if params[dataset]['qrs_only'] else params[dataset]['test']['T']) + 
							  str('QRS' if params[dataset]['qrs_only'] else '') + 
							  str('join_atrial' if params[dataset]['join_atrial'] else ''))

	#load or generate data
	try:
		Xtr = torch.load(data_path_tr)
		Xvl = torch.load(data_path_vl)
		Xte = torch.load(data_path_te)
		Ytr = None
		Yvl = None
		Yte = None
		print('Data loaded')
		#if loaded data does not match the parameter settings assert and re generate the data 
		if dataset != 'ecg':
			assert Xtr.shape[0] == params[dataset]['train']['N'] and Xtr.shape[1] == params[dataset]['train']['T'] 
			assert Xvl.shape[0] == params[dataset]['valid']['N'] and Xvl.shape[1] == params[dataset]['valid']['T']
			assert Xte.shape[0] == params[dataset]['test']['N'] and Xte.shape[1] == params[dataset]['test']['T']
		else:
			with open(data_path_ytr, "rb") as f:
				Ytr = pickle.load(f)
			with open(data_path_yvl, "rb") as f:
				Yvl = pickle.load(f)
			with open(data_path_yte, "rb") as f:
				Yte = pickle.load(f)
			print('GT loaded')
	except:
		print(f'Could not load data from {data_path_tr}, {data_path_vl}, {data_path_te} or GT from {data_path_ytr}, {data_path_yvl}, {data_path_yte}. Generating data...')
		with open(folder_path+'/gen_info.txt', 'w') as f:
			f.write(json.dumps(params[dataset]))

		if dataset=='sin':
			data_loader_fnc = gen_sin_data
		elif dataset == 'lv':
			data_loader_fnc = gen_lv_data
		elif dataset == 'rot_mnist':
			data_loader_fnc = gen_rmnist_data
		elif dataset == 'bb':
			data_loader_fnc = gen_bb_data
		elif dataset == 'mocap':
			data_loader_fnc = gen_mocap_data
		elif dataset == 'mocap_shift':
			data_loader_fnc = gen_mocap_shift_data
		elif dataset == 'ecg':
			data_loader_fnc = gen_ecg_data

		data_loader_fnc(data_path_tr, data_path_ytr, params, flag='train')
		data_loader_fnc(data_path_vl, data_path_yvl, params, flag='valid')
		data_loader_fnc(data_path_te, data_path_yte, params, flag='test')

		Xtr = torch.load(data_path_tr)
		Xvl = torch.load(data_path_vl)
		Xte = torch.load(data_path_te)

		with open(data_path_ytr, "rb") as f:
				Ytr = pickle.load(f)
		with open(data_path_yvl, "rb") as f:
			Yvl = pickle.load(f)
		with open(data_path_yte, "rb") as f:
			Yte = pickle.load(f)

	if dataset == 'bb':
		Xtr = torch.Tensor(Xtr).unsqueeze(2)
		Xvl = torch.Tensor(Xvl).unsqueeze(2)
		Xte = torch.Tensor(Xte).unsqueeze(2)

	if dataset == 'ecg':
		exclude_leads = params[dataset]['exclude_leads']
		lead_idx = {
			'I': 0, 
			'II': 1,
			'III': 2, 
			'aVR': 3, 
			'aVL': 4,
			'aVF': 5,
			'V1': 6,
			'V2': 7,
			'V3': 8, 
			'V4': 9, 
			'V5': 10, 
			'V6': 11
		}

		include_idx = [idx for lead, idx in lead_idx.items() if lead not in exclude_leads]
		Xtr = Xtr[:, :, include_idx]
		Xvl = Xvl[:, :, include_idx]
		Xte = Xte[:, :, include_idx]

	Xtr = Xtr.to(device).to(dtype)
	Xvl = Xvl.to(device).to(dtype)
	Xte = Xte.to(device).to(dtype)

	print('Train data: ', Xtr.shape)
	print('Val   data: ', Xvl.shape)
	print('Test  data: ', Xte.shape)

	return __build_dataset(args.num_workers, args.batch_size, Xtr, Xvl, Xte, Ytr, Yvl, Yte), params


class Dataset(data.Dataset):
	def __init__(self, Xtr, Ytr=None):
		self.Xtr = Xtr # N,T,_
		self.Ytr = Ytr
	def __len__(self):
		return len(self.Xtr)
	def __getitem__(self, idx):
		X = self.Xtr[idx]
		y = self.Ytr[idx] if self.Ytr is not None else 0
		return self.Xtr[idx], y
	@property
	def shape(self):
		return self.Xtr.shape


def __build_dataset(num_workers, batch_size, Xtr, Xvl, Xte, Ytr=None, Yvl=None, Yte=None, shuffle=True):
	# Data generators
	if num_workers>0:
		from multiprocessing import Process, freeze_support
		torch.multiprocessing.set_start_method('spawn', force="True")

	tr_params = {'batch_size': min(batch_size,Xtr.shape[0]), 'shuffle': shuffle, 'num_workers': num_workers, 'drop_last': True}
	trainset  = Dataset(Xtr, Ytr)
	trainset  = data.DataLoader(trainset, **tr_params)
	vl_params = {'batch_size': min(batch_size,Xvl.shape[0]), 'shuffle': shuffle, 'num_workers': num_workers, 'drop_last': True}
	validset  = Dataset(Xvl, Yvl)
	validset  = data.DataLoader(validset, **vl_params)
	te_params = {'batch_size': min(batch_size,Xte.shape[0]), 'shuffle': shuffle, 'num_workers': num_workers, 'drop_last': True}
	testset   = Dataset(Xte, Yte)
	testset   = data.DataLoader(testset, **te_params)
	return trainset, validset, testset

