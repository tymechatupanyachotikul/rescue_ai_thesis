import os
import yaml
import json
import torch
import random
import numpy as np
from   torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from model.misc import io_utils
from scipy.signal import medfilt, iirnotch, filtfilt, butter, resample

DEFAULT_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MIMIC_IV_LEADS = ['I', 'II', 'III', 'aVF', 'aVR', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

LEADS_DICT = {
    'medalcare-xl': DEFAULT_LEADS,
    'uk_biobank': DEFAULT_LEADS,
    'mimic-iv': MIMIC_IV_LEADS
}

def filter_bandpass(signal, fs):
	"""
	Bandpass filter
	:param signal: 2D numpy array of shape (channels, time)
	:param fs: sampling frequency
	:return: filtered signal
	"""
	is_tensor = isinstance(signal, torch.Tensor)

	if is_tensor:
		device = signal.device
		signal = signal.detach().cpu().numpy()

	transposed = signal.shape[0] > signal.shape[1]
	if transposed:
		signal = signal.T

	# Remove power-line interference
	b, a = iirnotch(50, 30, fs)
	filtered_signal = np.zeros_like(signal)
	for c in range(signal.shape[0]):
		filtered_signal[c] = filtfilt(b, a, signal[c])

	# Simple bandpass filter
	b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
	for c in range(signal.shape[0]):
		filtered_signal[c] = filtfilt(b, a, filtered_signal[c])

	# Remove baseline wander
	baseline = np.zeros_like(filtered_signal)
	for c in range(filtered_signal.shape[0]):
		kernel_size = int(0.4 * fs) + 1
		if kernel_size % 2 == 0:
			kernel_size += 1  # Ensure kernel size is odd
		baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
	filter_ecg = filtered_signal - baseline

	if transposed:
		filter_ecg = filter_ecg.T
	if is_tensor:
		return torch.from_numpy(filter_ecg).to(device)
	
	return filter_ecg

def load_data(args, dtype):
	if args.task in ['rot_mnist', 'rot_mnist_ou', 'mov_mnist', 'sin', 'lv', 'spiral', 'bb', 'mocap', 'mocap_shift', 'ecg'] :
		(trainset, valset, testset, manager), params = __load_data(args, dtype, args.task)
	else:
		return ValueError(r'Invalid task {arg.task}')
	return trainset, valset, testset, manager, params  #, N, T, D, data_settings

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

def get_data_params(root_dir, dataset, sample_type, beat_type, task, exclude_leads=[]):

	splits = ['train', 'valid', 'test']
	data_param_path = os.path.join('data', task, f'{dataset}_{beat_type}_{sample_type}_data_params.json')
	if os.path.exists(data_param_path):
		with open(data_param_path, 'r') as f:
			split_dict = json.load(f)
			
			return split_dict['train'], split_dict['valid'], split_dict['test']
		
	split_dict = {}
	for split in splits:
		base_dir = os.path.join(root_dir, dataset, 'segments', split, beat_type, sample_type)

		split_info = {
			'file_paths': [],
			'class': [], 
			'run_id': [],
			'exclude_leads_in': exclude_leads
		}

		for file_path in os.listdir(base_dir):
			if file_path.endswith('.pth'):
				if dataset.lower() == 'medalcare-xl':
					split_info['file_paths'].append(os.path.join(base_dir, file_path))

					path_split = file_path.split('_')
					split_info['run_id'].append(path_split[1])
					split_info['class'].append('_'.join(path_split[3:]).split('.')[0])
				else:
					split_info['file_paths'].append(os.path.join(base_dir, file_path))

					path_split = file_path.split('_')
					split_info['run_id'].append(path_split[-1].split('.')[0])

		split_dict[split] = split_info
	
	with open(data_param_path, 'w') as f:
		json.dump(split_dict, f, indent=4)

	return split_dict['train'], split_dict['valid'], split_dict['test']


def __load_data(args, dtype, dataset=None):
	#load data parameters
	with open("data/config.yml", 'r') as stream:
		try:
			params = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	params[dataset]['beat_type'] = args.segment_type
	folder_path = os.path.join(args.data_root,args.task)

	io_utils.makedirs(folder_path)
	train_params, valid_params, test_params = get_data_params(args.dataset_root, params[dataset]['dataset'], params[dataset]['sample_type'], params[dataset]['beat_type'], dataset, params[dataset]['exclude_leads_in'])

	return __build_dataset(args.num_workers, args.batch_size, train_params, valid_params, test_params, dtype, params[dataset]['dataset'], use_cache=params[dataset]['use_cache']), params


class ECGDataset(data.Dataset):
	def __init__(self, file_paths, labels, run_id, dtype, dataset, exclude_leads=[], shared_cache=None, return_file_path=False):
		self.file_paths = file_paths
		self.labels = labels if labels else None
		self.run_id = run_id
		self.exclude_leads = exclude_leads
		self.cache = shared_cache
		self.dtype = dtype
		self.return_file_path = return_file_path
		
		self.lead_idx = {
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

		if self.exclude_leads:
			self.include_idx = [idx for lead, idx in self.lead_idx.items() if lead not in self.exclude_leads]
		else:
			self.include_idx = None

		self.idx_map = None 
		self.permute_lead = False

		if not LEADS_DICT[dataset.lower()] == DEFAULT_LEADS:
			source_leads = LEADS_DICT[dataset.lower()]
			self.idx_map = [source_leads.index(lead) for lead in DEFAULT_LEADS]
			self.permute_lead = True

	def __len__(self):
		return len(self.file_paths)
	
	def __getitem__(self, idx):
		if self.cache is not None and idx in self.cache:
			X = self.cache[idx]
		else:
			X = torch.load(self.file_paths[idx]).to(dtype=self.dtype)
			if self.permute_lead:
				X = X[:, self.idx_map]
			
			if self.include_idx is not None:
				X = X[:, self.include_idx]
			if self.dataset.lower() != 'medalcare-xl':
				X = filter_bandpass(X, 500) 
			if self.cache is not None and idx not in self.cache:
				self.cache[idx] = X
		
		if self.return_file_path:
			y = (self.labels[idx], self.run_id[idx], self.file_paths[idx]) if self.labels is not None else 0
		else:
			y = (self.labels[idx], self.run_id[idx]) if self.labels is not None else 0

		return X, y
	
	def get_class_samples(self, k=3, classes=None):

		all_classes = set(self.labels) if classes == None else classes
		classes_dict = {}
		for _cls in all_classes:
			idx = [i for i, val in enumerate(self.labels) if val == _cls]
			idx = random.sample(idx, k=k)

			samples = []
			for i in idx:
				if self.cache is not None and i in self.cache:
					samples.append(self.cache[i])
				else:
					X = torch.load(self.file_paths[i]).to(dtype=self.dtype)
					if self.permute_lead:
						X = X[:, self.idx_map]
					
					if self.include_idx is not None:
						X = X[:, self.include_idx]
					samples.append(X)
			
			classes_dict[_cls] = pad_sequence(samples, batch_first=True, padding_value=0.0)
		
		return classes_dict
	

def pad_collate(batch):
	sequences = [torch.nan_to_num(item[0], nan=0.0) for item in batch]
	labels = [item[1] for item in batch]
	lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)

	padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

	N, T = padded_sequences.shape[:2]
	T_idx = torch.arange(T).unsqueeze(0).expand(N, T) 
	mask = (T_idx < lengths.unsqueeze(1)).to(padded_sequences.device)

	return padded_sequences, labels, mask

def __build_dataset(num_workers, batch_size, train_params, valid_params, test_params, dtype, dataset, use_cache=True, shuffle=True):
	# Data generators
	manager = None
	if num_workers>0:
		import multiprocessing
		torch.multiprocessing.set_start_method('spawn', force="True")
		manager = multiprocessing.Manager() 

		train_cache = manager.dict() if use_cache else None
		valid_cache = manager.dict() if use_cache else None
		test_cache  = manager.dict() if use_cache else None
	else:
		train_cache = {} if use_cache else None
		valid_cache = {} if use_cache else None
		test_cache  = {} if use_cache else None

	tr_params = {
		'batch_size': min(batch_size, len(train_params['file_paths'])), 
		'shuffle': shuffle, 
		'num_workers': num_workers, 
		'drop_last': True, 
		'collate_fn': pad_collate,
		'pin_memory': True,
		'persistent_workers': num_workers>0,
		'prefetch_factor': 2 if num_workers>0 else None
	}
	
	trainset  = ECGDataset(
		train_params['file_paths'], 
		train_params.get('class', None), 
		train_params.get('run_id', None), 
		dtype, 
		dataset,
		train_params['exclude_leads_in'],
		shared_cache=train_cache
	)
	trainset  = data.DataLoader(trainset, **tr_params)

	vl_params = {
		'batch_size': min(batch_size, len(valid_params['file_paths'])), 
		'shuffle': False, 
		'num_workers': num_workers,
		'drop_last': False,
		'collate_fn': pad_collate,
		'pin_memory': True,
		'persistent_workers': num_workers>0,
		'prefetch_factor': 2 if num_workers>0 else None
	}
	validset  = ECGDataset(
		valid_params['file_paths'], 
		valid_params['class'], 
		valid_params['run_id'], 
		dtype, 
		dataset,
		valid_params['exclude_leads_in'], 
		shared_cache=valid_cache
	)
	validset  = data.DataLoader(validset, **vl_params)

	te_params = {
		'batch_size': min(batch_size, len(test_params['file_paths'])), 
		'shuffle': False, 
		'num_workers': num_workers, 
		'drop_last': False,
		'collate_fn': pad_collate,
		'pin_memory': True,
		'persistent_workers': num_workers>0,
		'prefetch_factor': 2 if num_workers>0 else None
	}

	testset   = ECGDataset(
		test_params['file_paths'], 
		test_params['class'],
		test_params['run_id'], 
		dtype, 
		dataset,
		test_params['exclude_leads_in'], 
		shared_cache=test_cache
	)
	testset   = data.DataLoader(testset, **te_params)
	return trainset, validset, testset, manager

