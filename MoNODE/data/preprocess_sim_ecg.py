import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal 
import os 
import json
import argparse 
import neurokit2 as nk
import random 

def save_ecg(file_path, save_path, target_hz=500, default_hz=500, time=10, segment_qrs=False):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(file_path, header=None)
    x = df.to_numpy()

    if segment_qrs:
        window = int(round(time/2))

        signals, info = nk.ecg_process(x[1, :], sampling_rate=1000) # use lead 2 as reference lead
        rpeaks = info["ECG_R_Peaks"]

        intervals = [(max(0, peak - window), min(x.shape[1], peak + window)) for peak in rpeaks]
        for idx, interval in enumerate(random.sample(intervals, 3)):
            save_paths = save_path.split('/')
            save_paths[-1] = str(idx) + '_' + save_paths[-1]
            cur_save_path = '/'.join(save_paths)
            np.save(cur_save_path, x[:, interval[0]: interval[1]].astype(np.float32))
    else:
        x = signal.resample_poly(x, up=target_hz, down=default_hz, axis=1)
        x = x[:, :int(time * target_hz)]
        

        mu = np.mean(x, axis=1, keepdims=True)
        sigma = np.std(x, axis=1, keepdims=True)

        x = (x - mu) / (sigma + 1e-8)

        np.save(save_path, x.astype(np.float32))


def save_params(v_param_path, a_param_path, save_target):

    params = {
        'ventricular': {}, 
        'atrial': {}
    }
    
    with open(v_param_path, "r") as f:
        for line in f:
            try:
                k, v = line.strip().split("=")
            except Exception as e:
                k, v = line.strip().split(" ")
            params['ventricular'][k.strip()] = v.strip()
    with open(a_param_path, "r") as f:
        for line in f:
            try:
                k, v = line.strip().split("=")
            except Exception as e:
                k, v = line.strip().split(" ")
            params['ventricular'][k.strip()] = v.strip()

    with open(save_target, 'w') as f:
        json.dump(params, f)

def process_split(cur_dir, split, cur_save_dir, ecg_type, target_hz, time, segment_qrs):
    cur_dir = os.path.join(cur_dir, split)
    os.makedirs(cur_save_dir, exist_ok=True)
    param_dir = 'WP2_largeDataset_ParameterFiles'

    num_samples = 0

    for run_dir in os.listdir(cur_dir):
        cur_run_dir = os.path.join(cur_dir, run_dir)
        if os.path.isdir(cur_run_dir):
            for ecg_file in os.listdir(cur_run_dir):
                if ecg_file.endswith(f'{ecg_type}.csv'):
                    _id = ecg_file.split('_')[0]
                    filename = f'{_id}_{run_dir}_preprocessed.npy'
                    save_ecg(os.path.join(cur_run_dir, ecg_file), os.path.join(cur_save_dir, filename), target_hz=target_hz, time=time, segment_qrs=segment_qrs)

                    param_dir = cur_run_dir.replace('Noise', 'ParameterFiles')
                    param_a_file = os.path.join(param_dir, f'{_id}_AtrialParameters.txt')
                    param_v_file = os.path.join(param_dir, f'{_id}_VentricularParameters.txt')

                    params_filename = f'{_id}_{run_dir}_params.json'
                    save_params(param_v_file, param_a_file, os.path.join(cur_save_dir, params_filename))

                    num_samples += 1 
    print(f'Found {num_samples} samples in {split} set')


def main(args):
    root_dir = args.root_dir 
    split = args.split 
    ecg_type = args.ecg_type 
    target_hz = args.target_hz 
    time = args.time 
    segment_qrs = args.segment_qrs

    if segment_qrs:
        time = 150

    dir_path = (
    	f"preprocessed/"
    	f"T{f'{time:.2f}'.replace('.', '_')}_f{target_hz}_{ecg_type}{'_QRS'if segment_qrs else ''}/{split}"
	)
    save_dir = os.path.join(root_dir, dir_path)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    dataset_dir = 'WP2_largeDataset_Noise'

    print('Starting preprocessing')
    target_dir = os.path.join(root_dir, dataset_dir)
    print(f'Target dir : {target_dir}')

    for dir in os.listdir(target_dir):
        cur_dir = os.path.join(target_dir, dir)
        if os.path.isdir(cur_dir):
            cur_save_dir = os.path.join(save_dir, dir)
            
            print(f'----- Processing {dir} -----')
            if dir != 'mi':
                process_split(cur_dir, split, cur_save_dir, ecg_type, target_hz, time, segment_qrs)
            else:
                for mi_dir in os.listdir(cur_dir):
                    if mi_dir != 'examples':
                        cur_save_dir = os.path.join(save_dir, mi_dir)
                        t_dir = os.path.join(cur_dir, mi_dir)

                        process_split(t_dir, split, cur_save_dir, ecg_type, target_hz, time, segment_qrs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MedalCare-XL Dataset")
    
    parser.add_argument("--root_dir", type=str, default="/projects/prjs1890/MedalCare-XL")
    
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "valid"])
    parser.add_argument("--ecg_type", type=str, default="raw", help="e.g. raw, clean")
    parser.add_argument("--target_hz", type=int, default=250)
    parser.add_argument("--time", type=int, default=10, help="Seconds of ECG to keep")
    parser.add_argument("--segment_qrs", action="store_true", help="segment QRS complex only")

    args = parser.parse_args()
    main(args)