import os
import json 
import numpy as np 
import argparse
import pandas as pd 
from collections import defaultdict 

def analysis(args):

    root_dir = args.root_dir
    out_dir = args.out_dir 

    cls_dict = {}
    classes = []
    cls_ecg = {}
    idx_map = {
        0: 'x_lv_ant',
        1: 'x_lv_post',
        2: 'x_lv_sept',
        3: 'x_rv_sept',
        4: 'x_rv_mod'
    }

    for cl in os.listdir(root_dir):
        cls_dir = os.path.join(root_dir, cl)
        if not os.path.isdir(cls_dir):
            continue
        cls_dict[cl] = {
            'x_lv_ant': {
                'z': [],
                'thr': [],
                'phi': [],
                'ven': [],
                'rho_eps': [],
                'time': [],
                'rho': []
            },
            'x_lv_post': {
                'z': [],
                'thr': [],
                'phi': [],
                'ven': [],
                'rho_eps': [],
                'time': [],
                'rho': []
            },
            'x_lv_sept': {
                'z': [],
                'thr': [],
                'phi': [],
                'ven': [],
                'rho_eps': [],
                'time': [],
                'rho': []
            },
            'x_rv_mod': {
                'z': [],
                'thr': [],
                'phi': [],
                'ven': [],
                'rho_eps': [],
                'time': [],
                'rho': []
            },
            'x_rv_sept': {
                'z': [],
                'thr': [],
                'phi': [],
                'ven': [],
                'rho_eps': [],
                'time': [],
                'rho': []
            }
        }
        X_stack = []
        n_per_class = 0
        for file in os.listdir(cls_dir):
            file_path = os.path.join(cls_dir, file)
            # if file_path.endswith('.json'):
            #     with open(file_path, 'r') as f:
            #         data = json.load(f)['ventricular']
            #         for i in range(0, 5):
            #             params = ['z', 'thr', 'phi', 'ven', 'rho_eps', 'time', 'rho']
            #             for param in params:
            #                 key = f'stim[{i}].{param}'
            #                 cls_dict[cl][idx_map[i]][param].append(float(data.get(key, 0)))
            if file_path.endswith('.npy'):
                # x = np.load(file_path)
                # if x.shape[1] == 76:
                #     X_stack.append(x) #lead x time 
                n_per_class += 1
        classes.append(cl)
        print(f"Class {cl}: {n_per_class} samples")

    # for cl, params in cls_dict.items():
    #     for param, sub_params in params.items():
    #         for sub_param, values in sub_params.items():
    #             cls_dict[cl][param][sub_param] = {
    #                 'mean': np.mean(values) if values else 0,
    #                 'std': np.std(values) if values else 0
    #             }

    # sim_matrix = np.zeros((len(classes), len(classes)))
    # print(classes)
    # for i, cl1 in enumerate(classes):
    #     for j, cl2 in enumerate(classes):
    #         if i == j:
    #             sim_matrix[i, j] = 1.0
    #         else:
    #             lead_sims = [] 
    #             for lead in range(12):
    #                 lead_sim = np.corrcoef(cls_ecg[cl1][lead], cls_ecg[cl2][lead])[0, 1]
    #                 if np.isnan(lead_sim):
    #                     lead_sim = 0.0
    #                 lead_sims.append(lead_sim)
    #             lead_correlations = np.mean(lead_sims)
    #             sim_matrix[i, j] = lead_correlations
    
    # with open(os.path.join(out_dir, 'cls_analysis.json'), 'w') as f:
    #     json.dump(cls_dict, f, indent=4)
    # np.save(os.path.join(out_dir, 'ecg_sim.npy'), sim_matrix)


def find_anomoly_ecg(data_split_csv, save_dir):

    df = pd.read_csv(data_split_csv)
    split = data_split_csv.split('/')[-1].split('_')[1]
    segment_type = data_split_csv.split('/')[-1].split('_')[2].split('.')[0]
    anomalous_files = []
    max_v_dist_per_lead = defaultdict(list)
    DEFAULT_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    for file_path in df['data_path']:
        ecg = pd.read_csv(file_path, header=None).to_numpy()
        max_amplitude_per_lead = np.max(np.abs(ecg), axis=1)

        if np.any(max_amplitude_per_lead > 5):
            anomalous_files.append((file_path, np.max(max_amplitude_per_lead)))

        max_amplitude_per_lead = max_amplitude_per_lead.tolist()
        for lead_idx, amplitude in enumerate(max_amplitude_per_lead):
            max_v_dist_per_lead[DEFAULT_LEADS[lead_idx]].append(amplitude)
    
    with open(os.path.join(save_dir, f'anomoly_analysis_{split}_{segment_type}.json'), 'w') as f:
        json.dump({
            'anomalous_files': anomalous_files,
            'max_v_dist_per_lead': max_v_dist_per_lead
        }, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MedalCare-XL ventricular parameters and ECG similarity.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the dataset directory containing class folders.")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save the results.")
    
    args = parser.parse_args()
    find_anomoly_ecg(args.root_dir, args.out_dir)