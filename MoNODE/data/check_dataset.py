import json
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import itertools
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import defaultdict
import shutil
from scipy import stats as scipy_stats
import re 

def get_time_stats(base_dir, anomoly_ecg_path=None, plot=False):

    if anomoly_ecg_path is not None:
        with open(anomoly_ecg_path, "rb") as f:
            anomoly_ecg = pickle.load(f)
    else:
        anomoly_ecg = []
    
    for a in anomoly_ecg:
        try:
            session_id = a.split('/')[-1].split('_')[0]
            run_id = a.split('/')[-2].split('_')[1]
            _cls = a.split('/')[-4].replace('.', '')
            anomoly_ecg.append(f'{run_id}_{session_id}_{_cls}')
        except Exception as e:
            print(f"Error processing {a}: {e}")
            
    time = [] 
    file_list = []
    anomoly_found = 0
    for f in os.listdir(base_dir):
        if f.endswith('.pth'):
            if '_'.join(f.split('_')[1:-1]) in anomoly_ecg:

                anomoly_found += 1 
                continue
            time.append(int(f.split('_')[0][1:]))
            file_list.append(f)
    
    print(f'Total anomoly ECG found : {anomoly_found}/{len(anomoly_ecg)}')
    time = np.array(time)
    total_sample = len(time)
    print(f"Dataset: {base_dir}")
    print(f"Total samples: {total_sample}")
    print(f"Time stats: \n  mean: {time.mean():.2f}\n   std: {time.std():.2f}\n min: {time.min()}\n  max: {time.max()}")

    t_1 = (np.abs(time - time.mean()) <= time.std()).sum()
    print(f'Total samples within 1 std of mean: {t_1}/{total_sample}({t_1/total_sample*100:.2f}%) ({time.mean() - time.std():.2f} - {time.mean() + time.std():.2f})')
    t_1_5 = (np.abs(time - time.mean()) <= 1.5 * time.std()).sum()
    print(f'Total samples within 1.5 std of mean: {t_1_5}/{total_sample} ({t_1_5/total_sample*100:.2f}%) ({time.mean() - 1.5 * time.std():.2f} - {time.mean() + 1.5 * time.std():.2f})')
    t_2 = (np.abs(time - time.mean()) <= 2 * time.std()).sum()
    print(f'Total samples within 2 std of mean: {t_2}/{total_sample} ({t_2/total_sample*100:.2f}%) ({time.mean() - 2 * time.std():.2f} - {time.mean() + 2 * time.std():.2f})')

    if 'atrial' in base_dir:
        t_custom   = ((time >= 20) & (time <= 75)).sum()
        t_custom_2 = ((time >= 20) & (time <= 70)).sum()
        t_custom_3 = ((time >= 20) & (time <= 60)).sum()
        
        print(f'Total samples between 20 and 75: {t_custom}/{total_sample} ({t_custom/total_sample*100:.2f}%)')
        print(f'Total samples between 20 and 70: {t_custom_2}/{total_sample} ({t_custom_2/total_sample*100:.2f}%)')
        print(f'Total samples between 20 and 60: {t_custom_3}/{total_sample} ({t_custom_3/total_sample*100:.2f}%)')
    elif 'ventricular' in base_dir:
        t_custom   = ((time >= 160) & (time <= 250)).sum()
        t_custom_2 = ((time >= 160) & (time <= 240)).sum()
        t_custom_3 = ((time >= 160) & (time <= 230)).sum()
        
        print(f'Total samples between 160 and 250: {t_custom}/{total_sample} ({t_custom/total_sample*100:.2f}%)')
        print(f'Total samples between 160 and 240: {t_custom_2}/{total_sample} ({t_custom_2/total_sample*100:.2f}%)')
        print(f'Total samples between 160 and 230: {t_custom_3}/{total_sample} ({t_custom_3/total_sample*100:.2f}%)')

    with open(os.path.join('/home/tchatupanyacho/rescue_ai_thesis/results/ecg_segments', '_'.join(base_dir.split('/')[-3:]) + '_time.pkl'), "wb") as f:
        pickle.dump(file_list, f)

    if plot:
        fname = '_'.join(base_dir.split('/')[-3:]) + '_time.png'
        plt.figure(figsize=(8, 5))
        plt.hist(time, color='skyblue', edgecolor='black')

        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(fname)
        plt.close()

dirs = [
    '/projects/prjs1890/MedalCare-XL/segments/train/ventricular/sampled',
    '/projects/prjs1890/MedalCare-XL/segments/train/atrial/sampled',
    '/projects/prjs1890/MedalCare-XL/segments/train/ventricular/median',
    '/projects/prjs1890/MedalCare-XL/segments/train/atrial/median',
    '/projects/prjs1890/MedalCare-XL/segments/valid/ventricular/sampled',
    '/projects/prjs1890/MedalCare-XL/segments/valid/atrial/sampled',
    '/projects/prjs1890/MedalCare-XL/segments/valid/ventricular/median',
    '/projects/prjs1890/MedalCare-XL/segments/valid/atrial/median'
    '/projects/prjs1890/MedalCare-XL/segments/test/ventricular/sampled',
    '/projects/prjs1890/MedalCare-XL/segments/test/atrial/sampled',
    '/projects/prjs1890/MedalCare-XL/segments/test/ventricular/median',
    '/projects/prjs1890/MedalCare-XL/segments/test/atrial/median'
]

def convert_npy_to_pth(base_dir):
    base_path = Path(base_dir)
    splits = ['train', 'valid', 'test']
    beat_type  = ['ventricular', 'atrial']
    sample_type = ['sampled', 'median']

    combinations = itertools.product(splits, beat_type, sample_type)

    for split, beat, sample in combinations:
        cur_dir = base_path / split / beat / sample

        if not cur_dir.exists():
            continue
        
        npy_files = list(cur_dir.glob('*.npy'))
        total_files = len(npy_files)

        print(f'Processing directory: {cur_dir}')
    
        processed = 0
        desc_label = f"{split}/{beat}/{sample}"
        for npy_file in tqdm(npy_files, desc=desc_label):
            pth_file = npy_file.with_suffix('.pth')
            try:
                data = np.load(npy_file)
                tensor_data = torch.from_numpy(data)
                
                torch.save(tensor_data, pth_file)
                
                if pth_file.exists() and pth_file.stat().st_size > 0:
                    npy_file.unlink() 
                    processed += 1
                else:
                    tqdm.write(f"Warning: {pth_file} failed to save properly. Keeping original.")
                    
            except Exception as e:
                tqdm.write(f"Error processing {npy_file.name}: {e}")
                if pth_file.exists():
                    pth_file.unlink()

        tqdm.write(f'Finished processing {cur_dir}. Successfully converted {processed}/{total_files} files.')

def get_uk_bb_split(root_dir):
    train_split = 0.7 
    val_split = 0.15 

    root_dir = Path(root_dir)
    npy_files = list(root_dir.glob('*.npy'))
    random.seed(42)
    random.shuffle(npy_files)

    num_patients = len(npy_files)
    train_end = int(num_patients * train_split)
    val_end = train_end + int(num_patients * val_split)

    train_ids = npy_files[:train_end]
    val_ids = npy_files[train_end:val_end]
    test_ids = npy_files[val_end:]

    train_paths = {'data_path': [root_dir / str(f) for f in train_ids]}
    val_paths = {'data_path': [root_dir /  str(f) for f in val_ids]}
    test_paths = {'data_path': [root_dir / str(f) for f in test_ids]}  

    df_train = pd.DataFrame(train_paths)
    df_train.to_csv("/projects/prjs1890/uk_biobank/data_split/ukbb_train.csv", index=False)

    df_val = pd.DataFrame(val_paths)
    df_val.to_csv("/projects/prjs1890/uk_biobank/data_split/ukbb_valid.csv", index=False)

    df_test = pd.DataFrame(test_paths)
    df_test.to_csv("/projects/prjs1890/uk_biobank/data_split/ukbb_test.csv", index=False)

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

def get_mimic_split(root_dir, dest_dir, lvef_csv):

    df = pd.read_csv(lvef_csv)

    patient_id_dict = defaultdict(lambda: defaultdict(list))
    ecg_save_dir = os.path.join(dest_dir, 'raw')
    for row in tqdm(df.itertuples(), total=len(df), desc="Moving Files & Grouping"):
        # file_path = os.path.join(root_dir, str(row.waveform_path))
        # if not os.path.exists(file_path + '.dat') or not os.path.exists(file_path + '.hea'):
        #     if os.path.exists(os.path.join(ecg_save_dir, os.path.basename(file_path) + '.dat')) and os.path.exists(os.path.join(ecg_save_dir, os.path.basename(file_path) + '.hea')):
        #         pass
        #     else:
        #         print(f"Warning: File {file_path} does not exist. Skipping.")
        #         continue
        # else:
        #     if not os.path.exists(os.path.join(ecg_save_dir, os.path.basename(file_path) + '.dat')):
        #         shutil.move(file_path + '.dat', ecg_save_dir)
        #     if not os.path.exists(os.path.join(ecg_save_dir, os.path.basename(file_path) + '.hea')):
        #         shutil.move(file_path + '.hea', ecg_save_dir)

        filename = str(row.waveform_path).split('/')[3][1:]
        file_path = os.path.join(ecg_save_dir, filename)

        patient_id_dict[row.subject_id]['file_path'].append(file_path)
        patient_id_dict[row.subject_id]['lvef'].append(row.LVEF)
        patient_id_dict[row.subject_id]['class'].append(row[3])
    
    n_train = int(len(df) * 0.8)
    n_val = int(len(df) * 0.1)

    patient_ids = list(patient_id_dict.keys())
    random.seed(42)
    random.shuffle(patient_ids)

    train_paths = {'data_path': [], 'lvef': [], 'class': []}
    val_paths = {'data_path': [], 'lvef': [], 'class': []}
    test_paths = {'data_path': [], 'lvef': [], 'class': []}

    for paths in tqdm(patient_id_dict.values(), desc="Splitting Dataset"):
        if len(train_paths['data_path']) < n_train:
            train_paths['data_path'].extend(paths['file_path'])
            train_paths['lvef'].extend(paths['lvef'])
            train_paths['class'].extend(paths['class'])
        elif len(val_paths['data_path']) < n_val:
            val_paths['data_path'].extend(paths['file_path'])
            val_paths['lvef'].extend(paths['lvef'])
            val_paths['class'].extend(paths['class'])
        else:
            test_paths['data_path'].extend(paths['file_path'])
            test_paths['lvef'].extend(paths['lvef'])
            test_paths['class'].extend(paths['class'])
    
    out_dir = os.path.join(dest_dir, 'data_split')
    os.makedirs(out_dir, exist_ok=True)

    df_train = pd.DataFrame(train_paths)
    df_train.to_csv(os.path.join(out_dir, "mimic-iv_train.csv"), index=False)

    df_val = pd.DataFrame(val_paths)
    df_val.to_csv(os.path.join(out_dir, "mimic-iv_valid.csv"), index=False)

    df_test = pd.DataFrame(test_paths)
    df_test.to_csv(os.path.join(out_dir, "mimic-iv_test.csv"), index=False)

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

def find_anomoly_ecg(csv_path):

    df = pd.read_csv(csv_path)
    anomoly_ecg = []
    for row in df.itertuples():
        file_path = str(row.data_path)
        ecg = pd.read_csv(file_path, header=None).to_numpy()
        if np.abs(ecg).max() > 10:
            anomoly_ecg.append(file_path) 

    return anomoly_ecg

def plot_ecg(file_path, root_dir):
    type = ''
    sample_type = ''
    if 'atrial' in file_path:
        type = 'atrial'
    elif 'ventricular' in file_path:
        type = 'ventricular'

    if 'upper' in file_path:
        sample_type = 'upper'
    elif 'lower' in file_path:
        sample_type = 'lower'
    
    save_dir = os.path.join(root_dir, type, sample_type)
    os.makedirs(save_dir, exist_ok=True)
    
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            file_paths = pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            file_paths = json.load(f)
    
    f = 500
    for file in file_paths:
        try:
            ecg = pd.read_csv(file, header=None).to_numpy()
            
            t = ecg.shape[1]
            l = ecg.shape[0]

            fig, axes = plt.subplots(l, 1, sharex=True)
            time = np.arange(t) / f

            for j in range(l):
                ax = axes[j]
                ax.plot(time, ecg[j, :], linewidth=0.7)

                if j < l - 1:
                    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            plt.xlabel("Time (seconds)", fontsize=12)
            plt.savefig(os.path.join(root_dir,os.path.basename(file).replace('.csv', '.png')), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error processing {file}: {e}")

def remove_anomoly_ecg(base_dir, remove_dir, anomoly_ecg = []):
    
    for a in anomoly_ecg:
        try:
            session_id = a.split('/')[-1].split('_')[0]
            run_id = a.split('/')[-2].split('_')[1]
            _cls = a.split('/')[-4].replace('.', '')
            anomoly_ecg.append(f'{run_id}_{session_id}_{_cls}')
        except Exception as e:
            pass

    def check_range(time, type):
        if type == 'atrial':
            return 30 <= time <= 70
        elif type == 'ventricular':
            return 150 <= time <= 250

    if 'atrial' in base_dir:
        type = 'atrial'
    elif 'ventricular' in base_dir:
        type = 'ventricular'

    total_ecg = 0
    for f in os.listdir(base_dir):
        if f.endswith('.pth'):
            total_ecg += 1
            if '_'.join(f.split('_')[1:-1]) in anomoly_ecg or check_range(int(f.split('_')[0][1:]), type) == False:
                 filename = os.path.join(base_dir, f)
                 shutil.move(filename, os.path.join(remove_dir, f))

    remaining = 0
    for f in os.listdir(base_dir):
        if f.endswith('.pth'):
            remaining += 1 

    print(f'Total ECG: {total_ecg} | Remaining ECG: {remaining} | Removed ECG: {total_ecg - remaining}')

def get_filepath(uid, split):

    splits = uid.split('_')
    run_id = splits[0]
    session = splits[1]

    if len(splits) > 3:
        _cls = 'mi'
        subclass = '_'.join(splits[2:]) 
        filepath = f'/projects/prjs1890/MedalCare-XL/WP2_largeDataset_ParameterFiles/{_cls}/{subclass}/{split}/run_{run_id}'
    else:
        _cls = splits[2]
        filepath = f'/projects/prjs1890/MedalCare-XL/WP2_largeDataset_ParameterFiles/{_cls}/{split}/run_{run_id}'
    
    return (os.path.join(filepath, f'{session}_VentricularParameters.txt'), os.path.join(filepath, f'{session}_AtrialParameters.txt'))

def update_metadata(metadata_path, split):
    atrial_keys = ['cv_t.BulkTissue', 'cv_t.CristaTerminalis', 'cv_t.PectinateMuscles', 'cv_t.BachmannsBundle', 'cv_t.InferiorIsthmus',
                   'ar.BulkTissue', 'ar.CristaTerminalis', 'ar.PectinateMuscles', 'ar.BachmannsBundle', 'ar.InferiorIsthmus']
    ventricular_keys = [
        "cv.rvmyo_s_r",
        "cv.lvmyo_s_r",
        "cv.lvmyo_n_r",
        "cv.lvendo_s_r",
        "cv.rvmyo_n_r",
        "cv.rvendo_n_r",
        "cv.lvmyo_f",
        "cv.lvendo_n_r",
        "cv.lvendo_f",
        "cv.rvendo_s_r",
        "cv.rvmyo_f",
        "cv.rvendo_f",
    ]

    with open(metadata_path, 'r') as f:
        metadata = json.load(f) 

    for uid in metadata.keys():
        v_file, a_file = get_filepath(uid, split)

        with open(v_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line_split = line.split('=')
                param = line_split[0].strip()
                if param in ventricular_keys:
                    value = float(line_split[1].strip().replace('mm/s', '')) 
                    metadata[uid]['labels'][param] = value

        with open(a_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line_split = line.split('=')
                param = line_split[0].strip()
                if param in atrial_keys:
                    value = float(line_split[1].strip().replace('mm/s', '')) 
                    metadata[uid]['labels'][param] = value


    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def remove_anomaly_test(metadata_path, error_path, split, anomaly_dir, root_dir):

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    with open(error_path, 'r') as f:
        error_dict = json.load(f)

    # ── Collect (uid, time) pairs for each segment type ──────────────────────
    atrial_time = []
    ventricular_time = []

    for uid, info in metadata.items():
        seg_lengths = info.get('segment_lengths', {})
        if 'atrial' in seg_lengths:
            t = seg_lengths['atrial']
            if isinstance(t, list):
                t = t[0]
            atrial_time.append((uid, int(t)))
        if 'ventricular' in seg_lengths:
            t = seg_lengths['ventricular']
            if isinstance(t, list):
                t = t[0]
            ventricular_time.append((uid, int(t)))

    # ── Two-tailed z-test: flag UIDs whose segment length is a statistical
    #    outlier (p < 0.05).  Cutoff times are mean ± z_crit * std. ──────────
    def find_anomalies(time_list: list[tuple[str, int]]):
        if len(time_list) < 2:
            return [], None, None

        uids  = [u for u, _ in time_list]
        times = np.array([t for _, t in time_list], dtype=float)

        mean  = times.mean()
        std   = times.std()

        # z_critical for two-tailed α = 0.05: norm.ppf(0.975) ≈ 1.96
        z_crit  = scipy_stats.norm.ppf(0.975)
        lower   = float(mean - z_crit * std)
        upper   = float(mean + z_crit * std)

        z_scores = (times - mean) / (std + 1e-8)
        p_values = 2 * scipy_stats.norm.sf(np.abs(z_scores))

        anomaly_uids = [uid for uid, p in zip(uids, p_values) if p < 0.05]
        return anomaly_uids, lower, upper

    atrial_uids,      atrial_lower,      atrial_upper      = find_anomalies(atrial_time)
    ventricular_uids, ventricular_lower, ventricular_upper = find_anomalies(ventricular_time)

    # ── Write anomaly summary into error_dict ─────────────────────────────────
    error_dict['anomaly'] = {
        'atrial': {
            'count':        len(atrial_uids),
            'uid':          atrial_uids,
            'cutoff_time':  (atrial_lower, atrial_upper),
        },
        'ventricular': {
            'count':        len(ventricular_uids),
            'uid':          ventricular_uids,
            'cutoff_time':  (ventricular_lower, ventricular_upper),
        },
    }

    print(f"[{split}] Anomalous UIDs — atrial: {len(atrial_uids)}, "
          f"ventricular: {len(ventricular_uids)}")
    print(f"  Atrial    cutoff: [{atrial_lower:.1f}, {atrial_upper:.1f}]")
    print(f"  Ventricular cutoff: [{ventricular_lower:.1f}, {ventricular_upper:.1f}]")

    # ── Union of anomalous UIDs across both segment types ────────────────────
    anomaly_uid_set = set(atrial_uids) | set(ventricular_uids)
    print(f"  Combined unique anomalous UIDs: {len(anomaly_uid_set)}")

    os.makedirs(anomaly_dir, exist_ok=True)

    # ── Move matching files from all three seg_type/median dirs ──────────────
    seg_types     = ['atrial', 'ventricular', 'whole']
    total_moved     = 0
    total_remaining = 0

    for seg_type in seg_types:
        seg_dir = os.path.join(root_dir, split, seg_type, 'median')
        if not os.path.isdir(seg_dir):
            print(f"  Skipping {seg_dir} (directory not found)")
            continue

        moved     = 0
        remaining = 0
        for fname in os.listdir(seg_dir):
            if not fname.endswith('.pth'):
                continue
            uid = os.path.splitext(fname)[0]
            if uid in anomaly_uid_set:
                new_name = f"{uid}_{seg_type}.pth"
                shutil.move(
                    os.path.join(seg_dir, fname),
                    os.path.join(anomaly_dir, new_name),
                )
                moved += 1
            else:
                remaining += 1

        print(f"  [{split}/{seg_type}/median] moved: {moved}, remaining: {remaining}")
        total_moved     += moved
        total_remaining += remaining

    error_dict['anomaly']['files_moved']     = total_moved
    error_dict['anomaly']['files_remaining'] = total_remaining

    # ── Persist updated error_dict ────────────────────────────────────────────
    with open(error_path, 'w') as f:
        json.dump(error_dict, f, indent=2)

    print(f"\nTotal files moved: {total_moved} | Total remaining: {total_remaining}")
    print(f"Updated error_dict saved to {error_path}")



def remove_anomaly_train(metadata_path, error_path, split, anomaly_dir, root_dir):
    """Identify statistically anomalous segment lengths per seg-type via two-tailed
    z-test (p < 0.05), move only the matching seg-type files to *anomaly_dir*, and
    write rich statistics back into the error JSON.

    Atrial anomalies  → files moved from {root_dir}/{split}/atrial/median/ only.
    Ventricular       → files moved from {root_dir}/{split}/ventricular/median/ only.
    Whole             → untouched.
    """

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    with open(error_path, 'r') as f:
        error_dict = json.load(f)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _time_stats(times: np.ndarray) -> dict:
        """Descriptive statistics for an array of segment lengths."""
        q1, q3 = np.percentile(times, [25, 75])
        return {
            'mean':   float(times.mean()),
            'std':    float(times.std()),
            'min':    int(times.min()),
            'max':    int(times.max()),
            'median': float(np.median(times)),
            'iqr':    float(q3 - q1),
            'q1':     float(q1),
            'q3':     float(q3),
            'n':      int(len(times)),
        }

    def find_anomalies(time_list: list[tuple[str, int]]):
        """Two-tailed z-test.  Returns (anomaly_records, lower_cutoff, upper_cutoff).

        anomaly_records : list of (uid, time, p_value) for every outlier.
        Cutoff          : mean ± z_crit * std  where z_crit ≈ 1.96 (α = 0.05).
        """
        if len(time_list) < 2:
            return [], None, None

        uids  = [u for u, _ in time_list]
        times = np.array([t for _, t in time_list], dtype=float)

        mean = times.mean()
        std  = times.std()

        z_crit = scipy_stats.norm.ppf(0.975)   # ≈ 1.96
        lower  = float(mean - z_crit * std)
        upper  = float(mean + z_crit * std)

        z_scores = (times - mean) / (std + 1e-8)
        p_values = 2 * scipy_stats.norm.sf(np.abs(z_scores))

        anomaly_records = [
            (uid, int(t), float(p))
            for uid, t, p in zip(uids, times, p_values)
            if p < 0.05
        ]
        return anomaly_records, lower, upper

    def count_pth(directory: str) -> int:
        if not os.path.isdir(directory):
            return 0
        return sum(1 for f in os.listdir(directory) if f.endswith('.pth'))

    # ── Collect (uid, time) pairs ─────────────────────────────────────────────
    atrial_time      = []
    ventricular_time = []

    for uid, info in metadata.items():
        seg_lengths = info.get('segment_lengths', {})
        if 'atrial' in seg_lengths:
            t = seg_lengths['atrial']
            if isinstance(t, list):
                t = t[0]
            atrial_time.append((uid, int(t)))
        if 'ventricular' in seg_lengths:
            t = seg_lengths['ventricular']
            if isinstance(t, list):
                t = t[0]
            ventricular_time.append((uid, int(t)))

    # ── Segment counts before any moves ───────────────────────────────────────
    whole_dir       = os.path.join(root_dir, split, 'whole',       'median')
    atrial_dir      = os.path.join(root_dir, split, 'atrial',      'median')
    ventricular_dir = os.path.join(root_dir, split, 'ventricular', 'median')

    atrial_total      = count_pth(atrial_dir)
    ventricular_total = count_pth(ventricular_dir)
    whole_total       = count_pth(whole_dir)

    print(f"[{split}] Files before removal:")
    print(f"  atrial:      {atrial_total}")
    print(f"  ventricular: {ventricular_total}")
    print(f"  whole:       {whole_total}  (untouched)")

    # ── Statistical testing ───────────────────────────────────────────────────
    atrial_anomalies,      atrial_lower,      atrial_upper      = find_anomalies(atrial_time)
    ventricular_anomalies, ventricular_lower, ventricular_upper = find_anomalies(ventricular_time)

    atrial_all_times      = np.array([t for _, t in atrial_time],      dtype=float)
    ventricular_all_times = np.array([t for _, t in ventricular_time], dtype=float)

    # ── Build per-seg-type result dict ────────────────────────────────────────

    def _build_seg_result(anomaly_records, all_times, lower, upper,
                          seg_type, seg_dir, anomaly_uid_set):
        """Move files for *seg_type* only and return the stats dict."""
        os.makedirs(anomaly_dir, exist_ok=True)

        moved     = 0
        remaining = 0
        for fname in (os.listdir(seg_dir) if os.path.isdir(seg_dir) else []):
            if not fname.endswith('.pth'):
                continue
            uid = os.path.splitext(fname)[0]
            if uid in anomaly_uid_set:
                new_name = f"{uid}_{seg_type}.pth"
                shutil.move(
                    os.path.join(seg_dir, fname),
                    os.path.join(anomaly_dir, new_name),
                )
                moved += 1
            else:
                remaining += 1

        # Collect rich per-removed-sample info
        removed_details = []
        for uid, t, p in anomaly_records:
            entry = metadata.get(uid, {})
            labels = entry.get('labels', {})
            removed_details.append({
                'uid':        uid,
                'patient_id': labels.get('patient_id', uid.split('_')[0]),
                'class':      labels.get('class', None),
                'time':       t,
                'p_value':    round(p, 6),
                'side':       'low' if t < lower else 'high',
            })

        # Class distribution of removed samples
        class_dist: dict = {}
        for d in removed_details:
            cls = d['class'] or 'unknown'
            class_dist[cls] = class_dist.get(cls, 0) + 1

        pct_removed = round(100 * moved / max(int(all_times.size), 1), 2)

        return {
            'time_stats':        _time_stats(all_times),
            'cutoff_time':       (lower, upper),
            'count':             moved,
            'percent_removed':   pct_removed,
            'files_moved':       moved,
            'files_remaining':   remaining,
            'class_distribution_removed': class_dist,
            'removed':           removed_details,
        }

    atrial_uid_set      = {uid for uid, _, _ in atrial_anomalies}
    ventricular_uid_set = {uid for uid, _, _ in ventricular_anomalies}

    print(f"\n[{split}] Statistical anomalies (p < 0.05):")
    print(f"  atrial      [{atrial_lower:.1f}, {atrial_upper:.1f}]: "
          f"{len(atrial_uid_set)} / {len(atrial_time)}")
    print(f"  ventricular [{ventricular_lower:.1f}, {ventricular_upper:.1f}]: "
          f"{len(ventricular_uid_set)} / {len(ventricular_time)}")

    atrial_result      = _build_seg_result(
        atrial_anomalies, atrial_all_times,
        atrial_lower, atrial_upper,
        'atrial', atrial_dir, atrial_uid_set,
    )
    ventricular_result = _build_seg_result(
        ventricular_anomalies, ventricular_all_times,
        ventricular_lower, ventricular_upper,
        'ventricular', ventricular_dir, ventricular_uid_set,
    )

    # ── Assemble error_dict entry ─────────────────────────────────────────────
    error_dict['anomaly_train'] = {
        'summary': {
            'atrial_total_before':      atrial_total,
            'ventricular_total_before': ventricular_total,
            'whole_total':              whole_total,
            'atrial_remaining':         atrial_result['files_remaining'],
            'ventricular_remaining':    ventricular_result['files_remaining'],
            'total_files_moved':        atrial_result['files_moved'] + ventricular_result['files_moved'],
        },
        'atrial':      atrial_result,
        'ventricular': ventricular_result,
    }

    # ── Human-readable summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ANOMALY REMOVAL SUMMARY  [{split}]")
    print(f"{'='*60}")
    for seg, res in [('atrial', atrial_result), ('ventricular', ventricular_result)]:
        st = res['time_stats']
        lo, hi = res['cutoff_time']
        print(f"\n  {seg.upper()}")
        print(f"    Samples      : {st['n']}")
        print(f"    Time — mean±std : {st['mean']:.1f} ± {st['std']:.1f}  "
              f"[{st['min']}, {st['max']}]  median={st['median']:.1f}  IQR={st['iqr']:.1f}")
        print(f"    Cutoff       : [{lo:.2f}, {hi:.2f}]")
        print(f"    Removed      : {res['files_moved']} ({res['percent_removed']}%)")
        print(f"    Remaining    : {res['files_remaining']}")
        if res['class_distribution_removed']:
            print(f"    Removed class dist : {res['class_distribution_removed']}")
    print(f"\n  WHOLE  : {whole_total} files (untouched)")
    print(f"  Total moved : {error_dict['anomaly_train']['summary']['total_files_moved']}")
    print(f"{'='*60}\n")

    with open(error_path, 'w') as f:
        json.dump(error_dict, f, indent=2)

    print(f"Updated error_dict saved to {error_path}")

def change_name(root_dir):
    """Rename all .pth files under root_dir by stripping the first underscore-delimited
    component from the filename.

    Example: T50_S65_000029_LCX_03_ant.pth  →  S65_000029_LCX_03_ant.pth

    Files whose name does not contain an underscore are left untouched.
    Walks all subdirectories recursively.
    """
    renamed  = 0
    skipped  = 0
    conflict = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith('.pth'):
                continue

            parts = os.path.splitext(fname)[0].split('_', 1)
            if len(parts) < 2:
                skipped += 1
                continue

            new_fname = parts[1] + '.pth'
            src = os.path.join(dirpath, fname)
            dst = os.path.join(dirpath, new_fname)

            if os.path.exists(dst):
                print(f"  Conflict — destination already exists, skipping: {dst}")
                conflict += 1
                continue

            os.rename(src, dst)
            renamed += 1

    print(f"Done. Renamed: {renamed} | Skipped (no underscore): {skipped} | Conflicts: {conflict}")

def change_name_2(root_dir):

    renamed  = 0
    skipped  = 0
    conflict = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith('.pth'):
                continue

            stem = os.path.splitext(fname)[0]
            if len(stem.split('_')) < 4:
                skipped += 1
                continue
            stem = re.sub(r'_(\d)(\d)(?=(_|$))', r'_\1.\2', stem)
            new_fname = stem + '.pth'
            src = os.path.join(dirpath, fname)
            dst = os.path.join(dirpath, new_fname)

            if os.path.exists(dst):
                print(f"  Conflict — destination already exists, skipping: {dst}")
                conflict += 1
                continue

            os.rename(src, dst)
            renamed += 1

    print(f"Done. Renamed: {renamed} | Skipped (no underscore): {skipped} | Conflicts: {conflict}")


def refine_metadata_labels(metadata_path, split):
    """Remove unwanted label keys and add APD ventricular keys for each UID.

    Keys removed from labels:
        ar.BulkTissue, ar.PectinateMuscles,
        cv.lvendo_f, cv.lvmyo, cv.lvymyo_n_r, cv.lvmyo_s_r,
        cv.rvendo_f, cv.rvymyo_f, cv.rvmyo_n_r, cv.rvmyo_s_r

    Keys added from VentricularParameters.txt:
        APD.min, APD.v_d, APD.z_d, APD.max
    """
    KEYS_TO_REMOVE = {
        'ar.BulkTissue',
        'ar.PectinateMuscles',
        'cv.lvendo_f',
        'cv.lvmyo',
        'cv.lvmyo_n_r',
        'cv.lvmyo_s_r',
        'cv.lvmyo_f',
        'cv.rvendo_f',
        'cv.rvymyo_f',
        'cv.rvmyo_n_r',
        'cv.rvmyo_f',
        'cv.rvmyo_s_r',
        'cv.lvendo_s_r',
        'ar.InferiorIsthmus',
        'cv.rvendo_n_r',
        'cv.rvendo_s_r',
        'cv.lvendo_n_r'
    }
    APD_KEYS = {'APD.min', 'APD.v_d', 'APD.z_d', 'APD.max'}

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    for uid, entry in metadata.items():
        labels = entry.get('labels', {})

        # Remove unwanted keys
        for key in KEYS_TO_REMOVE:
            labels.pop(key, None)

        # Add APD keys from VentricularParameters.txt
        v_file, _ = get_filepath(uid, split)
        try:
            with open(v_file, 'r') as f:
                for line in f:
                    if '=' not in line:
                        continue
                    param, _, value_str = line.partition('=')
                    param = param.strip()
                    if param in APD_KEYS:
                        try:
                            labels[param] = float(value_str.strip().replace('mm/s', ''))
                        except ValueError:
                            pass
        except FileNotFoundError:
            print(f"Warning: VentricularParameters.txt not found for uid={uid} — APD keys skipped")

        entry['labels'] = labels

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"refine_metadata_labels complete: {len(metadata)} UIDs processed → {metadata_path}")

def fix_age(root_dir):

    for split in ['train', 'valid', 'text']:
        data_path = os.path.join(root_dir, f'{split}_metadata')
        with open(data_path, 'r') as f:
            data = json.load(f)

        for uid, info in data.items():
            age = info['labels'].get('age')

            if isinstance(age, (int, float)):
                age = 2026 - age 

                data[uid]['labels']['age'] = age
    
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=4)


fix_age('/home/tchatupanyacho/project/uk_biobank/segments')