import pandas as pd 
from collections import defaultdict 
import matplotlib.pyplot as plt 
import wfdb
import argparse 
import numpy as np 
import torch 
from util import filter_bandpass
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import os 
import neurokit2 as nk
import re 

def get_stats(label_path):
    df = pd.read_csv(label_path)

    n_data = len(df)
    subject_visit_dict = defaultdict(int)
    for row in df.itertuples():
        subject_visit_dict[row.subject_id] += 1 

    n_subjects = len(subject_visit_dict)
    n_visit_dict = defaultdict(int)
    for n_visit in subject_visit_dict.values():
        n_visit_dict[n_visit] += 1 

    n_visits = [_ for _ in n_visit_dict.keys()]
    freq = [_ for _ in n_visit_dict.values()]

    print(f'Total data point : {n_data}')
    print(f'Total patients with more than one visit : {n_subjects - n_visit_dict[1]} / {n_subjects}')

    n_visit_r = {}

    c = 0
    for n_visit, freq in dict(sorted(n_visit_dict.items(), reverse=True)).items():
        c += freq
        n_visit_r[n_visit] = c

    n_visit_r = dict(sorted(n_visit_r.items()))
    n_visits = [_ for _ in n_visit_r.keys()]
    freq = [_ for _ in n_visit_r.values()]

    plt.figure()
    plt.bar(n_visits, freq)
    plt.title('Frequency of number of visits')
    plt.xlabel('Number of visits')
    plt.ylabel('Frequency')
    plt.xlim((0, 40))
    plt.grid()
    plt.show()

def preprocess_single_record(file_info, save_dir):
    filename, data_dir = file_info
    data_quality = defaultdict(int)

    try:
        wave_path = os.path.join(data_dir, filename)
        data = [wfdb.rdsamp(wave_path)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))

        try:
            num_leads = min(12, data.shape[0])
            for i in range(num_leads):
                ecg_cleaned = nk.ecg_clean(data[i, :], sampling_rate=300)
                data_quality[nk.ecg_quality(ecg_cleaned,
                sampling_rate=300,
                method="zhao2018",
                approach="fuzzy")] += 1
            quality = sorted(data_quality.items(), key=lambda item: item[1], reverse=True)[0][0]
        except Exception as e:
            quality = None

        data = filter_bandpass(data, 500) 
        signal = (data - np.mean(data)) / (np.std(data) +1e-8)

        subject_id = re.search(r"/(p\d{8})/", wave_path).group(1)
        recording_id = wave_path.split('/')[-1]

        subject_dir = os.path.join(save_dir, subject_id)
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)

        torch.save(torch.FloatTensor(signal), os.path.join(save_dir, subject_id, recording_id + '.pt'))

        return True, (filename, quality)
    
    except Exception as e:
        return f"Error on {filename}: {str(e)}", None
    
def main(args):
    
    label_df = pd.read_csv(args.label_path)
    files = [(row.waveform_path, args.data_dir) for row in label_df.itertuples()]
    num_workers = max(1, cpu_count() - 2)

    func = partial(preprocess_single_record, save_dir=args.save_dir)

    print(f'Starting processing with {num_workers} workers')
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(func, files), total=len(files)))
    errors = [] 
    label_df['quality'] = None
    quality_map = {}

    for res in results:
        err, quality = res
        if err is not True and err is not None:
            errors.append(err)

        if quality is not None:
            wv_path, q = quality
            quality_map[wv_path] = str(q)

    if errors:
        print(f"\nCompleted with {len(errors)} errors.")
        print("Sample errors:", errors[:5])
    else:
        print("\nAll files processed successfully.")

    label_df['quality'] = label_df['waveform_path'].map(quality_map)
    label_df.to_csv(args.label_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIMIC Dataset")
    
    parser.add_argument("--data_dir", type=str, default="../mimic")
    parser.add_argument("--save_dir", type=str, default="../mimic_lvef")
    parser.add_argument("--label_path", type=str, default="./csv/LVEF.csv")

    args = parser.parse_args()
    main(args)