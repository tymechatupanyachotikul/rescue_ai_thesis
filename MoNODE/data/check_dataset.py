import os
from turtle import pd
import numpy as np
import torch
from pathlib import Path
import itertools
import pickle 
import matplotlib.pyplot as plt
from tqdm import tqdm
import random 

def get_time_stats(base_dir, plot=False):
    
    time = [] 
    file_list = []
    for f in os.listdir(base_dir):
        if f.endswith('.npy'):
            time.append(int(f.split('_')[0][1:]))
            file_list.append(f)
    
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

root_dir = '/projects/prjs1890/uk_biobank/processed'
get_uk_bb_split(root_dir)