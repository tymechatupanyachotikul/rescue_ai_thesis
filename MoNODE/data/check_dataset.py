import os 
from collections import defaultdict 
from typing import Any
import random 
import pandas as pd 

def cleanup_dataset(base_dir, csv_path):
    hash_counts: dict[str, dict[str, Any]] = defaultdict(lambda: {'count': 0, 'filename': []})

    n_files = 0
    for file_path in os.listdir(base_dir):
        if file_path.endswith('.npy'):
            hash = file_path.split('_')[1]
            hash_counts[hash]['count'] += 1
            hash_counts[hash]['filename'].append(file_path) 
            n_files += 1

    for hash, info in hash_counts.items():
        if info['count'] > 2:
            print(f"Hash {hash} appears {info['count']} times")

            random.shuffle(info['filename'])
            patient_0 = []
            patient_1 = []
            for file in info['filename']:
                if file.split('_')[-1][0] == '0':
                    patient_0.append(file)
                else:
                    patient_1.append(file)
            
            if len(patient_0) > 1:
                os.remove(os.path.join(base_dir, patient_0[0]))
            if len(patient_1) > 1:
                os.remove(os.path.join(base_dir, patient_1[0]))

    df = pd.read_csv(csv_path)
    original_length = len(df)
    df = df[~df['hash'].isin(hash_counts.keys())]

    fn = csv_path.split('/')
    fn[-1] = 'cleaned_' + fn[-1]
    fn = '/'.join(fn)
    df.to_csv(fn, index=False)

    print(f'{len([_ for _ in os.listdir(base_dir)])}/{n_files} files remaining')
    print(f'{len(df)}/{original_length} records remaining')


dirs = [
    ('/projects/prjs1890/MedalCare-XL/segments/train/ventricular/sampled', '/projects/prjs1890/MedalCare-XL/data_split/medalcare_xl_train_ventricular.csv'),
    ('/projects/prjs1890/MedalCare-XL/segments/train/atrial/sampled', '/projects/prjs1890/MedalCare-XL/data_split/medalcare_xl_train_atrial.csv'),
    ('/projects/prjs1890/MedalCare-XL/segments/valid/ventricular/sampled', '/projects/prjs1890/MedalCare-XL/data_split/medalcare_xl_valid_ventricular.csv'),
    ('/projects/prjs1890/MedalCare-XL/segments/valid/atrial/sampled', '/projects/prjs1890/MedalCare-XL/data_split/medalcare_xl_valid_atrial.csv'),
    ('/projects/prjs1890/MedalCare-XL/segments/test/ventricular/sampled', '/projects/prjs1890/MedalCare-XL/data_split/medalcare_xl_test_ventricular.csv'),
    ('/projects/prjs1890/MedalCare-XL/segments/test/atrial/sampled', '/projects/prjs1890/MedalCare-XL/data_split/medalcare_xl_test_atrial.csv'),
]
for base_dir, csv_path in dirs:
    cleanup_dataset(base_dir, csv_path)