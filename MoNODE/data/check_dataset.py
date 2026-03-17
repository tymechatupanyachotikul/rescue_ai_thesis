import os 
import numpy as np 
import matplotlib.pyplot as plt

def get_time_stats(base_dir):
    
    time = [] 
    for f in os.listdir(base_dir):
        if f.endswith('.npy'):
            time.append(int(f.split('_')[0][1:]))
    
    time = np.array(time)
    print(f"Dataset: {base_dir}")
    print(f"Total samples: {len(time)}")
    print(f"Time stats - \n mean: {time.mean():.2f}\n   std: {time.std():.2f}\nmin: {time.min()}\n  max: {time.max()}")
    
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
    '/projects/prjs1890/MedalCare-XL/segments/train/atrial/median'
]

for base_dir in dirs:
    get_time_stats(base_dir)