import os 
import numpy as np 
import matplotlib.pyplot as plt

def get_time_stats(base_dir, plot=False):
    
    time = [] 
    for f in os.listdir(base_dir):
        if f.endswith('.npy'):
            time.append(int(f.split('_')[0][1:]))
    
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
    '/projects/prjs1890/MedalCare-XL/segments/train/atrial/median'
]

for base_dir in dirs:
    get_time_stats(base_dir)