import os
import pickle 
import numpy as np 
import matplotlib.pyplot as plt

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

for base_dir in dirs:
    get_time_stats(base_dir)