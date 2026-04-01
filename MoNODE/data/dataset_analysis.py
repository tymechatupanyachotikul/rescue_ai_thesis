import os
import json
import re
import numpy as np
import argparse
import pandas as pd
from collections import defaultdict
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

def clean_dataset():
    with open('/projects/prjs1890/MedalCare-XL/removed_anomoly_segments/metadata/remove_info_atrial.json', 'r') as f:
        remove_info = json.load(f)

    remove_dict = {
        'train': [],
        'valid': [],
        'test': []
    }
    for split, info in remove_info.items():
        print(f'------------------ {split.upper()} ------------------')
        
        remove_set = set(info['remove_files'])
        target_dir = info['dir']
        
        if not remove_set:
            print("No files targeted for removal. Skipping directory.\n")
            continue
        # OPTIMIZATION 2: Compile a single Regex pattern to check all strings at once
        # This creates a pattern like: (S66_...|S74_...|S68_...)
        pattern = re.compile('|'.join(map(re.escape, remove_set)))
        matches = defaultdict(list)

        # OPTIMIZATION 3: Use scandir instead of listdir for massive speed/memory improvements
        if os.path.exists(target_dir):
            with os.scandir(target_dir) as entries:
                for entry in entries:
                    if entry.name.endswith('.pth') and entry.is_file():
                        # Search the filename using the compiled C-level regex
                        match = pattern.search(entry.name)
                        if match:
                            matches[match.group()].append(entry.name)
                            remove_dict[split].append(os.path.join(target_dir, entry.name))

        if not matches:
            print("No matching files found in directory.\n")
        else:
            for key, files in matches.items():
                print(f"Files matching {key} ({len(files)}):")
                print(*(f"\t{f}" for f in files), sep="\n")
            print()

    with open(os.path.join('/projects/prjs1890/MedalCare-XL/removed_anomoly_segments/metadata', 'removed_files_atrial.json'), 'w') as f:
        json.dump(remove_dict, f, indent=4)

def adjust_dataset():
    classes = ['avblock', 'fam', 'iab', 'lae', 'lbbb', 'rbbb', 'sinus']
    
    for split in ['valid', 'test']:
        root_dir = f'/projects/prjs1890/MedalCare-XL/segments/{split}/ventricular/median'
        before_count = 0
        cls_count = defaultdict(int)
        file_dict = defaultdict(list)

        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if file_path.endswith('.pth'):
                data_cls = file.split('_')[-1].split('.')[0]
                if data_cls in classes:
                    cls_count[data_cls] += 1 
                    file_dict[data_cls].append(file_path)
                before_count += 1
    
        for _cls, count in cls_count.items():
            print(f"Class {_cls}: {count} samples")

        num_sinus_cls = (max(cls_count['lbbb'], cls_count['rbbb']) * 2 ) // 5 

        sinus_classes = ['avblock', 'fam', 'iab', 'lae', 'sinus']
        new_dir = os.path.join(root_dir, 'adjusted_removed')
        os.makedirs(new_dir, exist_ok=True)

        for _cls in sinus_classes:
            files = file_dict[_cls]
            random.shuffle(files)
            for file in files[num_sinus_cls:]:
                shutil.move(file, new_dir)
        
        final_cls_count = defaultdict(int)
        total_size = 0
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if file_path.endswith('.pth'):
                data_cls = file.split('_')[-1].split('.')[0]
                if data_cls in sinus_classes:
                    final_cls_count['sinus'] += 1
                else:
                    final_cls_count[data_cls] += 1
                
                total_size += 1 
        
        print(f"Final class distribution after adjustment (split {split}):")
        for cls, count in final_cls_count.items():
            print(f"  {cls}: {count}")
        print(f"Total files after adjustment: {total_size} ({before_count})")

def remove_files(seg_type):
    with open(os.path.join('/projects/prjs1890/MedalCare-XL/removed_anomoly_segments/metadata', f'removed_files_{seg_type}.json'), 'r') as f:
        remove_dict = json.load(f)
    
    for split, files in remove_dict.items():
        print(f'------------------ {split.upper()} ------------------')
        count = 0
        for file in files:
            if os.path.exists(file):
                shutil.move(file, f'/projects/prjs1890/MedalCare-XL/removed_anomoly_segments/{split}/{seg_type}/median/')
                count += 1
                print(f"Removed: {file}")
            else:
                print(f"File not found (skipped): {file}")
        
        print(f"Total files removed in {split}/{seg_type}: {count}\n")

def _plot_distribution(counts_per_split, seg_type, out_path):
    """Bar charts — one panel per split — showing per-class sample frequency."""
    splits = [s for s in ['train', 'valid', 'test'] if s in counts_per_split]
    n_splits = len(splits)

    # Collect all classes that appear across any split, sorted alphabetically
    all_classes = sorted({cls for split in splits for cls in counts_per_split[split]})
    x = np.arange(len(all_classes))
    bar_width = 0.55

    # Colour palette — one muted colour per split
    palette = {'train': '#4C72B0', 'valid': '#55A868', 'test': '#C44E52'}

    fig, axes = plt.subplots(
        1, n_splits,
        figsize=(max(6, len(all_classes) * 1.1) * n_splits, 6),
        sharey=False,
    )
    if n_splits == 1:
        axes = [axes]

    fig.patch.set_facecolor('#F8F8F8')

    for ax, split in zip(axes, splits):
        counts = [counts_per_split[split].get(cls, 0) for cls in all_classes]
        colour = palette[split]

        bars = ax.bar(x, counts, width=bar_width, color=colour, alpha=0.88,
                      linewidth=0.6, edgecolor='white', zorder=3)

        # Value labels on top of each bar
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.012,
                f'{count:,}',
                ha='center', va='bottom',
                fontsize=8.5, color='#333333', fontweight='medium',
            )

        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=35, ha='right', fontsize=10)
        ax.set_title(split.capitalize(), fontsize=13, fontweight='bold',
                     color='#222222', pad=10)
        ax.set_xlabel('Class', fontsize=11, labelpad=6)
        ax.set_ylabel('Number of samples', fontsize=11, labelpad=6)

        # Grid, spine styling
        ax.set_facecolor('#FFFFFF')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#CCCCCC', zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#BBBBBB')

        # Add a bit of headroom so labels don't get clipped
        ax.set_ylim(0, max(counts) * 1.15 if counts else 1)

    fig.suptitle(
        f'Dataset class distribution — {seg_type}',
        fontsize=15, fontweight='bold', color='#111111', y=1.01,
    )
    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    save_file = os.path.join(out_path, f'class_distribution_{seg_type}.png')
    fig.savefig(save_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved distribution plot to {save_file}")


def get_dataset_distribution(seg_type, out_path):

    ventricular_classes = ['lbbb', 'rbbb', 'mi']
    atrial_classes = ['avblock', 'fam', 'iab']

    if seg_type == 'ventricular':
        sinus_classes = ['sinus'] + atrial_classes
    else:
        sinus_classes = ['sinus'] + ventricular_classes

    counts_per_split = {}

    for split in ['train', 'valid', 'test']:
        root_dir = f'/projects/prjs1890/MedalCare-XL/segments/{split}/{seg_type}/median'
        cls_count = defaultdict(int)

        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if file_path.endswith('.pth'):
                data_cls = file.split('_')[-1].split('.')[0]
                if data_cls not in ventricular_classes + sinus_classes:
                    if seg_type == 'ventricular':
                        data_cls = '_'.join(file.split('_')[-2:]).split('.')[0]
                    else:
                        data_cls = 'mi'

                if data_cls in sinus_classes:
                    cls_count['sinus'] += 1
                else:
                    cls_count[data_cls] += 1

        counts_per_split[split] = dict(cls_count)
        for _cls, count in cls_count.items():
            print(f"Class {_cls}: {count} samples in split {split}")

    _plot_distribution(counts_per_split, seg_type, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MedalCare-XL ventricular parameters and ECG similarity.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the dataset directory containing class folders.")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save the results.")
    
    args = parser.parse_args()
    #find_anomoly_ecg(args.root_dir, args.out_dir)
    #clean_dataset()
    #remove_files('atrial')
    #adjust_dataset()

    get_dataset_distribution('atrial', args.out_dir)
    get_dataset_distribution('ventricular', args.out_dir)