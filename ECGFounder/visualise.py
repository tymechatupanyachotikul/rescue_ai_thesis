from collections import defaultdict
import argparse
import json
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt

def save_umap_reduction(split, seg_type, root_dir):
    atrial_classes = ['avblock', 'fam', 'iab', 'lae']
    ventricular_classes = ['mi', 'lbbb', 'rbbb']
    all_classes = atrial_classes + ventricular_classes + ['sinus']
    m = None
    if split == 'all':
        z0, m0 = [], []
        labels = defaultdict(list)
        for split in ['train', 'test', 'valid']:
            with open(os.path.join(root_dir, f'latent_meta_dict_{split}.json'), 'r') as f:
                meta_dict = json.load(f)

            latents_info = np.load(os.path.join(root_dir, f'latent_tensors_{split}.npz'))
            _z0 = latents_info['z0']
            if 'm' in latents_info:
                _m = latents_info['m']
                m0.append(_m)
            z0.append(_z0)
            
            for info in meta_dict:
                for key, value in info['labels'].items():
                    if value in atrial_classes and seg_type == 'ventricular':
                        value = 'sinus'
                    elif seg_type == 'atrial':
                        if value not in all_classes:
                            print(f'{value} - mi')
                            value = 'mi'
                        if value in ventricular_classes:
                            value = 'sinus'
                    labels[key].append(value)
        z0 = np.concatenate(z0, axis=0)
        if m0:
            m = np.concatenate(m0, axis=0)
    else:
        with open(os.path.join(root_dir, f'latent_meta_dict_{split}.json'), 'r') as f:
            meta_dict = json.load(f)

        latents_info = np.load(os.path.join(root_dir, f'latent_tensors_{split}.npz'))
        z0 = latents_info['z0']
        if 'm' in latents_info:
            m = latents_info['m']
        labels = defaultdict(list)

        for info in meta_dict:
            for key, value in info['labels'].items():
                labels[key].append(value)

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42,
        verbose=True
    )

    embeddings = {}

    latent_dict = {'z0': z0}
    if m is not None:
        latent_dict['m'] = m
        latent_dict['combined'] = np.concatenate([z0, m], axis=-1)
    print(f'Performing UMAP reduction')
    for key, latent in latent_dict.items():
        embeddings[key] = reducer.fit_transform(latent)
    print(f'UMAP reduction completed')

    embeddings_save_path = os.path.join(root_dir, f'umap_embeddings_{split}.npz')
    np.savez(embeddings_save_path, **{k: v for k, v in embeddings.items()})
    print(f'Embeddings saved: {embeddings_save_path}')

    BINARY_LABELS = {'afib', 'mi'}

    for key, value in labels.items():
        out_labels = []
        indices = []
        for i, v in enumerate(value):
            if isinstance(v, float) and not math.isnan(v):
                out_labels.append(v)
                indices.append(i)
            elif isinstance(v, (int, bool)):
                out_labels.append(float(v))
                indices.append(i)
            elif isinstance(v, str):
                # Optionally filter out explicit string "nan"s if they exist in your data
                if v.lower() not in ['nan', 'none', '']:
                    out_labels.append(v)
                    indices.append(i)

        is_binary = key.lower() in BINARY_LABELS
        is_categorical = isinstance(out_labels[0], str)
        plot_dir = os.path.join(root_dir, f'plots')
        os.makedirs(plot_dir, exist_ok=True)

        for latent_name, latent_out in embeddings.items():
            out_embedding = latent_out[indices]
            save_path = os.path.join(plot_dir, f'embeddings_{key}_{latent_name}_{split}.png')

            sns.set_context("talk")
            fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
            if is_categorical:
                out_labels_arr = np.array(out_labels)
                unique_classes = np.unique(out_labels_arr)
                
                palette = sns.color_palette("husl", len(unique_classes))
                
                for cls_val, color in zip(unique_classes, palette):
                    mask = out_labels_arr == cls_val
                    ax.scatter(
                        out_embedding[mask, 0],
                        out_embedding[mask, 1],
                        color=color,
                        label=str(cls_val),
                        s=15,
                        alpha=0.7,
                        edgecolors='none'
                    )
                ax.legend(title=key, markerscale=2, frameon=True, bbox_to_anchor=(1.04, 1), loc="upper left")
            elif is_binary:
                out_labels_arr = np.array(out_labels)
                colors = ['#2196F3', '#F44336']
                class_names = ['Negative (0)', 'Positive (1)']
                for cls_val, color, cls_name in zip([0.0, 1.0], colors, class_names):
                    mask = out_labels_arr == cls_val
                    ax.scatter(
                        out_embedding[mask, 0],
                        out_embedding[mask, 1],
                        c=color,
                        label=cls_name,
                        s=15,
                        alpha=0.7,
                        edgecolors='none'
                    )
                ax.legend(title=key, markerscale=2, frameon=True)
            else:
                sc = ax.scatter(
                    out_embedding[:, 0],
                    out_embedding[:, 1],
                    c=out_labels,
                    cmap='Spectral',
                    s=15,
                    alpha=0.7,
                    edgecolors='none'
                )
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(key, fontweight='bold', rotation=270, labelpad=20)

            ax.set_xlabel('UMAP Dimension 1', fontweight='bold')
            ax.set_ylabel('UMAP Dimension 2', fontweight='bold')
            ax.set_title(f'{key} — {latent_name} ({split})', fontweight='bold')

            sns.despine(ax=ax)
            ax.grid(True, linestyle='--', alpha=0.15)

            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {save_path}')

def load_and_plot_umap(file_path):
    data = np.load(file_path)

    embedding_2d = data['embeddings']
    labels = data['labels']

    print(f"Loaded {embedding_2d.shape[0]} samples.")

    sns.set_context("talk")
    plt.figure(figsize=(12, 9), dpi=150)

    sc = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=labels, 
        cmap='Spectral', 
        s=10, 
        alpha=0.4,
        edgecolors='none'
    )

    cbar = plt.colorbar(sc)
    cbar.set_label('LVEF (%)', fontweight='bold', rotation=270, labelpad=20)

    plt.xlabel('UMAP Dimension 1', fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontweight='bold')

    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.15)

    plot_save_path = file_path.replace('.npz', '_train_plot.png')
    plt.savefig(plot_save_path, bbox_inches='tight')
    print(f"Plot saved to: {plot_save_path}")

    plt.show()

def plot_split_umap(file_path):
    data = np.load(file_path)

    embedding_2d = data['embeddings']
    splits = data['splits']

    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette("Set1", n_colors=2)
    
    for i, category in enumerate(['train', 'test']):
        mask = (splits == category)
        plt.scatter(
            embedding_2d[mask, 0], 
            embedding_2d[mask, 1], 
            c=[colors[i]], 
            label=category,
            s=12, 
            alpha=0.6,
            edgecolors='none'
        )

    plt.xlabel('UMAP Dimension 1', fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontweight='bold')
    
    plt.legend(markerscale=2, frameon=True)
    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.15)

    plot_save_path = file_path.replace('.npz', '_split_plot.png')
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and save UMAP embeddings for ECG latents.')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test', 'valid', 'all'],
                        help="Which data split to use.")
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Directory containing latent .npz and metadata .json files.")
    parser.add_argument('--seg_type', type=str, required=True, choices=['atrial', 'ventricular'],
                        help="Segment type of the heartbeat.")
    args = parser.parse_args()
    save_umap_reduction(args.split, args.seg_type, args.root_dir)

    