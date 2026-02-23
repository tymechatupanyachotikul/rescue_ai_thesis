import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import umap
# import seaborn as sns
# import matplotlib.pyplot as plt

def save_umap_reduction(splits, save_dir):
    embeddings = []
    labels = []
    splits_out = []
    for split in splits:
        embeddings.append(torch.load(os.path.join(save_dir, f'{split}_embeddings.pt')).cpu().numpy())
        labels.append(torch.load(os.path.join(save_dir, f'{split}_labels.pt')).cpu().numpy())
        splits_out.append(np.array([split] * len(labels[-1])))

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0).flatten()
    splits_out = np.concatenate(splits_out, axis=0)

    embeddings_healthy = embeddings[labels >= 60]
    embeddings_unhealthy = embeddings[labels <= 50]

    split_healthy = splits_out[labels >= 60]
    split_unhealthy = splits_out[labels <= 50] 

    labels_healthy = labels[labels >= 60]
    labels_unhealthy = labels[labels <= 50]

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42,
        verbose=True
    )

    print(f'Performing UMAP reduction')
    embedding_2d = reducer.fit_transform(embeddings)
    embedding_2d_healthy = reducer.fit_transform(embeddings_healthy)
    embedding_2d_unhealthy = reducer.fit_transform(embeddings_unhealthy)    
    print(f'UMAP reduction completed')

    np.savez_compressed(
        os.path.join(save_dir, f'{"_".join(splits)}_embeddings.npz'), 
        embeddings=embedding_2d, 
        labels=labels,
        splits=splits_out,
        embeddings_healthy=embedding_2d_healthy,
        labels_healthy=labels_healthy,
        splits_healthy=split_healthy,
        embeddings_unhealthy=embedding_2d_unhealthy,
        labels_unhealthy=labels_unhealthy,
        splits_unhealthy=split_unhealthy
    )
    print('Saved UMAP embeddings')

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

directories = ['linear_probe_ft', 'full_ft']
for directory in directories:
    print(f'Processing directory: {directory}')

    #save_dir = f'/Users/tyme/Desktop/University/Thesis/rescue_ai/results/LVEF/embeddings/{directory}'
    save_dir = f'/home/tchatupanyacho/rescue_ai_thesis/results/LVEF/embeddings/{directory}'
    save_umap_reduction(['test'], save_dir)
            
#plot_split_umap('/Users/tyme/Desktop/University/Thesis/rescue_ai/results/LVEF/embeddings/full_ft/test_train_embeddings.npz')
#load_and_plot_umap('/Users/tyme/Desktop/University/Thesis/rescue_ai/results/LVEF/embeddings/linear_probe_ft/test_embeddings.npz')