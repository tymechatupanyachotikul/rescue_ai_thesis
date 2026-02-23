import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import umap



def save_umap_reduction(splits, save_dir):
    embeddings = []
    labels = []
    for split in splits:
        embeddings.append(torch.load(os.path.join(save_dir, f'{split}_embeddings.pt')).cpu().numpy())
        labels.append(torch.load(os.path.join(save_dir, f'{split}_labels.pt')).cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

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
    print(f'UMAP reduction completed')

    np.savez_compressed(
        os.path.join(save_dir, f'{"_".join(splits)}_embeddings.npz'), 
        embeddings=embedding_2d, 
        labels=labels
    )
    print('Saved UMAP embeddings')

directories = ['linear_probe_ft', 'full_ft']
for directory in directories:
    print(f'Processing directory: {directory}')
    for split in ['test', 'train', 'val']:
        print(f'Processing split: {split}')
        save_dir = f'/home/tchatupanyacho/rescue_ai_thesis/results/LVEF/embeddings{directory}'
        save_umap_reduction([split], save_dir)