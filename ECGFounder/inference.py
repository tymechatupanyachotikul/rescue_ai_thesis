import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LVEF_12lead_reg_Dataset, LVEF_1lead_reg_Dataset
import argparse 
import random 

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    num_lead = args.num_lead
    gpu_id = args.gpu_id
    df_label_path = args.label_path
    ecg_path = args.data_root
    tasks = ['class']
    saved_dir = args.save_dir
    num_workers = args.num_workers
    checkpoint_path = args.checkpoint_path
    data_split_path = args.data_split_path
    model_name = args.model_name

    set_seed(args.seed)

    save_dir = os.path.join(saved_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f'Saving results to {save_dir}')

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    n_classes = len(tasks)

    if num_lead == 12:
        ECGdataset = LVEF_12lead_reg_Dataset
        model = ft_12lead_ECGFounder(device, checkpoint_path, n_classes)
    elif num_lead == 1:
        ECGdataset = LVEF_1lead_reg_Dataset
        model = ft_1lead_ECGFounder(device, checkpoint_path, n_classes)

    model.dense = nn.Identity() # remove last layer to get embeddings
    model.eval()

    df_label = pd.read_csv(df_label_path)
    data_split = json.load(open(data_split_path, 'r'))
    train_df = df_label[df_label['waveform_path'].isin(data_split['train'])]
    val_df = df_label[df_label['waveform_path'].isin(data_split['val'])]
    test_df = df_label[df_label['waveform_path'].isin(data_split['test'])]

    train_dataset = ECGdataset(ecg_path=ecg_path,labels_df=train_df)
    val_dataset = ECGdataset(ecg_path=ecg_path,labels_df=val_df)
    test_dataset = ECGdataset(ecg_path=ecg_path,labels_df=test_df)

    trainloader = DataLoader(train_dataset, batch_size=512,num_workers=num_workers, shuffle=False)
    valloader = DataLoader(val_dataset, batch_size=512,num_workers=num_workers, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=512,num_workers=num_workers, shuffle=False)

    with torch.no_grad():
        for dataloader, split in zip([trainloader, valloader, testloader], ['train', 'val', 'test']):
            split_out = []
            y_true = []
            for batch in tqdm(dataloader,desc=f'Extracting embeddings for {split}'):
                input_x, input_y = tuple(t.to(device) for t in batch)
                outputs = model(input_x)
                split_out.append(outputs.cpu().detach())
                y_true.append(input_y.cpu().detach())
            y_true = torch.cat(y_true, dim=0)
            split_out = torch.cat(split_out, dim=0)
            
            torch.save(split_out, os.path.join(save_dir, f'{split}_embeddings.pt'))
            torch.save(y_true, os.path.join(save_dir, f'{split}_labels.pt'))

def get_args():
    parser = argparse.ArgumentParser(description="Inference for ECGFounder")

    # --- Hardware & System Arguments ---
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='Which GPU ID to use (default: 0)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed of run (default: 42)')

    # --- Dataset & Paths ---
    parser.add_argument('--num_lead', type=int, default=12, choices=[1, 12],
                        help='Number of ECG leads: 1 or 12 (default: 12)')
    parser.add_argument('--data_root', type=str, required=True, 
                        help='Path to the directory containing ECG data')
    parser.add_argument('--label_path', type=str, default='./csv/LVEF.csv', 
                        help='Path to the CSV file containing labels')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--data_split_path', type=str, required=True,
                        help='Path to the JSON file containing data splits')
    parser.add_argument('--model_name', type=str, default='ECGFounder',
                        help='Name of the model (used for saving results)')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    main(args)