import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import os
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from util import save_checkpoint, save_reg_checkpoint, my_eval_with_dynamic_thresh
from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import LVEF_12lead_cls_Dataset, LVEF_12lead_reg_Dataset, LVEF_1lead_cls_Dataset, LVEF_1lead_reg_Dataset
import argparse 
import wandb
import random 
from datetime import datetime 

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_split(train_df, val_df, test_df, save_dir):
    splits = {
        "train": train_df,
        "test": test_df,
        "val": val_df
    }
    
    for split_name, df in splits.items():
        splits[split_name] = df['waveform_path'].tolist()
    
    save_path = os.path.join(save_dir, 'data_splits.json')
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

def main(args):

    run = wandb.init(
        entity="tymechatu-university-of-amsterdam",
        project="rescue_ai_ecg_founder",
        config=vars(args),
    )

    set_seed(args.seed)
    num_lead = args.num_lead # 12-lead ECG or 1-lead ECG 
    gpu_id = args.gpu_id
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    early_stop_lr = args.early_stop_lr
    Epochs = args.epoch
    df_label_path = args.label_path
    ecg_path = args.data_root
    tasks = ['class']
    saved_dir = args.save_dir
    linear_prob = args.linear_prob
    num_workers = args.num_workers

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(saved_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    n_classes = len(tasks)

    if num_lead == 12:
        ECGdataset = LVEF_12lead_reg_Dataset
        pth = './checkpoint/12_lead_ECGFounder.pth'
        model = ft_12lead_ECGFounder(device, pth, n_classes,linear_prob=linear_prob)
    elif num_lead == 1:
        ECGdataset = LVEF_1lead_reg_Dataset
        pth = './checkpoint/1_lead_ECGFounder.pth'
        model = ft_1lead_ECGFounder(device, pth, n_classes,linear_prob=linear_prob)

    df_label = pd.read_csv(df_label_path)
    # Splitting the dataset into train, validation, and test sets

    train_df, test_df = train_test_split(df_label, test_size=0.2, shuffle=True, random_state=args.seed)
    val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=False, random_state=args.seed)
    save_split(train_df, val_df, test_df, save_dir)

    train_dataset = ECGdataset(ecg_path= ecg_path,labels_df=train_df)
    val_dataset = ECGdataset(ecg_path= ecg_path,labels_df=val_df)
    test_dataset = ECGdataset(ecg_path= ecg_path,labels_df=test_df)

    # Example DataLoader usage
    trainloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=False)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='min', verbose=True)

    ### train model
    best_mae = 100.
    step = 0
    current_lr = lr
    all_res = []
    pos_neg_counts = {}
    total_steps_per_epoch = len(trainloader)
    eval_steps = total_steps_per_epoch

    for epoch in range(Epochs):
        training_loss = 0
        train_mae = 0
        for batch in tqdm(trainloader,desc='Training'):
            input_x, input_y = tuple(t.to(device) for t in batch)
            outputs = model(input_x)
            loss = criterion(outputs, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            training_loss += loss.item()
            train_mae += np.mean(np.abs(outputs.cpu().data.numpy() - input_y.cpu().data.numpy()))
            run.log({
                'train/tr_loss': loss.item(),
            })
            if step % eval_steps == 0:

                # val
                model.eval()
                prog_iter_val = tqdm(valloader, desc="Validation", leave=False)
                all_gt = []
                all_pred_prob = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter_val):
                        input_x, input_y = tuple(t.to(device) for t in batch)
                        pred = model(input_x)
                        all_pred_prob.append(pred.cpu().data.numpy())
                        all_gt.append(input_y.cpu().data.numpy())
                all_pred_prob = np.concatenate(all_pred_prob)
                all_gt = np.concatenate(all_gt)
                all_gt = np.array(all_gt)
                val_mae = np.mean(np.abs(all_pred_prob - all_gt))
                val_rmse = np.sqrt(np.mean((all_pred_prob - all_gt) ** 2))

                print(f'MAE: {val_mae}')
                print(f'RMSE: {val_rmse}')
                run.log({
                    'val/RMSE': val_rmse.item(),
                    'val/MAE': val_mae.item(),
                })
                # test
                model.eval()
                prog_iter_test = tqdm(testloader, desc="Testing", leave=False)
                all_gt = []
                all_pred_prob = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter_test):
                        input_x, input_y = tuple(t.to(device) for t in batch)
                        pred = model(input_x)
                        #pred = torch.sigmoid(logits)
                        all_pred_prob.append(pred.cpu().data.numpy())
                        all_gt.append(input_y.cpu().data.numpy())
                all_pred_prob = np.concatenate(all_pred_prob)
                all_gt = np.concatenate(all_gt)
                all_gt = np.array(all_gt)
                mae = np.mean(np.abs(all_pred_prob - all_gt))
                test_rmse = np.sqrt(np.mean((all_pred_prob - all_gt) ** 2))

                ### save model and res
                is_best = bool(val_mae < best_mae)
                if is_best:
                    best_mae = val_mae
                    print('==> Saving a new val best!')
                    save_reg_checkpoint({
                        'epoch': epoch,
                        'step': step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'mae': val_mae,
                    }, save_dir)

                    results_df = pd.DataFrame({
                        'GT': all_gt.flatten(),
                        'Predicted': all_pred_prob.flatten(),
                        })
                    results_df.to_csv(os.path.join(save_dir, f'pred_gt_reg.csv'), index=False)

                current_lr = optimizer.param_groups[0]['lr']
                
                run.log({
                    'test/test_rmse': test_rmse.item(),
                    'test/test_mae': mae.item(),
                })

                scheduler.step(val_rmse)
                ### early stop
                current_lr = optimizer.param_groups[0]['lr']
                    
                model.train() # set back to train
        run.log({
            'train/loss_per_epoch': training_loss / len(trainloader),
            'train/mae_per_epoch': train_mae / len(trainloader),
        })
        
def get_args():
    parser = argparse.ArgumentParser(description="ECG LVEF Finetune Model Training")

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

    # --- Training Hyperparameters ---
    parser.add_argument('--epoch', type=int, default=5, 
                        help='Number of total epochs to run (default: 50)')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='Batch size for training and testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay/L2 regularization (default: 0.0001)')
    parser.add_argument('--early_stop_lr', type=float, default=1e-5, 
                        help='Minimum learning rate for early stopping (default: 1e-6)')
    parser.add_argument('--linear_prob', action='store_true', help='Linear probing')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    main(args)