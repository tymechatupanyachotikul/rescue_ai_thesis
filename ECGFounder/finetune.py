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

def main(args):
    num_lead = 12 # 12-lead ECG or 1-lead ECG 

    gpu_id = 4
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    early_stop_lr = args.early_stop_lr
    Epochs = args.epoch
    df_label_path = args.label_path
    ecg_path = args.data_root
    tasks = ['class']
    saved_dir = args.save_dir
    num_workers = args.num_workers

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    n_classes = len(tasks)

    if num_lead == 12:
        ECGdataset = LVEF_12lead_reg_Dataset()
        pth = './checkpoint/12_lead_ECGFounder.pth'
        model = ft_12lead_ECGFounder(device, pth, n_classes,linear_prob=False)
    elif num_lead == 1:
        ECGdataset = LVEF_1lead_reg_Dataset()
        pth = './checkpoint/1_lead_ECGFounder.pth'
        model = ft_1lead_ECGFounder(device, pth, n_classes,linear_prob=False)

    df_label = pd.read_csv(df_label_path)
    # Splitting the dataset into train, validation, and test sets

    train_df, test_df = train_test_split(df_label, test_size=0.2, shuffle=False)
    val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=False)

    train_dataset = ECGdataset(ecg_path= ecg_path,labels_df=train_df)
    val_dataset = ECGdataset(ecg_path= ecg_path,labels_df=val_df)
    test_dataset = ECGdataset(ecg_path= ecg_path,labels_df=test_df)

    # Example DataLoader usage
    trainloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=False)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max', verbose=True)

    ### train model
    best_mae = 100.
    step = 0
    current_lr = lr
    all_res = []
    pos_neg_counts = {}
    total_steps_per_epoch = len(trainloader)
    eval_steps = total_steps_per_epoch

    for epoch in range(Epochs):
        for batch in tqdm(trainloader,desc='Training'):
            input_x, input_y = tuple(t.to(device) for t in batch)
            outputs = model(input_x)
            loss = criterion(outputs, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

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
                rmse = np.sqrt(np.mean((all_pred_prob - all_gt) ** 2))

                print(f'MAE: {val_mae}')
                print(f'RMSE: {rmse}')

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
                rmse = np.sqrt(np.mean((all_pred_prob - all_gt) ** 2))

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
                    }, saved_dir)
                current_lr = optimizer.param_groups[0]['lr']

                columns = ['mae', 'rmse']
                
                all_res.append([mae, rmse])
                df = pd.DataFrame(all_res, columns=columns)

                df.to_csv(os.path.join(saved_dir, f'res_reg.csv'), index=False, float_format='%.5f')
                
                scheduler.step(rmse)
                ### early stop
                current_lr = optimizer.param_groups[0]['lr']
                    
                model.train() # set back to train