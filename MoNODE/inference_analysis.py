import json
import os 
import numpy as np
import torch
from model.build_model import build_model
from model.model_misc import compute_masked_mse
from data.data_utils import load_data
import argparse
import matplotlib.pyplot as plt 
from model.misc.plot_utils import plot_ecg_out
from model.model_misc import compute_loss
from matplotlib.animation import FFMpegWriter
from sklearn.manifold import TSNE
from collections import defaultdict 

SOLVERS   = ["euler", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams", "dopri5"]
TASKS     = ['rot_mnist', 'rot_mnist_ou', 'sin', 'bb', 'lv', 'mocap', 'mocap_shift', 'ecg']
MODELS     = ['node', 'sonode', 'hbnode']
GRADIENT_ESTIMATION = ['no_adjoint', 'adjoint', 'ac_adjoint']
parser = argparse.ArgumentParser('MoNODE')
np.seterr(all='raise')

#data
parser.add_argument('--task', type=str, default='mov_mnist', choices=TASKS,
                    help="Experiment type")
parser.add_argument('--noise', type=float, default=None,
                    help="set noise level for noise robustness experiments")  
parser.add_argument('--Nobj', type=int, default=1,
                    help="param that can be used for multiple object set-up")                 
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers")
parser.add_argument('--data_root', type=str, default='data/',
                    help="general data location")
parser.add_argument('--shuffle', type=eval, default=True,
               help='For Moving MNIST whetehr to shuffle the data')
parser.add_argument('--dataset_root', type=str, default='/projects/prjs1890/',
                    help="dataset location for ecg")
parser.add_argument('--segment_type', choices=['atrial', 'ventricular'],
                    help="Segment type of heart beat", type=str)
#de model
parser.add_argument('--model', type=str, default='node', choices=MODELS,
                    help='node model type')
parser.add_argument('--ode_latent_dim', type=int, default=10,
                    help="Latent ODE dimensionality")
parser.add_argument('--de_L', type=int, default=2,
                    help="Number of hidden layers in MLP diff func")
parser.add_argument('--de_H', type=int, default=100,
                    help="Number of hidden neurons in each layer of MLP diff func")


#invariance
parser.add_argument('--inv_fnc', type=str, default='MLP',
                    help="Invariant function")
parser.add_argument('--modulator_dim', type=int, default=0,
                    help = 'dim of the dynamics modulator variable')
parser.add_argument('--content_dim', type=int, default=0,
                    help = 'dim of the content variable')
parser.add_argument('--T_inv', type=int, default=5,
                    help="Time frames to select for RNN based Encoder for Invariance")
parser.add_argument('--cnn_filt_inv', type=int, default=16,
                    help="Nfilt invariant encoder cnn")


#ode stuff
parser.add_argument('--order', type=int, default=1,
                    help="order of ODE")
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--dt', type=float, default=0.1,
                    help="numerical solver dt")
parser.add_argument('--use_adjoint', type=str, default='no_adjoint', choices=GRADIENT_ESTIMATION, #we used False
                    help="Use adjoint method for gradient computation")

#vae 
parser.add_argument('--T_in', type=int, default=10,
                    help="Time frames to select for RNN based Encoder for intial state")
parser.add_argument('--cnn_filt_enc', type=int, default=16,
                    help="Number of filters in the cnn encoder")
parser.add_argument('--cnn_filt_de', type=int, default=16,
                    help="Number of filters in the cnn decoder")
parser.add_argument('--rnn_hidden', type=int, default=10,
                    help="Encoder RNN latent dimensionality") 
parser.add_argument('--dec_H', type=int, default=100,
                    help="Number of hidden neurons in MLP decoder") 
parser.add_argument('--dec_L', type=int, default=2,
                    help="Number of hidden layers in MLP decoder") 
parser.add_argument('--dec_act', type=str, default='relu',
                    help="MLP Decoder activation") 
parser.add_argument('--enc_H', type=int, default=50,
                    help="Encoder hidden dimensionality for GRU unit") 
parser.add_argument('--sonode_v', type=str, default='MLP', choices=['MLP','RNN'],
                    help="velocity encoder for SONODE") 

#training 
parser.add_argument('--Nepoch', type=int, default=600,
                    help="Number of gradient steps for model training")
parser.add_argument('--Nincr', type=int, default=10,
                    help="Number of sequential increments of the sequence length")
parser.add_argument('--batch_size', type=int, default=25,
                    help="batch size")
parser.add_argument('--lr', type=float, default=0.002,
                    help="Learning rate for model training")
parser.add_argument('--sobolev_weight', type=float, default=0,
                    help="Weight of derivative loss likelihood")
parser.add_argument('--l_w', type=float, default=0,
                    help="Weight of likelihood scaled on derivative")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--continue_training', type=eval, default=False,
                    help="If set to True continoues training of a previous model")
parser.add_argument('--plot_every', type=int, default=20,
                    help="How often plot the training")
parser.add_argument('--plotL', type=int, default=1,
                    help="Number of MC draws for plotting")
parser.add_argument('--forecast_tr',type=int, default=2, 
                    help="Number of forecast steps for plotting train")
parser.add_argument('--forecast_vl',type=int, default=2,
                    help="Number of forecast steps for plotting test")
parser.add_argument('--exp_id', type=int, default=0,
                    help = 'exp ID for directory')

#log 
parser.add_argument('--save', type=str, default='results/',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--continue_dir', type=str, default='results/',
                    help="Directory name for continue training")
parser.add_argument('--model_path', type=str,
                    help="path of model")
parser.add_argument('--analysis_latent', required=False, default=False, action='store_true',
                    help="Whether to do latent space analysis")

def _collect_sample_results(dataloader, model, args, class_filter=None):
    """Run inference over a dataloader and collect per-sample results.

    Args:
        class_filter: optional set of class names to keep. None = keep all.
    """
    results = []
    loss_per_class = defaultdict(list)
    dataloader.dataset.return_file_path = True

    with torch.no_grad():
        for batch, batch_y, mask in dataloader:
            batch = batch.to(model.device)
            mask  = mask.to(model.device)

            Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C, c, m = model(batch, args.plotL, mask=mask)

            Xrec_mean = Xrec.mean(0)                           # (N, T, D)
            se        = (Xrec_mean - batch) ** 2               # (N, T, D)
            mask_exp  = mask.unsqueeze(-1).float()             # (N, T, 1)
            n_valid   = mask_exp.sum(dim=1).clamp(min=1)       # (N, 1)
            mse_per_lead   = (se * mask_exp).sum(dim=1) / n_valid   # (N, D)
            mse_per_sample = mse_per_lead.mean(dim=-1)         # (N,)

            classes     = [item[0] for item in batch_y]
            patient_ids = [item[1] for item in batch_y]
            filenames   = [item[2] for item in batch_y]

            for i in range(batch.shape[0]):
                if class_filter is not None and classes[i] not in class_filter:
                    continue
                results.append({
                    'filename':          filenames[i],
                    'class':             classes[i],
                    'patient_id':        patient_ids[i],
                    'mse':               mse_per_sample[i].item(),
                    'mse_per_lead':      mse_per_lead[i].cpu().tolist(),
                    'original_ecg':      batch[i].cpu(),
                    'reconstructed_ecg': Xrec_mean[i].cpu(),
                    'mask':              mask[i].cpu(),
                })

            for cls in set(classes):
                if class_filter is not None and cls not in class_filter:
                    continue
                idx = [j for j, v in enumerate(classes) if v == cls]
                cls_mask    = mask_exp[idx]
                cls_se      = se[idx]
                cls_n_valid = cls_mask.sum(dim=1).clamp(min=1)
                cls_mse     = (cls_se * cls_mask).sum(dim=1) / cls_n_valid
                loss_per_class[cls].append(cls_mse.mean().item())

    return results, loss_per_class


def get_latent_trajectory(Q, method='pca'):
    [T,q] = Q.shape 

    if method == 'pca':
        U,S,V = torch.pca_lowrank(Q, q=min(q,10))
        Qpca = Q @ V[:,:2] 
        Qpca = Qpca.reshape(T,2).detach().cpu().numpy() # L,N,T,2
        S = S / S.sum()

        return {'PC1': Qpca[:,0], 'PC2': Qpca[:,1], 'S': S}
    elif method == 't-sne':
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')

        Q_tsne = tsne.fit_to_transform(Q.detach().cpu().numpy())
        return {'PC1': Q_tsne[:,0], 'PC2': Q_tsne[:,1]}


def plot_latent_velocity(zt):

    zt_v = torch.diff(zt, dim=1)
    zt_v_norm = torch.norm(zt_v, dim=-1, p=2)
    zt_v_avg = zt_v_norm.mean(dim=0).detach().cpu().numpy()
    
    t = np.arange(len(zt_v_avg))
    plt.figure()
    plt.title("Average Latent Velocity Magnitude (Mean of All Samples)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (Units/s)")
    plt.plot(t, zt_v_avg)

    model_name = args.model_path.split('/')[-1][:-4]
    save_path = f"{model_name}_velocity_analysis.png"
    plt.savefig(save_path)
    
if __name__ == '__main__':
    args = parser.parse_args()
    dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    analysis_latent = args.analysis_latent

    trainset, validset, testset, manager, params = load_data(args, dtype)
    
    if args.task == 'ecg':
        config = {
            'inp_dim': 12 - len(params[args.task]['exclude_leads_in']),
            'w_dt': args.sobolev_weight,
            'l_w': args.l_w,
            'out_dim': 12 - len(params[args.task]['exclude_leads_out'])
        }
    model = build_model(args, device, dtype, **config)
    model.to(device)
    model.to(dtype)


    ckpt = torch.load(args.model_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    total_samples = 100
    cur_samples = 0
    zt = []
    X_all = []
    X_true_all = []

    sample_results, loss_per_class = _collect_sample_results(testset, model, args)

    if analysis_latent:
        assert trainset is not None
        n_train = trainset.dataset.shape[0]
        testset.dataset.return_file_path = True
        for test_batch, test_y, test_mask in testset:
            test_batch = test_batch.to(model.device)
            X, _zt, _, _, C_vl, _, _ = model(test_batch, L=args.plotL)
            loss, nlhood, kl_z0, Xrec_tr, ztL_tr, tr_mse, _, _, _loss_per_class, nlhood_dt = compute_loss(model, test_batch, test_y, 1, num_observations=n_train)
            zt.append(_zt[0, :, :, :])
            X_all.append(X)
            X_true_all.append(test_batch)
            cur_samples += _zt.shape[1]

            if cur_samples >= total_samples:
                break
    
    
    if analysis_latent:
        zt = torch.cat(zt, dim=0)
        if args.order == 2:
            zt = zt[:, :, : zt.shape[-1] // 2]
        
        method = 'pca'
        latent_trajectory = get_latent_trajectory(zt[0], method)

        latent_y = latent_trajectory['PC1']
        latent_x = latent_trajectory['PC2']
        
        latent_all = np.vstack((latent_y, latent_x))
        zt_v = np.diff(latent_all, axis=1)
        zt_v_norm = np.linalg.norm(zt_v, axis=0)

        t = np.arange(zt_v_norm.shape[0])

        model_name = args.model_path.split('/')[-1][:-4]
        
        writer = FFMpegWriter(fps=15)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
        line1, = ax1.plot([], [], 'r-', lw=2)
        if method == 'pca':
            ax1.set_ylabel('PCA-1  ({:.2f})'.format(latent_trajectory['S'][0]),fontsize=15)
            ax1.set_xlabel('PCA-2  ({:.2f})'.format(latent_trajectory['S'][1]),fontsize=15)
        else:
            ax1.set_ylabel('PCA-1',fontsize=15)
            ax1.set_xlabel('PCA-2',fontsize=15)

        ax1.set_xlim(
            latent_x.min().item(), 
            latent_x.max().item()
            )
        ax1.set_ylim(
            latent_y.min().item(),
            latent_y.max().item()
        )

        ax2.set_ylabel('Latent velocity (PC)')
        ax2.set_xlabel('Time')

        ax2.set_xlim(t[0], t[-1])
        ax2.set_ylim(
            zt_v_norm.min().item(),
            zt_v_norm.max().item()
        )
        line2, = ax3.plot([], [], 'r--', lw=2, label='Reconstructed')
        line3, = ax3.plot([], [], 'b--', lw=2, label='Ground truth')
        line4, = ax2.plot([], [], 'b')

        ax3.set_title("ECG")

        ax3.set_xlim(t[0], t[-1])
        ax3.set_ylim(
            min(X_all[0][0, 0, :, 0].min().item(), X_true_all[0][0, :, 0].min().item()),
            max(X_all[0][0, 0, :, 0].max().item(), X_true_all[0][0, :, 0].max().item()),
        )
        ax3.legend(loc="upper left")
        fig.tight_layout()
        with writer.saving(fig, f"{model_name}_latent_reconstruction.mp4", dpi=100):
            for i in range(len(t)):            
                line1.set_data(latent_x[:i], latent_y[:i])
                line2.set_data(t[:i], X_all[0][0, 0, :i, 0].detach().cpu().numpy().flatten())
                line3.set_data(t[:i], X_true_all[0][0, :i, 0].detach().cpu().numpy().flatten())
                line4.set_data(t[:i], zt_v_norm[:i])
                writer.grab_frame()
    save_directory = os.path.join(os.path.dirname(args.model_path), 'analysis_results')
    os.makedirs(save_directory, exist_ok=True)

    def pad_to(tensor, length):
        T, D = tensor.shape
        if T < length:
            tensor = torch.cat([tensor, torch.zeros(length - T, D)], dim=0)
        return tensor

    def save_results(results, prefix):
        meta = [{k: v for k, v in r.items()
                 if k not in ('original_ecg', 'reconstructed_ecg', 'mask')}
                for r in results]
        with open(os.path.join(save_directory, f'{prefix}_results_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        max_len = max(r['original_ecg'].shape[0] for r in results)
        torch.save({
            'original_ecg':      torch.stack([pad_to(r['original_ecg'], max_len) for r in results]),
            'reconstructed_ecg': torch.stack([pad_to(r['reconstructed_ecg'], max_len) for r in results]),
            'mask':              torch.stack([pad_to(r['mask'].unsqueeze(-1).float(), max_len).squeeze(-1) for r in results]),
        }, os.path.join(save_directory, f'{prefix}_results_tensors.pt'))

    save_results(sample_results, 'test')

    TRAIN_CLASS_FILTER = {'lbbb', 'rbbb'}
    train_dataset = trainset.dataset
    filtered_indices = [i for i, lbl in enumerate(train_dataset.labels) if lbl in TRAIN_CLASS_FILTER]
    filtered_subset  = torch.utils.data.Subset(train_dataset, filtered_indices)
    filtered_loader  = torch.utils.data.DataLoader(
        filtered_subset,
        batch_size=trainset.batch_size,
        shuffle=False,
        num_workers=trainset.num_workers,
        collate_fn=trainset.collate_fn,
        pin_memory=trainset.pin_memory,
    )
    train_results, _ = _collect_sample_results(filtered_loader, model, args)
    save_results(train_results, 'train')