import json
import math
import os
import numpy as np
import torch
from model.build_model import build_model
from data.data_utils import load_data
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV,
    LogisticRegression, LogisticRegressionCV,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error,
    roc_auc_score, accuracy_score, f1_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

SOLVERS   = ["euler", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams", "dopri5"]
TASKS     = ['rot_mnist', 'rot_mnist_ou', 'sin', 'bb', 'lv', 'mocap', 'mocap_shift', 'ecg']
MODELS     = ['node', 'sonode', 'hbnode', 'vae']
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
parser.add_argument('--label_path', type=str, default='/projects/prjs1890/uk_biobank/phenotype_targets.pt',
                    help="Path to phenotype labels for ECG dataset")
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
parser.add_argument('--model_dir', type=str,
                    help="directory of model")
parser.add_argument('--analysis_latent', required=False, default=False, action='store_true',
                    help="Whether to do latent space analysis")

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def prepare_latents_and_labels(latents, metadata):
    """Extract valid (non-NaN/None) indices and label values for each phenotype parameter.

    Args:
        latents:  dict with 'z0' (np.ndarray [N, d]) and optionally 'm' (np.ndarray or None).
        metadata: list of N dicts, each containing {'patient_id': ..., 'labels': {param: value}}.

    Returns:
        clean_latents:  dict with 'z0' always present; 'm' included only when not None.
        valid_indices:  dict mapping param_name -> list[int] of row indices with valid values.
        valid_labels:   dict mapping param_name -> list of the corresponding label values.
    """
    clean_latents = {k: v for k, v in latents.items() if v is not None}

    all_params = set()
    for meta in metadata:
        all_params.update(meta['labels'].keys())

    valid_indices = {param: [] for param in all_params}
    valid_labels  = {param: [] for param in all_params}

    for i, meta in enumerate(metadata):
        for param, value in meta['labels'].items():
            if isinstance(value, float) and not math.isnan(value):
                valid_indices[param].append(i)
                valid_labels[param].append(value)
            elif isinstance(value, (int, bool)):
                valid_indices[param].append(i)
                valid_labels[param].append(float(value))
            elif isinstance(value, str):
                if value.lower() not in ['nan', 'none', '']:
                    valid_indices[param].append(i)
                    valid_labels[param].append(value)

    return clean_latents, valid_indices, valid_labels


def _regression_models():
    """Return (name, model) pairs for the four regression probes."""
    return [
        ('ols',   LinearRegression()),
        ('ridge', RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
        ('lasso', LassoCV(cv=5, max_iter=5000, n_jobs=-1)),
        ('mlp',   MLPRegressor(hidden_layer_sizes=(128,), activation='relu',
                               max_iter=500, early_stopping=True,
                               validation_fraction=0.1, random_state=0)),
    ]


def _classification_models():
    """Return (name, model) pairs for the four classification probes."""
    return [
        ('ols',   LogisticRegression(penalty=None, max_iter=1000, n_jobs=-1)),
        ('ridge', LogisticRegressionCV(penalty='l2', cv=5, max_iter=1000, n_jobs=-1)),
        ('lasso', LogisticRegressionCV(penalty='l1', solver='saga', cv=5,
                                       max_iter=1000, n_jobs=-1)),
        ('mlp',   MLPClassifier(hidden_layer_sizes=(128,), activation='relu',
                                max_iter=500, early_stopping=True,
                                validation_fraction=0.1, random_state=0)),
    ]


def _eval_regression(model, X_tr, y_tr, X_te, y_te):
    y_tr_arr = np.array(y_tr, dtype=float)
    y_te_arr = np.array(y_te, dtype=float)
    model.fit(X_tr, y_tr_arr)
    y_pred = model.predict(X_te)
    return {
        'model':   model,
        'y_pred':  y_pred,
        'y_true':  y_te_arr,
        'metrics': {
            'mse': float(mean_squared_error(y_te_arr, y_pred)),
            'r2':  float(r2_score(y_te_arr, y_pred)),
        },
    }


def _eval_classification(model, X_tr, y_tr, X_te, y_te, le):
    y_tr_enc = le.transform(y_tr)
    y_te_enc = le.transform(y_te)
    model.fit(X_tr, y_tr_enc)
    y_pred = model.predict(X_te)
    acc = float(accuracy_score(y_te_enc, y_pred))
    f1  = float(f1_score(y_te_enc, y_pred, average='macro', zero_division=0))
    metrics = {'accuracy': acc, 'f1': f1}
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_te)
        try:
            if len(le.classes_) == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_te_enc, y_prob[:, 1]))
            else:
                metrics['roc_auc'] = float(
                    roc_auc_score(y_te_enc, y_prob, multi_class='ovr', average='macro')
                )
        except ValueError:
            pass
    return {
        'model':   model,
        'y_pred':  y_pred,
        'y_prob':  y_prob,
        'y_true':  y_te_enc,
        'metrics': metrics,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_regression_param(param, model_results, out_dir):
    """Scatter (y_true vs y_pred) for each model, 4-panel figure."""
    names = list(model_results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        res = model_results[name]
        y_true, y_pred = res['y_true'], res['y_pred']
        ax.scatter(y_true, y_pred, alpha=0.4, s=8, rasterized=True)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=1)
        r2  = res['metrics']['r2']
        mse = res['metrics']['mse']
        ax.set_title(f"{name}\nR²={r2:.3f}  MSE={mse:.3f}", fontsize=9)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
    fig.suptitle(param, fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'predictions.png'), dpi=120)
    plt.close(fig)


def _plot_classification_param(param, model_results, out_dir, le):
    """ROC curves (one panel) and confusion matrices (4-panel) side by side."""
    names  = list(model_results.keys())
    binary = len(le.classes_) == 2

    # --- ROC curves ---
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    for name, res in model_results.items():
        if res['y_prob'] is not None and binary:
            RocCurveDisplay.from_predictions(
                res['y_true'], res['y_prob'][:, 1],
                name=f"{name} (AUC={res['metrics'].get('roc_auc', float('nan')):.2f})",
                ax=ax_roc,
            )
    if binary:
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_title(f"{param} — ROC curves")
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(out_dir, 'roc_curves.png'), dpi=120)
    plt.close(fig_roc)

    # --- Confusion matrices ---
    fig_cm, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        res = model_results[name]
        ConfusionMatrixDisplay.from_predictions(
            res['y_true'], res['y_pred'],
            display_labels=le.classes_,
            ax=ax, colorbar=False,
        )
        acc = res['metrics']['accuracy']
        f1  = res['metrics']['f1']
        ax.set_title(f"{name}\nacc={acc:.3f}  f1={f1:.3f}", fontsize=9)
    fig_cm.suptitle(param, fontsize=11, fontweight='bold')
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(out_dir, 'confusion_matrices.png'), dpi=120)
    plt.close(fig_cm)


def _plot_summary_regression(all_results, out_dir):
    """Grouped bar chart: R² per param × model."""
    model_names = ['ols', 'ridge', 'lasso', 'mlp']
    params = sorted(all_results.keys())
    x = np.arange(len(params))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(params) * 1.2 + 2), 5))
    for i, metric in enumerate(['r2', 'mse']):
        ax = axes[i]
        for j, mname in enumerate(model_names):
            vals = [all_results[p][mname]['metrics'].get(metric, float('nan')) for p in params]
            ax.bar(x + j * width, vals, width, label=mname)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(params, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Regression summary — {metric.upper()}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'regression_summary.png'), dpi=120)
    plt.close(fig)


def _plot_summary_classification(all_results, out_dir):
    """Grouped bar chart: AUC / accuracy / F1 per param × model."""
    model_names = ['ols', 'ridge', 'lasso', 'mlp']
    params = sorted(all_results.keys())
    x = np.arange(len(params))
    width = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(params) * 1.5 + 2), 5))
    for i, metric in enumerate(['accuracy', 'f1', 'roc_auc']):
        ax = axes[i]
        for j, mname in enumerate(model_names):
            vals = [all_results[p][mname]['metrics'].get(metric, float('nan')) for p in params]
            ax.bar(x + j * width, vals, width, label=mname)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(params, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(f"Classification summary — {metric}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'classification_summary.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main probing entry-point
# ---------------------------------------------------------------------------

def run_linear_probes(train_latents, train_metadata, test_latents, test_metadata,
                      latent_key='z0', out_root=None):
    """Train four probes per phenotype and evaluate on the test set.

    Models
    ------
    Regression    : OLS, RidgeCV (L2), LassoCV (L1), MLP-1-hidden-layer
    Classification: Logistic (no penalty), LogisticRegressionCV L2, L1, MLP

    Metrics
    -------
    Regression    : MSE, R²
    Classification: Accuracy, F1 (macro), ROC-AUC

    Directory layout (when out_root is given)
    -----------------------------------------
    {out_root}/
      regression/
        {param}/  predictions.png  metrics.json
        _summary/ regression_summary.png
      classification/
        {param}/  roc_curves.png  confusion_matrices.png  metrics.json
        _summary/ classification_summary.png

    Returns
    -------
    results : dict  param -> model_name -> {'model', 'metrics', 'y_pred', 'y_true', ...}
    """
    _, tr_indices, tr_labels_all = prepare_latents_and_labels(train_latents, train_metadata)
    _, te_indices, te_labels_all = prepare_latents_and_labels(test_latents,  test_metadata)

    X_tr_full = train_latents[latent_key]
    X_te_full = test_latents[latent_key]

    all_params = set(tr_indices.keys()) & set(te_indices.keys())

    reg_results  = {}   # param -> {model_name -> eval_dict}
    clf_results  = {}

    for param in sorted(all_params):
        tr_idx = tr_indices[param]
        te_idx = te_indices[param]

        if len(tr_idx) < 10 or len(te_idx) < 2:
            print(f"  [{param}] skipped — too few samples "
                  f"(train={len(tr_idx)}, test={len(te_idx)})")
            continue

        y_tr = tr_labels_all[param]
        y_te = te_labels_all[param]
        is_categorical = isinstance(y_tr[0], str)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_full[tr_idx])
        X_te = scaler.transform(X_te_full[te_idx])

        if is_categorical:
            le = LabelEncoder().fit(y_tr + y_te)
            param_results = {}
            for name, mdl in _classification_models():
                param_results[name] = _eval_classification(mdl, X_tr, y_tr, X_te, y_te, le)
                m = param_results[name]['metrics']
                print(f"  [{param}][{name}]  acc={m['accuracy']:.3f}  "
                      f"f1={m['f1']:.3f}  auc={m.get('roc_auc', float('nan')):.3f}")

            clf_results[param] = param_results

            if out_root:
                pdir = os.path.join(out_root, 'classification', param)
                os.makedirs(pdir, exist_ok=True)
                _plot_classification_param(param, param_results, pdir, le)
                _save_metrics_json(param_results, pdir)
        else:
            param_results = {}
            for name, mdl in _regression_models():
                param_results[name] = _eval_regression(mdl, X_tr, y_tr, X_te, y_te)
                m = param_results[name]['metrics']
                print(f"  [{param}][{name}]  R²={m['r2']:.3f}  MSE={m['mse']:.4f}")

            reg_results[param] = param_results

            if out_root:
                pdir = os.path.join(out_root, 'regression', param)
                os.makedirs(pdir, exist_ok=True)
                _plot_regression_param(param, param_results, pdir)
                _save_metrics_json(param_results, pdir)

    # Summary plots
    if out_root:
        if reg_results:
            sdir = os.path.join(out_root, 'regression', '_summary')
            os.makedirs(sdir, exist_ok=True)
            _plot_summary_regression(reg_results, sdir)
        if clf_results:
            sdir = os.path.join(out_root, 'classification', '_summary')
            os.makedirs(sdir, exist_ok=True)
            _plot_summary_classification(clf_results, sdir)

    return {'regression': reg_results, 'classification': clf_results}



def _save_metrics_json(param_results, out_dir):
    """Persist metrics (no model objects) to metrics.json."""
    serialisable = {name: res['metrics'] for name, res in param_results.items()}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(serialisable, f, indent=2)



def _collect_sample_latents(dataloader, model, split, args):
    """Run inference over a dataloader and collect per-sample latent.

    """
    save_directory = os.path.join(args.model_dir, 'latents')
    # For a plain DataLoader the dataset is ECGDataset; for a Subset it's one level deeper
    ecg_dataset = dataloader.dataset
    ecg_dataset.return_file_path = True
    model.eval()
    model.return_latent = True

    metadata_dict = []
    if args.model == 'vae':
        latent_tensors = {
            'z0': [],
        }
    elif args.model == 'node':
        latent_tensors = {
            'z0': [],
            'm': [],
        }
    else:
        raise ValueError(f"Latent collection not implemented for model type {args.model}")

    phenotypes = torch.load(args.label_path)
    eids = phenotypes['eids']
    eid_to_idx = {eid: i for i, eid in enumerate(eids)}   # O(1) lookup
    targets_np = phenotypes['targets'].cpu().numpy()       # convert once
    columns = phenotypes['columns']
    not_found = 0
    print(f'Dataset size : {len(dataloader.dataset.file_paths)}')
    with torch.inference_mode():
        for batch, batch_y, mask in tqdm(dataloader, desc="Collecting latents"):
            batch = batch.to(model.device)
            mask  = mask.to(model.device)

            z0, m = model(batch, args.plotL, mask=mask)
            z0 = z0.squeeze(0).squeeze(1).cpu().numpy()
            if m is not None:
                m = m.squeeze(0).cpu().numpy()

            patient_ids = [item[1] for item in batch_y]

            for i in range(batch.shape[0]):
                eid_idx = eid_to_idx.get(patient_ids[i])
                if eid_idx is None:
                    not_found += 1
                    continue

                labels = dict(zip(columns, targets_np[eid_idx].tolist()))
                metadata_dict.append({'patient_id': patient_ids[i], 'labels': labels})

                latent_tensors['z0'].append(z0[i])
                if m is not None and 'm' in latent_tensors:
                    latent_tensors['m'].append(m[i])
    
    print(f"Finished collecting latents. {not_found} patient_ids were not found in phenotypes and were skipped.")
    latents = {k: np.stack(v, axis=0) for k, v in latent_tensors.items() if len(v) > 0}

    os.makedirs(save_directory, exist_ok=True)
    np.savez(os.path.join(save_directory, f'{split}_latents.npz'), **latents)
    with open(os.path.join(save_directory, f'{split}_metadata.json'), 'w') as f:
        json.dump(metadata_dict, f)
    print(f"Saved latents and metadata to {save_directory}/{split}_*")

    return latents, metadata_dict
    
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

    ckpt = torch.load(os.path.join(args.model_dir, 'model.pth'), map_location=torch.device(device), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    latents_dict = {}

    for split, dataloader in zip(['train', 'test'], [trainset, testset]):
        latents, metadata = _collect_sample_latents(dataloader, model, split, args)
        latents_dict[split] = {
            'latents': latents,
            'metadata': metadata
        }

    finetune_root = os.path.join(args.model_dir, 'finetune_results')

    # Build combined z0+m latent when m is available
    for split in ['train', 'test']:
        lats = latents_dict[split]['latents']
        if 'm' in lats:
            lats['z0_m'] = np.concatenate([lats['z0'], lats['m']], axis=1)

    print("\n=== Linear probes (z0) ===")
    results_z0 = run_linear_probes(
        latents_dict['train']['latents'], latents_dict['train']['metadata'],
        latents_dict['test']['latents'],  latents_dict['test']['metadata'],
        latent_key='z0',
        out_root=os.path.join(finetune_root, 'z0'),
    )

    if 'm' in latents_dict['train']['latents']:
        print("\n=== Linear probes (m) ===")
        results_m = run_linear_probes(
            latents_dict['train']['latents'], latents_dict['train']['metadata'],
            latents_dict['test']['latents'],  latents_dict['test']['metadata'],
            latent_key='m',
            out_root=os.path.join(finetune_root, 'm'),
        )

        print("\n=== Linear probes (z0 + m combined) ===")
        results_z0_m = run_linear_probes(
            latents_dict['train']['latents'], latents_dict['train']['metadata'],
            latents_dict['test']['latents'],  latents_dict['test']['metadata'],
            latent_key='z0_m',
            out_root=os.path.join(finetune_root, 'z0_m'),
        )

