import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV,
    LogisticRegression, LogisticRegressionCV,
)
from sklearn.metrics import (
    r2_score, mean_squared_error,
    roc_auc_score, accuracy_score, f1_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder



# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_split(latents_dir: str, split: str) -> tuple[dict, list]:
    """Load pre-saved latents and metadata for one split.

    Supports two file naming conventions:
      - finetune.py style:    {split}_latents.npz  /  {split}_metadata.json
      - inference_analysis.py style: latent_tensors_{split}.npz  / latent_meta_dict_{split}.json
    """
    npz_candidates = [
        os.path.join(latents_dir, f'{split}_latents.npz'),
        os.path.join(latents_dir, f'latent_tensors_{split}.npz'),
    ]
    json_candidates = [
        os.path.join(latents_dir, f'{split}_metadata.json'),
        os.path.join(latents_dir, f'latent_meta_dict_{split}.json'),
    ]

    npz_path  = next((p for p in npz_candidates  if os.path.exists(p)), None)
    json_path = next((p for p in json_candidates if os.path.exists(p)), None)

    if npz_path is None:
        raise FileNotFoundError(
            f"No latent .npz found for split '{split}' in {latents_dir}.\n"
            f"Tried: {npz_candidates}")
    if json_path is None:
        raise FileNotFoundError(
            f"No metadata .json found for split '{split}' in {latents_dir}.\n"
            f"Tried: {json_candidates}")

    npz      = np.load(npz_path)
    latents  = dict(npz)          # {'z0': ndarray, 'm': ndarray, ...}

    with open(json_path) as f:
        raw_meta = json.load(f)

    # Normalise metadata: inference_analysis.py stores labels flat under 'labels',
    # but for MedalCare-XL the labels dict only has 'class' and 'patient_id'.
    # Ensure every entry has a 'labels' key.
    metadata = []
    for entry in raw_meta:
        if 'labels' not in entry:
            # Flatten all non-latent fields into labels
            entry = dict(entry)
            entry['labels'] = {k: v for k, v in entry.items()
                               if k not in ('filename', 'patient_id')}
        metadata.append(entry)

    print(f"  [{split}] loaded {latents['z0'].shape[0]} samples "
          f"from {os.path.basename(npz_path)}")
    return latents, metadata

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
    with np.errstate(under='ignore', divide='ignore'):
        model.fit(X_tr, y_tr_enc)
    y_pred = model.predict(X_te)
    acc = float(accuracy_score(y_te_enc, y_pred))
    f1  = float(f1_score(y_te_enc, y_pred, average='macro', zero_division=0))
    metrics = {'accuracy': acc, 'f1': f1}
    y_prob = None
    if hasattr(model, 'predict_proba'):
        with np.errstate(under='ignore'):   # sklearn softmax triggers harmless underflow
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


def _per_class_accuracy(y_true, y_pred, classes):
    """Return dict of class_label -> accuracy for each class."""
    result = {}
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() == 0:
            result[cls] = float('nan')
        else:
            result[cls] = float((y_pred[mask] == i).mean())
    return result


def _plot_classification_param(param, model_results, out_dir, le):
    """Per-param plots: confusion matrix per model, ROC curves, per-class accuracy, group summary."""
    import json as _json
    names  = list(model_results.keys())
    classes = list(le.classes_)
    binary  = len(classes) == 2
    n_cls   = len(classes)
    
    # Class grouping for summary bar chart
    ventricular = [c for c in classes if c.lower() in ('lcx_03_ant', 'lcx_03_post', 'rca_0_3', 'rca_10', 'lad_10', 'lad_03', 'lcx_10_post', 'lbbb', 'rbbb')]
    atrial      = [c for c in classes if c.lower() in ('avblock', 'fam', 'iab', 'lae')]
    sinus       = [c for c in classes if c.lower() == 'sinus']

    # --- Confusion matrix: one figure per model, saved separately ---
    cell_size = max(0.7, 5.0 / n_cls)   # shrink cells for many classes
    tick_fs   = max(5, 9 - n_cls // 3)  # shrink tick font for many classes

    for name in names:
        res = model_results[name]
        fig_cm, ax = plt.subplots(figsize=(cell_size * n_cls + 1.5,
                                           cell_size * n_cls + 1.5))
        disp = ConfusionMatrixDisplay.from_predictions(
            res['y_true'], res['y_pred'],
            display_labels=classes,
            ax=ax, colorbar=True,
            xticks_rotation=45,
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fs, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fs)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        acc = res['metrics']['accuracy']
        f1  = res['metrics']['f1']
        ax.set_title(f"{param} — {name}\nacc={acc:.3f}  f1={f1:.3f}", fontsize=10, fontweight='bold')
        fig_cm.tight_layout()
        fig_cm.savefig(os.path.join(out_dir, f'confusion_matrix_{name}.png'), dpi=130)
        plt.close(fig_cm)

    # --- Per-class accuracy: grouped bar chart across models ---
    pca_data = {name: _per_class_accuracy(model_results[name]['y_true'],
                                           model_results[name]['y_pred'],
                                           classes)
                for name in names}

    # Save per-class accuracy to JSON
    with open(os.path.join(out_dir, 'per_class_accuracy.json'), 'w') as f:
        _json.dump(pca_data, f, indent=2)

    x      = np.arange(n_cls)
    width  = 0.8 / max(len(names), 1)
    offsets = np.linspace(-(len(names) - 1) / 2 * width,
                           (len(names) - 1) / 2 * width, len(names))
    colours = [f'C{i}' for i in range(len(names))]

    fig_pca, ax_pca = plt.subplots(figsize=(max(8, n_cls * 0.9 + 2), 4))
    for (name, colour, offset) in zip(names, colours, offsets):
        vals = [pca_data[name].get(cls, float('nan')) for cls in classes]
        ax_pca.bar(x + offset, vals, width, label=name, color=colour, alpha=0.85)
    ax_pca.set_xticks(x)
    ax_pca.set_xticklabels(classes, rotation=40, ha='right', fontsize=8)
    ax_pca.set_ylabel('Accuracy', fontsize=10)
    ax_pca.set_ylim(0, 1.08)
    ax_pca.axhline(1.0, color='grey', lw=0.6, linestyle='--')
    ax_pca.set_title(f"{param} — per-class accuracy", fontsize=11, fontweight='bold')
    ax_pca.legend(fontsize=8, framealpha=0.8)
    ax_pca.spines[['top', 'right']].set_visible(False)
    fig_pca.tight_layout()
    fig_pca.savefig(os.path.join(out_dir, 'per_class_accuracy.png'), dpi=130)
    plt.close(fig_pca)

    # --- Group-summary bar chart (ventricular / atrial / sinus) ---
    groups = [('Ventricular', ventricular), ('Atrial', atrial), ('Sinus', sinus)]
    groups = [(g, cls_list) for g, cls_list in groups if cls_list]  # skip absent groups

    if groups:
        group_names  = [g for g, _ in groups]
        group_width  = 0.8 / max(len(names), 1)
        gx           = np.arange(len(group_names))
        g_offsets    = np.linspace(-(len(names) - 1) / 2 * group_width,
                                    (len(names) - 1) / 2 * group_width, len(names))

        fig_grp, ax_grp = plt.subplots(figsize=(max(5, len(group_names) * 1.8 + 2), 4))
        for (name, colour, offset) in zip(names, colours, g_offsets):
            pca = pca_data[name]
            group_accs = []
            for _, cls_list in groups:
                vals = [pca[c] for c in cls_list if c in pca and not np.isnan(pca[c])]
                group_accs.append(float(np.mean(vals)) if vals else float('nan'))
            bars = ax_grp.bar(gx + offset, group_accs, group_width,
                              label=name, color=colour, alpha=0.85)
            for bar, v in zip(bars, group_accs):
                if not np.isnan(v):
                    ax_grp.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.01,
                                f'{v:.2f}', ha='center', va='bottom', fontsize=7)

        ax_grp.set_xticks(gx)
        ax_grp.set_xticklabels(
            [f"{g}\n({', '.join(cls_list)})" for g, cls_list in groups],
            fontsize=8,
        )
        ax_grp.set_ylabel('Mean accuracy', fontsize=10)
        ax_grp.set_ylim(0, 1.12)
        ax_grp.axhline(1.0, color='grey', lw=0.6, linestyle='--')
        ax_grp.set_title(f"{param} — group accuracy summary", fontsize=11, fontweight='bold')
        ax_grp.legend(fontsize=8, framealpha=0.8)
        ax_grp.spines[['top', 'right']].set_visible(False)
        fig_grp.tight_layout()
        fig_grp.savefig(os.path.join(out_dir, 'group_accuracy_summary.png'), dpi=130)
        plt.close(fig_grp)

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


def _plot_summary_regression(all_results, out_dir):
    """Grouped bar chart: R² per param × model."""
    params = sorted(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())
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
    params = sorted(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())
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
                      latent_key='z0', out_root=None, methods=None):
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
        # Binary float labels (0.0/1.0) are treated as binary classification
        is_binary = (not is_categorical) and (set(y_tr) <= {0.0, 1.0})

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_full[tr_idx])
        X_te = scaler.transform(X_te_full[te_idx])

        if is_categorical or is_binary:
            if is_binary:
                y_tr = [int(v) for v in y_tr]
                y_te = [int(v) for v in y_te]
            if len(set(y_tr)) < 2:
                print(f"  [{param}] skipped — training set contains only one class "
                      f"(all samples are '{next(iter(set(y_tr)))}')")
                continue
            le = LabelEncoder().fit(y_tr + y_te)
            param_results = {}
            for name, mdl in _classification_models():
                if methods and name not in methods:
                    continue
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
        else:  # continuous regression
            param_results = {}
            for name, mdl in _regression_models():
                if methods and name not in methods:
                    continue
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


_ATRIAL_CLASSES      = {'avblock', 'fam', 'iab', 'lae'}
_VENTRICULAR_CLASSES = {'mi', 'lbbb', 'rbbb'}
_ALL_KNOWN_CLASSES   = _ATRIAL_CLASSES | _VENTRICULAR_CLASSES | {'sinus'}


def _remap_class(value: str, seg_type: str) -> str:
    """Remap a MedalCare-XL class label according to the segment type.

    Atrial model:
      - Any MI subclass not in the known list → 'mi', then treated as ventricular → 'sinus'
      - Ventricular classes (mi, lbbb, rbbb) → 'sinus'
      - Atrial classes and 'sinus' → unchanged

    Ventricular model:
      - Any class not in the known list → 'mi' (MI subclass)
      - Atrial classes → 'sinus'
      - Ventricular classes and 'sinus' → unchanged
    """
    if seg_type == 'atrial':
        if value not in _ALL_KNOWN_CLASSES:
            value = 'mi'               # unknown → MI subclass
        if value in _VENTRICULAR_CLASSES:
            value = 'sinus'            # ventricular → sinus for atrial model
    elif seg_type == 'ventricular':
        if value in _ATRIAL_CLASSES:
            value = 'sinus'            # atrial → sinus for ventricular model
    return value


def _remap_metadata(metadata: list, seg_type: str) -> list:
    """Apply class remapping to all records in-place (returns new list)."""
    remapped = []
    for entry in metadata:
        entry = dict(entry)
        if 'labels' in entry and 'class' in entry['labels']:
            entry['labels'] = dict(entry['labels'])
            entry['labels']['class'] = _remap_class(entry['labels']['class'], seg_type)
        remapped.append(entry)
    return remapped


# ---------------------------------------------------------------------------
# Post-training pipeline: latent collection + linear probes + wandb logging
# ---------------------------------------------------------------------------

def collect_latents(dataloader, model, task_params, args, device):
    """Run inference over *dataloader* and collect z0 (and m) latents.

    Handles both MedalCare-XL (class labels) and UK Biobank (phenotype targets).

    Returns
    -------
    latents  : dict  {'z0': ndarray [N, d]}  + optional 'm': ndarray [N, m_dim]
    metadata : list of N dicts, each with a 'labels' key
    """
    dataset_name = task_params.get('dataset', 'medalcare-xl').lower()

    # UK Biobank: load phenotype targets once
    if dataset_name != 'medalcare-xl':
        pheno_path = os.path.join(args.dataset_root, 'uk_biobank', 'phenotype_targets.pt')
        phenotype_data = torch.load(pheno_path, map_location='cpu', weights_only=False)
        eids    = phenotype_data['eids']
        targets = phenotype_data['targets'].cpu()
        columns = phenotype_data['columns']

    ecg_dataset = getattr(dataloader.dataset, 'dataset', dataloader.dataset)
    ecg_dataset.return_file_path = True

    model.eval()
    model.return_latent = True

    z0_list, m_list, metadata = [], [], []
    has_m = True   # set False if model returns m=None
    not_found = 0

    with torch.no_grad():
        for batch, batch_y, mask in tqdm(dataloader, desc="Collecting latents"):
            batch = batch.to(device)
            mask  = mask.to(device)

            z0, m = model(batch, 1, mask=mask)   # [N, d], [N, m_dim] or None
            if m is None:
                has_m = False

            patient_ids = [item[1] for item in batch_y]

            for i in range(batch.shape[0]):
                if dataset_name == 'medalcare-xl':
                    cls = batch_y[i][0]
                    z0_list.append(z0[i].detach().cpu().numpy())
                    if has_m:
                        m_list.append(m[i].detach().cpu().numpy())
                    metadata.append({
                        'patient_id': patient_ids[i],
                        'labels': {'class': cls},
                    })
                else:
                    pid = patient_ids[i]
                    try:
                        eid_idx = eids.index(pid)
                    except ValueError:
                        not_found += 1
                        continue
                    labels = {col: targets[eid_idx, j].item()
                              for j, col in enumerate(columns)}
                    z0_list.append(z0[i].detach().cpu().numpy())
                    if has_m:
                        m_list.append(m[i].detach().cpu().numpy())
                    metadata.append({'patient_id': pid, 'labels': labels})

    if not_found:
        print(f"  {not_found} patient IDs not found in phenotype targets — skipped.")

    latents = {'z0': np.stack(z0_list, axis=0)}
    if has_m and m_list:
        latents['m'] = np.stack(m_list, axis=0)

    model.return_latent = False
    return latents, metadata


def log_probe_metrics(probe_results, latent_key, seg_type, run):
    """Log linear probe metrics to a wandb run.

    Keys follow the pattern:
        probe/{seg_type}/{latent_key}/{regression|classification}/{param}/{method}/{metric}
    """
    prefix = f"probe/{seg_type}/{latent_key}" if seg_type else f"probe/{latent_key}"
    log_dict = {}
    for task_type in ('regression', 'classification'):
        for param, model_results in probe_results.get(task_type, {}).items():
            for method, res in model_results.items():
                for metric, val in res['metrics'].items():
                    if not (isinstance(val, float) and np.isnan(val)):
                        log_dict[f"{prefix}/{task_type}/{param}/{method}/{metric}"] = val
    if log_dict:
        run.log(log_dict)
        print(f"  Logged {len(log_dict)} probe metrics to wandb.")


def run_post_training_probes(args, model, device, trainset, testset, task_params, run):
    """Load best checkpoint, collect latents, run OLS linear probes, log to wandb.

    Results are saved to:
        {args.save}/latents/           — z0/m arrays + metadata JSON
        {args.save}/finetune_results/  — per-param metrics.json + plots

    The saved latent files follow the finetune.py naming convention so the
    standalone ``finetune.py`` can be re-run on them for full probe analysis.
    """
    ckpt_path = os.path.join(args.save, 'model.pth')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path} — skipping post-training probes.")
        return

    print("\n========== Post-training linear probes ==========")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

    dataset_name = task_params.get('dataset', 'medalcare-xl').lower()
    seg_type     = getattr(args, 'segment_type', None)

    with np.errstate(all='ignore'):
        print("Collecting train latents...")
        tr_latents, tr_metadata = collect_latents(trainset, model, task_params, args, device)
        print("Collecting test latents...")
        te_latents, te_metadata = collect_latents(testset,  model, task_params, args, device)

    # Persist to disk — finetune.py _load_split() can read these back
    latents_dir = os.path.join(args.save, 'latents')
    os.makedirs(latents_dir, exist_ok=True)
    np.savez(os.path.join(latents_dir, 'train_latents.npz'), **tr_latents)
    np.savez(os.path.join(latents_dir, 'test_latents.npz'),  **te_latents)
    with open(os.path.join(latents_dir, 'train_metadata.json'), 'w') as f:
        json.dump(tr_metadata, f, indent=2)
    with open(os.path.join(latents_dir, 'test_metadata.json'), 'w') as f:
        json.dump(te_metadata, f, indent=2)
    print(f"  Saved latents to {latents_dir}")

    # Remap MedalCare-XL class labels for the segment type
    if dataset_name == 'medalcare-xl' and seg_type:
        tr_metadata = _remap_metadata(tr_metadata, seg_type)
        te_metadata = _remap_metadata(te_metadata, seg_type)

    # Build combined latent key when modulator is present
    has_m = 'm' in tr_latents and 'm' in te_latents
    if has_m:
        tr_latents['z0_m'] = np.concatenate([tr_latents['z0'], tr_latents['m']], axis=1)
        te_latents['z0_m'] = np.concatenate([te_latents['z0'], te_latents['m']], axis=1)

    run_label     = seg_type if seg_type else 'all_classes'
    finetune_root = os.path.join(args.save, 'finetune_results', run_label)

    latent_keys = ['z0'] + (['m', 'z0_m'] if has_m else [])
    for lkey in latent_keys:
        print(f"\n=== Linear probes ({lkey}) ===")
        with np.errstate(all='ignore'):
            probe_results = run_linear_probes(
                tr_latents, tr_metadata,
                te_latents, te_metadata,
                latent_key=lkey,
                out_root=os.path.join(finetune_root, lkey),
                methods={'ols'},
            )
        log_probe_metrics(probe_results, lkey, seg_type, run)

    print("========== Post-training probes complete ==========\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run linear probes on pre-saved latents.')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Directory containing the saved latent .npz and metadata .json files.')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        help='Splits to load (default: train test).')
    parser.add_argument('--seg_type', type=str, choices=['atrial', 'ventricular'], default=None,
                        help='Segment type of the model. When set, class labels in MedalCare-XL '
                             'metadata are remapped so that out-of-domain classes become "sinus". '
                             'Also used to name the output directory so runs do not overwrite each other.')
    parser.add_argument('--methods', nargs='+', default=None,
                        choices=['ols', 'ridge', 'lasso', 'mlp'],
                        help='Probe methods to run. Defaults to ols only if not specified.')
    args = parser.parse_args()

    root_dir = args.root_dir
    methods  = set(args.methods) if args.methods else {'ols'}

    # Output directory: named by seg_type when provided so runs don't overwrite each other;
    # otherwise use a timestamp to guarantee uniqueness.
    if args.seg_type:
        run_label = args.seg_type
    else:
        run_label = 'all_classes'

    finetune_root = os.path.normpath(os.path.join(root_dir, 'finetune_results', run_label))

    latents_dir = os.path.join(root_dir, 'latents')
    print("Loading latents...")
    latents_dict = {}
    for split in args.splits:
        latents, metadata = _load_split(latents_dir, split)
        if args.seg_type:
            metadata = _remap_metadata(metadata, args.seg_type)
        latents_dict[split] = {'latents': latents, 'metadata': metadata}

    train_split = args.splits[0]
    test_split  = args.splits[-1]

    # Require m in both splits before running m-dependent probes
    has_m = all('m' in latents_dict[s]['latents'] for s in [train_split, test_split])

    if has_m:
        for split in [train_split, test_split]:
            lats = latents_dict[split]['latents']
            lats['z0_m'] = np.concatenate([lats['z0'], lats['m']], axis=1)

    def _lats(split):
        return latents_dict[split]['latents']

    def _meta(split):
        return latents_dict[split]['metadata']

    print(f"\nOutput directory: {finetune_root}")
    print(f"Methods: {sorted(methods)}\n")

    print("=== Linear probes (z0) ===")
    run_linear_probes(
        _lats(train_split), _meta(train_split),
        _lats(test_split),  _meta(test_split),
        latent_key='z0',
        out_root=os.path.join(finetune_root, 'z0'),
        methods=methods,
    )

    if has_m:
        print("\n=== Linear probes (m) ===")
        run_linear_probes(
            _lats(train_split), _meta(train_split),
            _lats(test_split),  _meta(test_split),
            latent_key='m',
            out_root=os.path.join(finetune_root, 'm'),
            methods=methods,
        )

        print("\n=== Linear probes (z0 + m combined) ===")
        run_linear_probes(
            _lats(train_split), _meta(train_split),
            _lats(test_split),  _meta(test_split),
            latent_key='z0_m',
            out_root=os.path.join(finetune_root, 'z0_m'),
            methods=methods,
        )

