import argparse
import json
import math
import os
import re
from collections import Counter

import matplotlib.cm as mpl_cm
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import (
    skew as sp_skew, kurtosis as sp_kurtosis, norm as sp_norm,
    kruskal as sp_kruskal,
)
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV,
    LogisticRegression, LogisticRegressionCV,
)
from sklearn.metrics import (
    r2_score, mean_squared_error,
    roc_auc_score, accuracy_score, f1_score,
    adjusted_rand_score,
    silhouette_score, silhouette_samples,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder



# ---------------------------------------------------------------------------
# MedalCare-XL UID helpers
# ---------------------------------------------------------------------------

_T_PREFIX_RE = re.compile(r'^T\d+_')

def _medalcare_uid_from_stem(stem: str) -> str:
    """Strip the run-directory prefix from a MedalCare-XL .pth file stem.

    The preprocessing pipeline saves files as  ``T{run_id}_{session_id}_{cls}.pth``
    (e.g. ``T49_S62_000069_lae.pth``), but the ALADIN metadata JSON is keyed by
    ``{session_id}_{cls}`` (e.g. ``S62_000069_lae``).  If the stem does not start
    with ``T\\d+_`` it is returned unchanged.
    """
    stem = os.path.splitext(os.path.basename(stem))[0]
    return _T_PREFIX_RE.sub('', stem)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_split(latents_dir: str, split: str,
                aladin_metadata: dict | None = None) -> tuple[dict, list]:
    """Load pre-saved latents and metadata for one split.

    Supports two file naming conventions:
      - finetune.py style:    {split}_latents.npz  /  {split}_metadata.json
      - inference_analysis.py style: latent_tensors_{split}.npz  / latent_meta_dict_{split}.json

    If *aladin_metadata* (a dict keyed by UID as produced by aladin_preprocess.py) is
    provided, labels are re-derived from it using the 'uid' field stored in each metadata
    entry.  This overrides whatever labels were saved in the metadata JSON.
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

    # Handle ALADIN preprocess format: a dict keyed by UID rather than a list.
    # Each value is {labels: {...}, p_wave_estimated: ..., segment_lengths: {...}}.
    # Convert to the list format used throughout finetune.py.
    if isinstance(raw_meta, dict):
        raw_meta = [
            {'uid': uid, 'patient_id': uid, 'labels': entry.get('labels', {})}
            for uid, entry in raw_meta.items()
        ]

    # Normalise metadata: inference_analysis.py stores labels flat under 'labels',
    # but for MedalCare-XL the labels dict only has 'class' and 'patient_id'.
    # Ensure every entry has a 'labels' key.
    metadata = []
    for entry in raw_meta:
        if 'labels' not in entry:
            # Flatten all non-latent fields into labels
            entry = dict(entry)
            entry['labels'] = {k: v for k, v in entry.items()
                               if k not in ('filename', 'patient_id', 'uid')}
        metadata.append(entry)

    # Override labels from ALADIN metadata when provided, joining on the 'uid' field.
    if aladin_metadata is not None:
        not_found = 0
        for entry in metadata:
            uid = _medalcare_uid_from_stem(entry.get('uid') or entry.get('filename') or '')
            aladin_entry = aladin_metadata.get(uid)
            if aladin_entry is not None:
                entry['labels'] = aladin_entry.get('labels', {})
            else:
                print(uid)
                not_found += 1
        if not_found:
            print(f"  [{split}] {not_found} UIDs not found in ALADIN metadata — labels left as-is.")

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

    all_dataset_stats: dict = {}   # param -> stats dict, saved to a single JSON at the end

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

        # ── Dataset statistics ────────────────────────────────────────────────
        dstats = _compute_dataset_stats(y_tr, y_te, is_categorical, is_binary)
        all_dataset_stats[param] = dstats
        _print_dataset_stats(param, dstats)

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
                _plot_dataset_stats(param, dstats, pdir)
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
                _plot_dataset_stats(param, dstats, pdir)
                _save_metrics_json(param_results, pdir)

    # ── Summary plots + consolidated dataset_stats.json ───────────────────────
    if out_root:
        if reg_results:
            sdir = os.path.join(out_root, 'regression', '_summary')
            os.makedirs(sdir, exist_ok=True)
            _plot_summary_regression(reg_results, sdir)
        if clf_results:
            sdir = os.path.join(out_root, 'classification', '_summary')
            os.makedirs(sdir, exist_ok=True)
            _plot_summary_classification(clf_results, sdir)

        # Single file with all parameters' dataset statistics side-by-side
        stats_path = os.path.join(out_root, 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(all_dataset_stats, f, indent=2)
        print(f"\n  Dataset statistics saved → {stats_path}")

    return {'regression': reg_results, 'classification': clf_results,
            'dataset_stats': all_dataset_stats}


def _compute_dataset_stats(y_tr, y_te, is_categorical: bool, is_binary: bool) -> dict:
    """Return descriptive statistics for one parameter's train / test label sets.

    For continuous parameters: mean, std, min, max, median, Q1, Q3, IQR,
    skewness, kurtosis, and the % of zero-valued samples (useful for sparse
    phenotypes such as disease indicators stored as floats).

    For categorical / binary parameters: class counts and class frequencies for
    both splits, plus the number of unique classes seen in each split.
    """
    stats: dict = {
        'n_train': len(y_tr),
        'n_test':  len(y_te),
        'n_total': len(y_tr) + len(y_te),
        'type':    'classification' if (is_categorical or is_binary) else 'regression',
    }

    if is_categorical or is_binary:
        tr_counts   = dict(Counter(str(v) for v in y_tr))
        te_counts   = dict(Counter(str(v) for v in y_te))
        all_classes = sorted(set(tr_counts) | set(te_counts))
        tr_freq     = {c: round(tr_counts.get(c, 0) / max(len(y_tr), 1), 4) for c in all_classes}
        te_freq     = {c: round(te_counts.get(c, 0) / max(len(y_te), 1), 4) for c in all_classes}
        stats.update({
            'n_classes':          len(all_classes),
            'classes':            all_classes,
            'train_class_count':  tr_counts,
            'test_class_count':   te_counts,
            'train_class_freq':   tr_freq,
            'test_class_freq':    te_freq,
        })
    else:
        arr_tr = np.array(y_tr, dtype=float)
        arr_te = np.array(y_te, dtype=float)

        def _desc(arr):
            q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            return {
                'mean':     float(arr.mean()),
                'std':      float(arr.std()),
                'min':      float(arr.min()),
                'max':      float(arr.max()),
                'median':   float(np.median(arr)),
                'q1':       q1,
                'q3':       q3,
                'iqr':      round(q3 - q1, 6),
                'skewness': float(sp_skew(arr)),
                'kurtosis': float(sp_kurtosis(arr)),
                'pct_zero': round(float((arr == 0).mean()) * 100, 2),
            }

        stats['train'] = _desc(arr_tr)
        stats['test']  = _desc(arr_te)

    return stats


def _print_dataset_stats(param: str, stats: dict) -> None:
    """Print a compact, human-readable dataset statistics block."""
    n_tr, n_te = stats['n_train'], stats['n_test']
    kind = stats['type']
    print(f"\n  ── [{param}] dataset stats ({kind}) ──")
    print(f"     train n={n_tr}   test n={n_te}   total={n_tr + n_te}")

    if kind == 'classification':
        print(f"     classes ({stats['n_classes']}): {stats['classes']}")
        print("     train class freq: "
              + "  ".join(f"{c}={stats['train_class_freq'][c]:.3f}"
                          for c in stats['classes']))
        print("     test  class freq: "
              + "  ".join(f"{c}={stats['test_class_freq'][c]:.3f}"
                          for c in stats['classes']))
    else:
        for split_name, key in [('train', 'train'), ('test', 'test')]:
            d = stats[key]
            print(f"     {split_name:5s}: mean={d['mean']:.4f}  std={d['std']:.4f}  "
                  f"[{d['min']:.4f}, {d['max']:.4f}]  "
                  f"median={d['median']:.4f}  IQR={d['iqr']:.4f}  "
                  f"skew={d['skewness']:.3f}  kurt={d['kurtosis']:.3f}  "
                  f"pct_zero={d['pct_zero']:.1f}%")


def _make_dataset_stats_fig(param: str, stats: dict):
    """Build and return a matplotlib Figure for the dataset stats of *param*.
    Caller is responsible for closing it with plt.close(fig).
    """
    kind = stats['type']

    if kind == 'classification':
        classes = stats['classes']
        tr_freq = [stats['train_class_freq'][c] for c in classes]
        te_freq = [stats['test_class_freq'][c]  for c in classes]
        x       = np.arange(len(classes))
        width   = 0.35

        fig, ax = plt.subplots(figsize=(max(5, len(classes) * 0.8 + 2), 4))
        ax.bar(x - width / 2, tr_freq, width, label=f'train (n={stats["n_train"]})', alpha=0.85)
        ax.bar(x + width / 2, te_freq, width, label=f'test  (n={stats["n_test"]})',  alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('Frequency')
        ax.set_ylim(0, 1.1)
        ax.set_title(f'{param} — class distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (split_name, key) in zip(axes, [('train', 'train'), ('test', 'test')]):
            d      = stats[key]
            n      = stats[f'n_{split_name}']
            lo, hi = d['min'], d['max']
            xs     = np.linspace(lo, hi, 300)
            ys     = sp_norm.pdf(xs, loc=d['mean'], scale=max(d['std'], 1e-8))
            ax.plot(xs, ys, lw=2)
            ax.axvline(d['mean'],   color='red',    lw=1.2, linestyle='--',
                       label=f"mean={d['mean']:.3f}")
            ax.axvline(d['median'], color='orange', lw=1.2, linestyle=':',
                       label=f"med={d['median']:.3f}")
            ax.axvspan(d['q1'], d['q3'], alpha=0.15, color='green', label='IQR')
            ax.set_title(f'{split_name} (n={n})\n'
                         f'std={d["std"]:.3f}  skew={d["skewness"]:.2f}  kurt={d["kurtosis"]:.2f}',
                         fontsize=9)
            ax.set_xlabel(param, fontsize=9)
            ax.legend(fontsize=7, framealpha=0.7)
            ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(f'{param} — value distribution', fontsize=11, fontweight='bold')

    fig.tight_layout()
    return fig


def _plot_dataset_stats(param: str, stats: dict, out_dir: str) -> None:
    """Save a distribution plot for this parameter to *out_dir*/dataset_stats.png."""
    fig = _make_dataset_stats_fig(param, stats)
    fig.savefig(os.path.join(out_dir, 'dataset_stats.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# GMM clustering
# ---------------------------------------------------------------------------

def run_gmm_clustering(
    train_latents: dict,
    train_metadata: list,
    test_latents: dict,
    test_metadata: list,
    latent_key: str = 'z0',
    dataset_name: str = 'medalcare-xl',
    n_clusters: int | None = None,
    pca_dim: int = 10,
    out_root: str | None = None,
    run=None,
) -> dict:
    """Fit a diagonal-covariance GMM in PCA-reduced latent space and evaluate clustering.

    Pipeline
    --------
    1. StandardScaler fit on train, applied to both splits.
    2. PCA (pca_dim components) fit on train, applied to both splits.
    3. GMM (covariance_type='diag', n_components=n_clusters) fit on *train*.
    4. Cluster assignment predicted on *test*.
    5. Per-cluster statistics:
       - Mean ± std of every continuous label parameter.
       - Epsilon-squared effect size from Kruskal-Wallis test per parameter.
       - OLS regression of each continuous parameter on cluster assignment (one-hot).
    6. Silhouette score (global + per-cluster mean) on test PCA embeddings.
    7. ARI vs true class labels (MedalCare-XL only, where 'class' label exists).

    Parameters
    ----------
    n_clusters  : inferred from unique 'class' labels in train metadata when None
                  (MedalCare-XL behaviour); must be supplied explicitly for UK Biobank.
    pca_dim     : number of PCA components to retain before clustering.
    dataset_name: used to gate ARI computation ('medalcare-xl' only).

    Directory layout (when out_root is given)
    ------------------------------------------
    {out_root}/clustering/{latent_key}/
      gmm_results.json          — all metrics and per-cluster stats
      cluster_scatter.png       — PCA-2D scatter coloured by predicted cluster
      cluster_scatter_true.png  — same scatter coloured by true class (MC-XL only)
      silhouette.png            — per-cluster silhouette bar chart
      param_epsilon_squared.png — effect-size bar chart across parameters
      cluster_{k}/
        param_distributions.png — violin plots of continuous params in this cluster
    """
    out_dir = None
    if out_root:
        out_dir = os.path.join(out_root, 'clustering', latent_key)
        os.makedirs(out_dir, exist_ok=True)

    is_medalcare = dataset_name.lower() == 'medalcare-xl'

    # ── 1. Extract latent arrays ──────────────────────────────────────────────
    X_tr = train_latents[latent_key].astype(np.float64)
    X_te = test_latents[latent_key].astype(np.float64)

    # ── 2. Standardise ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    # ── 3. PCA (fit on train) ─────────────────────────────────────────────────
    actual_pca_dim = min(pca_dim, X_tr.shape[1], X_tr.shape[0])
    pca = PCA(n_components=actual_pca_dim, random_state=42)
    Z_tr = pca.fit_transform(X_tr)
    Z_te = pca.transform(X_te)
    explained_var = float(pca.explained_variance_ratio_.sum())
    print(f"  [GMM/{latent_key}] PCA {X_tr.shape[1]}→{actual_pca_dim}d, "
          f"explained variance: {explained_var:.3f}")

    # ── 4. Determine number of clusters ───────────────────────────────────────
    if n_clusters is None:
        # Count unique non-None class labels in train metadata
        tr_classes = sorted({
            str(m['labels'].get('class', ''))
            for m in train_metadata
            if m.get('labels', {}).get('class') not in (None, '', 'None')
        })
        n_clusters = max(len(tr_classes), 2)
        print(f"  [GMM/{latent_key}] Inferred {n_clusters} clusters from train classes: {tr_classes}")
    else:
        tr_classes = None

    # ── 5. Fit GMM on train ───────────────────────────────────────────────────
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        max_iter=300,
        random_state=42,
        n_init=5,
    )
    gmm.fit(Z_tr)
    labels_te = gmm.predict(Z_te)          # cluster assignment per test sample

    # ── 6. Silhouette ─────────────────────────────────────────────────────────
    sil_global = float(silhouette_score(Z_te, labels_te))
    sil_sample = silhouette_samples(Z_te, labels_te)
    sil_per_cluster = {
        int(k): float(sil_sample[labels_te == k].mean())
        for k in range(n_clusters)
        if (labels_te == k).sum() > 0
    }
    print(f"  [GMM/{latent_key}] Silhouette (global): {sil_global:.4f}")

    # ── 7. ARI (MedalCare-XL only) ────────────────────────────────────────────
    ari = None
    true_classes_te = None
    if is_medalcare:
        raw_true = [m.get('labels', {}).get('class') for m in test_metadata]
        if any(v is not None for v in raw_true):
            le_cls = LabelEncoder()
            valid_mask = np.array([v is not None and str(v) not in ('', 'None')
                                   for v in raw_true])
            true_enc  = le_cls.fit_transform([str(v) for v, ok in zip(raw_true, valid_mask) if ok])
            pred_valid = labels_te[valid_mask]
            ari = float(adjusted_rand_score(true_enc, pred_valid))
            true_classes_te = np.array([str(v) if ok else 'unknown'
                                         for v, ok in zip(raw_true, valid_mask)])
            print(f"  [GMM/{latent_key}] ARI: {ari:.4f}")

    # ── 8. Per-cluster label statistics + Kruskal-Wallis ε² ──────────────────
    # Collect all continuous parameters present in test metadata
    all_params: set[str] = set()
    for m in test_metadata:
        for k, v in m.get('labels', {}).items():
            if k in ('class', 'patient_id'):
                continue
            if isinstance(v, (int, float)) and not (isinstance(v, bool)):
                all_params.add(k)

    # Build param → array of values aligned with test samples (NaN where missing)
    param_values: dict[str, np.ndarray] = {}
    for param in all_params:
        arr = np.array([
            float(m.get('labels', {}).get(param, float('nan')))
            if isinstance(m.get('labels', {}).get(param), (int, float))
            else float('nan')
            for m in test_metadata
        ])
        param_values[param] = arr

    # Per-cluster mean/std
    cluster_stats: dict[int, dict] = {}
    for k in range(n_clusters):
        mask = labels_te == k
        cstats: dict = {'n': int(mask.sum()), 'params': {}}
        for param, arr in param_values.items():
            vals = arr[mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                cstats['params'][param] = {'mean': None, 'std': None, 'n_valid': 0}
            else:
                cstats['params'][param] = {
                    'mean':    round(float(valid.mean()), 6),
                    'std':     round(float(valid.std()),  6),
                    'median':  round(float(np.median(valid)), 6),
                    'n_valid': int(len(valid)),
                }
        cluster_stats[k] = cstats

    # Kruskal-Wallis + epsilon-squared per parameter
    # ε² = (H - k + 1) / (N - k)  where H = KW stat, k = n_clusters, N = total valid
    eps_sq_results: dict[str, dict] = {}
    for param, arr in param_values.items():
        groups = []
        for k in range(n_clusters):
            vals = arr[labels_te == k]
            valid = vals[~np.isnan(vals)]
            groups.append(valid)

        groups_nonempty = [g for g in groups if len(g) >= 2]
        if len(groups_nonempty) < 2:
            eps_sq_results[param] = {'H': None, 'p_value': None, 'epsilon_squared': None}
            continue

        try:
            stat, pval = sp_kruskal(*groups_nonempty)
            N = sum(len(g) for g in groups_nonempty)
            k_used = len(groups_nonempty)
            eps_sq = (stat - k_used + 1) / (N - k_used) if N > k_used else float('nan')
            eps_sq_results[param] = {
                'H':               round(float(stat),   6),
                'p_value':         round(float(pval),   8),
                'epsilon_squared': round(float(eps_sq), 6),
            }
        except Exception as e:
            eps_sq_results[param] = {'H': None, 'p_value': None, 'epsilon_squared': None,
                                      'error': str(e)}

    # ── 9. OLS regression of each continuous param on cluster one-hot ─────────
    cluster_regression: dict[str, dict] = {}
    if n_clusters >= 2:
        # One-hot encode cluster assignments
        ohe = np.zeros((len(labels_te), n_clusters), dtype=np.float64)
        ohe[np.arange(len(labels_te)), labels_te] = 1.0
        # Drop last column to avoid perfect multicollinearity
        X_reg = ohe[:, :-1]
        for param, arr in param_values.items():
            valid_mask = ~np.isnan(arr)
            if valid_mask.sum() < n_clusters + 5:
                cluster_regression[param] = {'r2': None, 'mse': None}
                continue
            lr = LinearRegression()
            lr.fit(X_reg[valid_mask], arr[valid_mask])
            y_pred = lr.predict(X_reg[valid_mask])
            cluster_regression[param] = {
                'r2':  round(float(r2_score(arr[valid_mask], y_pred)),             6),
                'mse': round(float(mean_squared_error(arr[valid_mask], y_pred)),   6),
            }

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    if out_dir:
        _plot_gmm_scatter(Z_te, labels_te, n_clusters, out_dir, suffix='predicted')
        if true_classes_te is not None:
            _plot_gmm_scatter_true(Z_te, true_classes_te, out_dir)
        _plot_gmm_silhouette(sil_sample, labels_te, n_clusters, sil_global, out_dir)
        _plot_gmm_epsilon_squared(eps_sq_results, out_dir)
        _plot_gmm_cluster_violins(param_values, labels_te, n_clusters, out_dir)

    # ── 11. Wandb logging ─────────────────────────────────────────────────────
    if run is not None:
        _log_gmm_to_wandb(run, latent_key, sil_global, sil_per_cluster, ari,
                          eps_sq_results, out_dir)

    # ── 12. Assemble and save results ─────────────────────────────────────────
    results = {
        'latent_key':          latent_key,
        'n_clusters':          n_clusters,
        'pca_dim':             actual_pca_dim,
        'pca_explained_var':   round(explained_var, 6),
        'n_train':             len(Z_tr),
        'n_test':              len(Z_te),
        'silhouette_global':   round(sil_global, 6),
        'silhouette_per_cluster': {str(k): round(v, 6)
                                   for k, v in sil_per_cluster.items()},
        'ari':                 round(ari, 6) if ari is not None else None,
        'cluster_sizes':       {str(k): int((labels_te == k).sum())
                                for k in range(n_clusters)},
        'cluster_stats':       {str(k): v for k, v in cluster_stats.items()},
        'epsilon_squared':     eps_sq_results,
        'cluster_regression':  cluster_regression,
    }

    if out_dir:
        results_path = os.path.join(out_dir, 'gmm_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  [GMM/{latent_key}] Results → {results_path}")

    _print_gmm_summary(results)
    return results


# ── GMM plotting helpers ───────────────────────────────────────────────────────

def _plot_gmm_scatter(Z: np.ndarray, labels: np.ndarray, n_clusters: int,
                      out_dir: str, suffix: str = 'predicted') -> None:
    """2-D PCA scatter coloured by cluster assignment."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = mpl_cm.get_cmap('tab20', n_clusters)
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.5,
                   color=cmap(k), label=f'C{k} (n={mask.sum()})', rasterized=True)
    ax.set_xlabel('PC 1', fontsize=9)
    ax.set_ylabel('PC 2', fontsize=9)
    ax.set_title(f'GMM clusters ({suffix})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, markerscale=2, ncol=max(1, n_clusters // 8),
              framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'cluster_scatter_{suffix}.png'), dpi=130)
    plt.close(fig)


def _plot_gmm_scatter_true(Z: np.ndarray, true_classes: np.ndarray,
                           out_dir: str) -> None:
    """2-D PCA scatter coloured by true class labels."""
    unique_cls = sorted(set(true_classes))
    cmap = mpl_cm.get_cmap('tab20', len(unique_cls))
    cls_idx = {c: i for i, c in enumerate(unique_cls)}

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in unique_cls:
        mask = true_classes == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.5,
                   color=cmap(cls_idx[cls]), label=cls, rasterized=True)
    ax.set_xlabel('PC 1', fontsize=9)
    ax.set_ylabel('PC 2', fontsize=9)
    ax.set_title('True class labels', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, markerscale=2, ncol=max(1, len(unique_cls) // 8),
              framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cluster_scatter_true.png'), dpi=130)
    plt.close(fig)


def _plot_gmm_silhouette(sil_sample: np.ndarray, labels: np.ndarray,
                         n_clusters: int, sil_global: float, out_dir: str) -> None:
    """Horizontal silhouette plot (one bar per sample, grouped by cluster)."""
    fig, ax = plt.subplots(figsize=(7, max(4, n_clusters * 0.6)))
    cmap   = mpl_cm.get_cmap('tab20', n_clusters)
    y_lo   = 0
    yticks, ytick_labels = [], []

    for k in range(n_clusters):
        vals = np.sort(sil_sample[labels == k])
        size = len(vals)
        y_hi = y_lo + size
        ax.barh(np.arange(y_lo, y_hi), vals, height=1.0,
                color=cmap(k), edgecolor='none', alpha=0.85)
        yticks.append(y_lo + size / 2)
        ytick_labels.append(f'C{k}\n(n={size})')
        y_lo = y_hi + 4

    ax.axvline(sil_global, color='red', lw=1.2, linestyle='--',
               label=f'global={sil_global:.3f}')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7)
    ax.set_xlabel('Silhouette coefficient', fontsize=9)
    ax.set_title('Silhouette plot — GMM clusters', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'silhouette.png'), dpi=130)
    plt.close(fig)


def _plot_gmm_epsilon_squared(eps_sq_results: dict, out_dir: str) -> None:
    """Bar chart of ε² effect sizes, sorted descending."""
    params = [p for p, v in eps_sq_results.items()
              if v.get('epsilon_squared') is not None]
    if not params:
        return

    params  = sorted(params, key=lambda p: eps_sq_results[p]['epsilon_squared'],
                     reverse=True)
    eps_vals = [eps_sq_results[p]['epsilon_squared'] for p in params]
    pvals    = [eps_sq_results[p]['p_value']         for p in params]

    fig, ax = plt.subplots(figsize=(max(6, len(params) * 0.55 + 2), 4))
    colours = ['#d62728' if p < 0.05 else '#aec7e8' for p in pvals]
    bars = ax.bar(range(len(params)), eps_vals, color=colours, edgecolor='white')

    # Significance asterisks
    for _, (bar, p) in enumerate(zip(bars, pvals)):
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = ''
        if marker:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    marker, ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('ε² (Kruskal-Wallis effect size)', fontsize=9)
    ax.set_title('Effect size per parameter across GMM clusters', fontsize=11,
                 fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    ax.legend(handles=[mpl_patches.Patch(color='#d62728', label='p < 0.05'),
                        mpl_patches.Patch(color='#aec7e8', label='p ≥ 0.05')],
              fontsize=7, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'param_epsilon_squared.png'), dpi=130)
    plt.close(fig)


def _plot_gmm_cluster_violins(param_values: dict, labels: np.ndarray,
                               n_clusters: int, out_dir: str) -> None:
    """For each cluster: violin plots of every continuous parameter."""
    params = sorted(param_values.keys())
    if not params:
        return

    cmap = mpl_cm.get_cmap('tab20', n_clusters)
    for k in range(n_clusters):
        mask = labels == k
        cdir = os.path.join(out_dir, f'cluster_{k}')
        os.makedirs(cdir, exist_ok=True)

        data_for_plot = []
        labels_for_plot = []
        for param in params:
            vals = param_values[param][mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                data_for_plot.append(valid)
                labels_for_plot.append(param)

        if not data_for_plot:
            continue

        n_p  = len(data_for_plot)
        ncol = min(4, n_p)
        nrow = math.ceil(n_p / ncol)
        fig, axes = plt.subplots(nrow, ncol,
                                  figsize=(ncol * 3.5, nrow * 3),
                                  squeeze=False)
        for idx, (vals, pname) in enumerate(zip(data_for_plot, labels_for_plot)):
            ax  = axes[idx // ncol][idx % ncol]
            vp  = ax.violinplot(vals, positions=[0], showmedians=True)
            for pc in vp['bodies']:
                pc.set_facecolor(cmap(k))
                pc.set_alpha(0.75)
            ax.set_xticks([])
            ax.set_title(pname, fontsize=8)
            ax.set_ylabel('value', fontsize=7)
            ax.spines[['top', 'right']].set_visible(False)

        # Hide unused axes
        for idx in range(len(data_for_plot), nrow * ncol):
            axes[idx // ncol][idx % ncol].set_visible(False)

        fig.suptitle(f'Cluster {k}  (n={mask.sum()}) — parameter distributions',
                     fontsize=10, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(cdir, 'param_distributions.png'), dpi=120)
        plt.close(fig)


def _print_gmm_summary(results: dict) -> None:
    """Print a concise human-readable summary to stdout."""
    lkey  = results['latent_key']
    K     = results['n_clusters']
    sizes = results['cluster_sizes']
    print(f"\n{'─'*60}")
    print(f"GMM CLUSTERING  [{lkey}]  K={K}  "
          f"PCA {results['pca_dim']}d  "
          f"(var={results['pca_explained_var']:.3f})")
    print(f"  n_train={results['n_train']}  n_test={results['n_test']}")
    print(f"  Silhouette (global): {results['silhouette_global']:.4f}")
    for k in range(K):
        sc = results['silhouette_per_cluster'].get(str(k), float('nan'))
        print(f"    C{k:2d}: n={sizes.get(str(k), 0):5d}  sil={sc:.4f}")
    if results['ari'] is not None:
        print(f"  ARI vs true labels : {results['ari']:.4f}")

    print(f"\n  Top-5 parameters by ε² (Kruskal-Wallis):")
    eps = {p: v for p, v in results['epsilon_squared'].items()
           if v.get('epsilon_squared') is not None}
    for p, v in sorted(eps.items(), key=lambda x: x[1]['epsilon_squared'],
                        reverse=True)[:5]:
        sig = '*' if v['p_value'] < 0.05 else ''
        print(f"    {p:30s}  ε²={v['epsilon_squared']:.4f}  "
              f"H={v['H']:.2f}  p={v['p_value']:.4g}{sig}")
    print(f"{'─'*60}\n")


def _log_gmm_to_wandb(run, latent_key: str, sil_global: float,
                       sil_per_cluster: dict, ari: float | None,
                       eps_sq_results: dict, out_dir: str | None) -> None:
    """Log GMM metrics and images to an active wandb run."""
    import wandb as _wandb
    panel    = 'clustering'
    log_dict = {
        f'{panel}/{latent_key}/silhouette_global': sil_global,
    }
    for k, v in sil_per_cluster.items():
        log_dict[f'{panel}/{latent_key}/silhouette_cluster_{k}'] = v
    if ari is not None:
        log_dict[f'{panel}/{latent_key}/ari'] = ari

    for param, v in eps_sq_results.items():
        if v.get('epsilon_squared') is not None:
            log_dict[f'{panel}/{latent_key}/epsilon_sq/{param}'] = v['epsilon_squared']
            log_dict[f'{panel}/{latent_key}/kw_pvalue/{param}']  = v['p_value']

    if out_dir:
        for fname, key_suffix in [
            ('cluster_scatter_predicted.png', 'scatter_predicted'),
            ('cluster_scatter_true.png',      'scatter_true'),
            ('silhouette.png',                'silhouette'),
            ('param_epsilon_squared.png',     'epsilon_squared'),
        ]:
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                log_dict[f'{panel}/{latent_key}/{key_suffix}'] = _wandb.Image(fpath)

    run.log(log_dict)
    print(f"  Logged {len(log_dict)} GMM metrics to wandb under '{panel}' panel.")


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

def collect_latents(dataloader, model, task_params, args, device,
                    aladin_metadata: dict | None = None):
    """Run inference over *dataloader* and collect z0 (and m) latents.

    Handles MedalCare-XL (class labels), UK Biobank (phenotype targets), and
    ALADIN-preprocessed datasets (labels from *aladin_metadata*).

    Parameters
    ----------
    aladin_metadata : dict | None
        When provided, must be a dict keyed by UID as produced by
        aladin_preprocess.py: ``{uid: {labels: {...}, ...}}``.
        Labels are looked up by matching the file stem of each loaded .pth file
        against the UID keys.  When provided, the phenotype_targets.pt fallback
        for non-MedalCare splits is skipped.

    Returns
    -------
    latents  : dict  {'z0': ndarray [N, d]}  + optional 'm': ndarray [N, m_dim]
    metadata : list of N dicts, each with 'uid', 'patient_id', and 'labels' keys
    """
    dataset_name = task_params.get('dataset', 'medalcare-xl').lower()

    # UK Biobank: load phenotype targets once (skipped when aladin_metadata is given)
    phenotype_data = None
    if dataset_name != 'medalcare-xl' and aladin_metadata is None:
        pheno_path = os.path.join(args.dataset_root, 'uk_biobank', 'phenotype_targets.pt')
        phenotype_data = torch.load(pheno_path, map_location='cpu', weights_only=False)
        eids    = phenotype_data['eids']
        targets = phenotype_data['targets'].cpu()
        columns = phenotype_data['columns']

    # Unwrap Subset wrappers to reach the underlying ECGDataset (which has file_paths)
    ecg_dataset = dataloader.dataset
    while not hasattr(ecg_dataset, 'file_paths'):
        ecg_dataset = ecg_dataset.dataset
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
                # ── ALADIN metadata path (takes priority over dataset_name logic) ──
                if aladin_metadata is not None:
                    file_path = batch_y[i][2]   # set by return_file_path=True
                    uid = _medalcare_uid_from_stem(os.path.splitext(os.path.basename(file_path))[0])
                    aladin_entry = aladin_metadata.get(uid)
                    if aladin_entry is None:
                        not_found += 1
                        labels = {}
                    else:
                        labels = aladin_entry.get('labels', {})
                    z0_list.append(z0[i].detach().cpu().numpy())
                    if has_m:
                        m_list.append(m[i].detach().cpu().numpy())
                    metadata.append({'uid': uid, 'patient_id': uid, 'labels': labels})

                # ── MedalCare-XL: derive UID from file path, labels from metadata ──
                elif dataset_name == 'medalcare-xl':
                    file_path = batch_y[i][2]
                    uid       = _medalcare_uid_from_stem(os.path.splitext(os.path.basename(file_path))[0])
                    uid_parts = uid.split('_')
                    run_id    = uid_parts[0]
                    _cls      = '_'.join(uid_parts[2:])   # everything after run_id_session_id

                    # Look up full labels from ALADIN metadata when available;
                    # otherwise fall back to the class parsed from the UID.
                    aladin_entry = aladin_metadata.get(uid) if aladin_metadata is not None else None
                    if aladin_entry is not None:
                        labels = aladin_entry.get('labels', {})
                    else:
                        labels = {'class': _cls, 'patient_id': run_id}

                    z0_list.append(z0[i].detach().cpu().numpy())
                    if has_m:
                        m_list.append(m[i].detach().cpu().numpy())
                    metadata.append({'uid': uid, 'patient_id': run_id, 'labels': labels})

                # ── UK Biobank / other: look up phenotype targets by EID ──────────
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
                    metadata.append({'uid': pid, 'patient_id': pid, 'labels': labels})

    if not_found:
        if aladin_metadata is not None:
            print(f"  {not_found} UIDs not found in ALADIN metadata — labels set to {{}}.")
        else:
            print(f"  {not_found} patient IDs not found in phenotype targets — skipped.")

    latents = {'z0': np.stack(z0_list, axis=0)}
    if has_m and m_list:
        latents['m'] = np.stack(m_list, axis=0)

    model.return_latent = False
    return latents, metadata


def _probe_bar_chart(title, x_labels, bar_groups, ylabel, ylim=(0, 1)):
    """Return a matplotlib Figure with a grouped bar chart for probe results.

    Args:
        title      : figure title string
        x_labels   : list of x-axis tick labels
        bar_groups : list of (group_label, values_list) — one bar per group per x position
        ylabel     : y-axis label
        ylim       : (min, max) for y axis
    """
    n_x      = len(x_labels)
    n_groups = len(bar_groups)
    width    = 0.8 / max(n_groups, 1)
    offsets  = np.linspace(-(n_groups - 1) / 2 * width,
                            (n_groups - 1) / 2 * width, n_groups)
    x = np.arange(n_x)

    fig, ax = plt.subplots(figsize=(max(6, n_x * 0.7 + 2), 4))
    for (label, vals), offset in zip(bar_groups, offsets):
        ax.bar(x + offset, vals, width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, fontweight='bold')
    if n_groups > 1:
        ax.legend(fontsize=8, framealpha=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def log_probe_metrics(probe_results, latent_key, seg_type, run):
    """Log linear probe results to wandb under the 'linear probe' panel.

    Logs:
    - Dataset stats: n_train / n_test / class-freq or continuous distribution
                     plots per parameter (logged once, independent of latent_key)
    - Regression   : bar chart of R² per parameter (one bar per probe method)
    - Classification: bar chart of per-class accuracy + bar chart of overall
                      accuracy / F1 / ROC-AUC per probe method
    """
    import wandb as _wandb
    panel    = "linear probe"
    suffix   = f"{seg_type}/{latent_key}" if seg_type else latent_key
    log_dict = {}

    # ── Dataset statistics (sample counts + distribution per param) ───────────
    dstats_all = probe_results.get('dataset_stats', {})
    for param, dstats in dstats_all.items():
        kind   = dstats['type']
        n_tr   = dstats['n_train']
        n_te   = dstats['n_test']
        # Scalar metrics logged as wandb summary values
        log_dict[f"{panel}/dataset/{param}/n_train"] = n_tr
        log_dict[f"{panel}/dataset/{param}/n_test"]  = n_te
        log_dict[f"{panel}/dataset/{param}/n_total"] = n_tr + n_te

        if kind == 'classification':
            log_dict[f"{panel}/dataset/{param}/n_classes"] = dstats['n_classes']
            for cls in dstats['classes']:
                log_dict[f"{panel}/dataset/{param}/train_freq/{cls}"] = \
                    dstats['train_class_freq'][cls]
                log_dict[f"{panel}/dataset/{param}/test_freq/{cls}"] = \
                    dstats['test_class_freq'][cls]
        else:
            for split_name in ('train', 'test'):
                d = dstats[split_name]
                for metric in ('mean', 'std', 'min', 'max', 'median',
                               'iqr', 'skewness', 'kurtosis', 'pct_zero'):
                    log_dict[f"{panel}/dataset/{param}/{split_name}/{metric}"] = d[metric]

        # Distribution image (already saved to disk; re-render for wandb)
        fig = _make_dataset_stats_fig(param, dstats)
        log_dict[f"{panel}/dataset/{param}/distribution"] = _wandb.Image(fig)
        plt.close(fig)

    # ── Regression: R² per parameter ──────────────────────────────────────────
    reg = probe_results.get('regression', {})
    if reg:
        params  = sorted(reg.keys())
        methods = sorted(next(iter(reg.values())).keys())
        groups  = [
            (method, [reg[p][method]['metrics'].get('r2', float('nan')) for p in params])
            for method in methods
        ]
        fig = _probe_bar_chart(
            title=f"R² per parameter — {suffix}",
            x_labels=params,
            bar_groups=groups,
            ylabel="R²",
            ylim=(min(0, min(v for _, vs in groups for v in vs if not np.isnan(v)) - 0.05), 1.0),
        )
        log_dict[f"{panel}/regression/{suffix}"] = _wandb.Image(fig)
        plt.close(fig)

    # ── Classification: per-class accuracy + overall metrics ──────────────────
    clf = probe_results.get('classification', {})
    if clf:
        for param, model_results in clf.items():
            methods = sorted(model_results.keys())

            # Per-class accuracy bar chart
            sample_res = next(iter(model_results.values()))
            le_classes = list(range(max(sample_res['y_true']) + 1))
            # Use string class labels if available from per_class_accuracy computation
            pca = {m: _per_class_accuracy(model_results[m]['y_true'],
                                           model_results[m]['y_pred'],
                                           le_classes)
                   for m in methods}
            # Try to get string class names from the label encoder stored in results
            class_labels = [str(c) for c in le_classes]
            pca_groups = [
                (m, [pca[m].get(c, float('nan')) for c in le_classes])
                for m in methods
            ]
            fig = _probe_bar_chart(
                title=f"Per-class accuracy — {param} ({suffix})",
                x_labels=class_labels,
                bar_groups=pca_groups,
                ylabel="Accuracy",
                ylim=(0, 1.08),
            )
            log_dict[f"{panel}/classification/per_class_accuracy/{param}/{suffix}"] = _wandb.Image(fig)
            plt.close(fig)

            # Overall metrics bar chart (accuracy, f1, roc_auc)
            metric_keys = ['accuracy', 'f1', 'roc_auc']
            metric_groups = [
                (m, [model_results[m]['metrics'].get(k, float('nan')) for k in metric_keys])
                for m in methods
            ]
            fig = _probe_bar_chart(
                title=f"Overall metrics — {param} ({suffix})",
                x_labels=metric_keys,
                bar_groups=metric_groups,
                ylabel="Score",
                ylim=(0, 1.08),
            )
            log_dict[f"{panel}/classification/overall_metrics/{param}/{suffix}"] = _wandb.Image(fig)
            plt.close(fig)

    if log_dict:
        run.log(log_dict)
        print(f"  Logged {len(log_dict)} probe charts to wandb under '{panel}' panel.")


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

    # Load ALADIN metadata files when --aladin_metadata_dir is given.
    def _load_aladin_meta(split: str) -> dict | None:
        d = getattr(args, 'aladin_metadata_dir', None)
        if d is None:
            return None
        path = os.path.join(d, f'{split}_metadata.json')
        if not os.path.exists(path):
            print(f"  Warning: ALADIN metadata not found at {path} — falling back to default label source.")
            return None
        with open(path) as f:
            return json.load(f)

    with np.errstate(all='ignore'):
        print("Collecting train latents...")
        tr_latents, tr_metadata = collect_latents(
            trainset, model, task_params, args, device,
            aladin_metadata=_load_aladin_meta('train'))
        print("Collecting test latents...")
        te_latents, te_metadata = collect_latents(
            testset,  model, task_params, args, device,
            aladin_metadata=_load_aladin_meta('test'))

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
    parser.add_argument('--aladin_metadata_dir', type=str, default=None,
                        help='Directory containing {split}_metadata.json files produced by '
                             'aladin_preprocess.py (train_metadata.json, valid_metadata.json, '
                             'test_metadata.json).  When provided, labels are taken from these '
                             'files instead of from the metadata embedded in the latents JSON.')
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

    # Optionally load ALADIN metadata for label override.
    def _load_aladin_meta_for_split(split: str) -> dict | None:
        if args.aladin_metadata_dir is None:
            return None
        path = os.path.join(args.aladin_metadata_dir, f'{split}_metadata.json')
        if not os.path.exists(path):
            print(f"  Warning: ALADIN metadata not found at {path} — using embedded labels.")
            return None
        print(f"  Loading ALADIN metadata from {path}")
        with open(path) as f:
            return json.load(f)

    latents_dir = os.path.join(root_dir, 'latents')
    print("Loading latents...")
    latents_dict = {}
    for split in args.splits:
        aladin_meta = _load_aladin_meta_for_split(split)
        latents, metadata = _load_split(latents_dir, split, aladin_metadata=aladin_meta)
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

