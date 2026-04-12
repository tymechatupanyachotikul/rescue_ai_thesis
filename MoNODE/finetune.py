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
    kruskal as sp_kruskal, chi2_contingency as sp_chi2,
)
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV,
    LogisticRegression, LogisticRegressionCV,
)
from sklearn.metrics import (
    r2_score, mean_squared_error,
    roc_auc_score, accuracy_score, f1_score, recall_score,
    balanced_accuracy_score,
    adjusted_rand_score,
    silhouette_score, silhouette_samples,
    ConfusionMatrixDisplay,
)
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from umap import UMAP as _UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False



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
        os.path.join(latents_dir, f'latent_tensors_{split}.npz'),
    ]
    json_candidates = [
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
            uid = entry.get('uid')
            aladin_entry = aladin_metadata.get(uid)
            if aladin_entry is not None:
                entry['labels'] = aladin_entry.get('labels', {})
            else:
                print(entry)
                not_found += 1
        if not_found:
            print(f"  [{split}] {not_found} UIDs not found in ALADIN metadata — labels left as-is.")

    n_latents = latents['z0'].shape[0]
    if len(metadata) != n_latents:
        print(f"  [{split}] WARNING: metadata has {len(metadata)} entries but "
              f"latents have {n_latents} rows — truncating metadata to match.")
        metadata = metadata[:n_latents]
    print(f"  [{split}] loaded {n_latents} samples "
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


def _classification_models(binary: bool = False, imbalanced: bool = False):
    """Return (name, model) pairs for the four classification probes.

    class_weight='balanced' is applied to logistic regression for:
      - all multi-class probes, or
      - binary probes where the majority class exceeds 80 % of the training set.
    MLP has no class_weight parameter and is left unchanged.
    """
    cw = None if (binary and not imbalanced) else 'balanced'
    return [
        ('ols',   LogisticRegression(penalty=None, max_iter=1000, n_jobs=-1,
                                     class_weight=cw)),
        ('ridge', LogisticRegressionCV(penalty='l2', cv=5, max_iter=1000, n_jobs=-1,
                                       class_weight=cw)),
        ('lasso', LogisticRegressionCV(penalty='l1', solver='saga', cv=5,
                                       max_iter=1000, n_jobs=-1, class_weight=cw)),
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
            'r2': float(r2_score(y_te_arr, y_pred)),
        },
    }


def _eval_classification(model, X_tr, y_tr, X_te, y_te, le, imbalanced: bool = False):
    """Evaluate a classification probe.

    Binary balanced    → accuracy only.
    Binary imbalanced  → F1 (positive class), AUROC, balanced accuracy.
    Multi-class        → macro F1, macro AUROC, macro recall.
    """
    y_tr_enc = le.transform(y_tr)
    y_te_enc = le.transform(y_te)
    binary = len(le.classes_) == 2
    with np.errstate(under='ignore', divide='ignore'):
        model.fit(X_tr, y_tr_enc)
    y_pred = model.predict(X_te)

    y_prob = None
    if hasattr(model, 'predict_proba'):
        with np.errstate(under='ignore'):
            y_prob = model.predict_proba(X_te)

    if binary and imbalanced:
        metrics = {
            'f1_binary':          float(f1_score(y_te_enc, y_pred, pos_label=1,
                                                  average='binary', zero_division=0)),
            'balanced_accuracy':  float(balanced_accuracy_score(y_te_enc, y_pred)),
        }
        if y_prob is not None:
            try:
                metrics['auroc'] = float(roc_auc_score(y_te_enc, y_prob[:, 1]))
            except ValueError:
                pass
    elif binary:
        metrics = {'accuracy': float(accuracy_score(y_te_enc, y_pred))}
    else:
        metrics = {
            'f1_macro':     float(f1_score(y_te_enc, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_te_enc, y_pred, average='macro', zero_division=0)),
        }
        if y_prob is not None:
            try:
                metrics['auroc_macro'] = float(
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
        r2 = res['metrics']['r2']
        ax.set_title(f"{name}\nR²={r2:.3f}", fontsize=9)
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
    """Confusion matrix per model + combined metrics/per-class-accuracy JSON."""
    names   = list(model_results.keys())
    classes = [str(c) for c in le.classes_]
    n_cls   = len(classes)

    cell_size = max(0.7, 5.0 / n_cls)
    tick_fs   = max(5, 9 - n_cls // 3)

    for name in names:
        res = model_results[name]
        fig_cm, ax = plt.subplots(figsize=(cell_size * n_cls + 1.5,
                                           cell_size * n_cls + 1.5))
        ConfusionMatrixDisplay.from_predictions(
            res['y_true'], res['y_pred'],
            display_labels=classes,
            ax=ax, colorbar=True, xticks_rotation=45,
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fs, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fs)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        m = res['metrics']
        if 'accuracy' in m:
            metric_str = f"acc={m['accuracy']:.3f}"
        elif 'f1_binary' in m:
            metric_str = (f"f1={m.get('f1_binary', float('nan')):.3f}  "
                          f"auroc={m.get('auroc', float('nan')):.3f}  "
                          f"bal_acc={m.get('balanced_accuracy', float('nan')):.3f}")
        else:
            metric_str = (f"f1={m.get('f1_macro', float('nan')):.3f}  "
                          f"recall={m.get('recall_macro', float('nan')):.3f}  "
                          f"auroc={m.get('auroc_macro', float('nan')):.3f}")
        ax.set_title(f"{param} — {name}\n{metric_str}", fontsize=10, fontweight='bold')
        fig_cm.tight_layout()
        fig_cm.savefig(os.path.join(out_dir, f'confusion_matrix_{name}.png'), dpi=130)
        plt.close(fig_cm)

    # Combined metrics + per-class accuracy in one JSON
    pca_data = {name: _per_class_accuracy(model_results[name]['y_true'],
                                           model_results[name]['y_pred'],
                                           classes)
                for name in names}
    combined = {
        name: {
            'metrics':           model_results[name]['metrics'],
            'per_class_accuracy': pca_data[name],
        }
        for name in names
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(combined, f, indent=2)


def _plot_summary_regression(all_results, out_dir):
    """Grouped bar chart: R² per param × model."""
    params = sorted(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())
    x = np.arange(len(params))
    width = 0.8 / max(len(model_names), 1)
    offsets = np.linspace(-(len(model_names) - 1) / 2 * width,
                           (len(model_names) - 1) / 2 * width, len(model_names))

    fig, ax = plt.subplots(figsize=(max(8, len(params) * 1.2 + 2), 5))
    for j, (mname, offset) in enumerate(zip(model_names, offsets)):
        vals = [all_results[p][mname]['metrics'].get('r2', float('nan')) for p in params]
        ax.bar(x + offset, vals, width, label=mname, color=f'C{j}', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('R²', fontsize=10)
    ax.set_title('Regression summary — R²', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'regression_summary.png'), dpi=120)
    plt.close(fig)


def _plot_summary_classification(all_results, out_dir):
    """3-panel grouped bar chart combining binary-imbalanced and multi-class metrics.

    Each panel picks the first available key per param × model:
      F1               : f1_binary  (imbalanced binary)  | f1_macro  (multi-class)
      AUROC            : auroc      (imbalanced binary)  | auroc_macro (multi-class)
      Bal-acc / Recall : balanced_accuracy (imbalanced binary) | recall_macro (multi-class)

    Balanced binary params (accuracy only) produce NaN bars — they are not
    the focus of this chart.
    """
    params = sorted(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())
    x = np.arange(len(params))
    width = 0.8 / max(len(model_names), 1)
    offsets = np.linspace(-(len(model_names) - 1) / 2 * width,
                           (len(model_names) - 1) / 2 * width, len(model_names))

    # (title, [keys to try in order])
    panels = [
        ('F1 (binary pos. / macro)',            ['f1_binary',         'f1_macro']),
        ('AUROC (binary / macro)',               ['auroc',             'auroc_macro']),
        ('Balanced-acc / Macro recall',          ['balanced_accuracy',  'recall_macro']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(params) * 1.8 + 3), 5))
    for ax, (title, keys) in zip(axes, panels):
        for j, (mname, offset) in enumerate(zip(model_names, offsets)):
            vals = []
            for p in params:
                m = all_results[p][mname]['metrics']
                v = next((m[k] for k in keys if k in m), float('nan'))
                vals.append(v)
            ax.bar(x + offset, vals, width, label=mname, color=f'C{j}', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, framealpha=0.8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'classification_summary.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation-set resampling for class-balanced 'class' probe (MedalCare-XL)
# ---------------------------------------------------------------------------

def _resample_sinus_balanced(X_te: np.ndarray, y_te: list,
                              seed: int = 42) -> tuple[np.ndarray, list]:
    """Resample the evaluation set so that 'sinus' count equals the max count
    of any other class.

    Strategy (deterministic via fixed seed):
      1. Keep all non-sinus samples unchanged.
      2. Determine target_n = max count among non-sinus classes.
      3. Take as many original sinus samples as possible (up to target_n),
         selected by sorted index for reproducibility.
      4. If original sinus count < target_n, fill the remainder by sampling
         (with replacement if necessary) from the remaining non-sinus pool,
         using a seeded RNG so results are identical across runs.

    Returns resampled (X_te, y_te) with sinus count == target_n.
    """
    from collections import Counter as _Counter
    import numpy as np

    y_arr     = np.array(y_te)
    classes   = [c for c in _Counter(y_te) if c != 'sinus']
    if not classes:
        return X_te, y_te          # no non-sinus class → nothing to do

    counts    = _Counter(y_te)
    target_n  = max(counts[c] for c in classes)
    n_sinus   = counts.get('sinus', 0)

    if n_sinus == target_n:
        return X_te, y_te          # already balanced

    sinus_idx     = np.where(y_arr == 'sinus')[0]
    non_sinus_idx = np.where(y_arr != 'sinus')[0]

    # Sort for reproducibility before any RNG is involved
    sinus_idx     = np.sort(sinus_idx)
    non_sinus_idx = np.sort(non_sinus_idx)

    rng = np.random.default_rng(seed)

    if n_sinus >= target_n:
        # Downsample: keep first target_n sinus samples (sorted index → deterministic)
        chosen_sinus = sinus_idx[:target_n]
    else:
        # Use all original sinus samples, then fill from non-sinus pool
        n_extra  = target_n - n_sinus
        extra    = rng.choice(non_sinus_idx, size=n_extra,
                              replace=(n_extra > len(non_sinus_idx)))
        extra    = np.sort(extra)
        chosen_sinus = np.concatenate([sinus_idx, extra])

    all_idx = np.concatenate([non_sinus_idx, chosen_sinus])
    all_idx = np.sort(all_idx)   # keep original relative order

    X_new = X_te[all_idx]
    y_new = [y_te[i] for i in all_idx]
    return X_new, y_new


# ---------------------------------------------------------------------------
# Main probing entry-point
# ---------------------------------------------------------------------------

def run_linear_probes(train_latents, train_metadata, test_latents, test_metadata,
                      latent_key='z0', out_root=None, methods=None,
                      skip_params=None, balance_sinus: bool = False):
    """Train four probes per phenotype and evaluate on the test set.

    Models
    ------
    Regression    : OLS, RidgeCV (L2), LassoCV (L1), MLP-1-hidden-layer
    Classification: Logistic (no penalty), LogisticRegressionCV L2, L1, MLP

    Metrics
    -------
    Regression         : MSE, R²
    Binary classif.    : Accuracy
    Multi-class classif: Macro F1, Macro Recall, Macro AUROC

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
        if skip_params and param in skip_params:
            print(f"  [{param}] skipped (in skip_params)")
            continue

        tr_idx = tr_indices[param]
        te_idx = te_indices[param]

        if len(tr_idx) < 10 or len(te_idx) < 2:
            print(f"  [{param}] skipped — too few samples "
                  f"(train={len(tr_idx)}, test={len(te_idx)})")
            continue

        y_tr = tr_labels_all[param]
        y_te = te_labels_all[param]
        is_categorical = isinstance(y_tr[0], str)
        # Any param with exactly 2 distinct values → binary classification
        is_binary = (not is_categorical) and (len(set(y_tr)) == 2)

        # Imbalance check: majority class > 80 % of the training set
        is_imbalanced = False
        if is_binary or is_categorical:
            from collections import Counter as _Counter
            tr_counts = _Counter(y_tr)
            majority_frac = max(tr_counts.values()) / max(len(y_tr), 1)
            is_imbalanced = majority_frac > 0.80

        # ── Dataset statistics (print only, no file) ──────────────────────────
        dstats = _compute_dataset_stats(y_tr, y_te, is_categorical, is_binary)
        all_dataset_stats[param] = dstats
        _print_dataset_stats(param, dstats)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_full[tr_idx])
        X_te = scaler.transform(X_te_full[te_idx])

        # Resample eval set for 'class' probe so sinus == max other-class count
        if balance_sinus and param == 'class' and is_categorical:
            X_te, y_te = _resample_sinus_balanced(X_te, y_te)

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
            for name, mdl in _classification_models(binary=is_binary, imbalanced=is_imbalanced):
                if methods and name not in methods:
                    continue
                param_results[name] = _eval_classification(
                    mdl, X_tr, y_tr, X_te, y_te, le, imbalanced=is_imbalanced)
                m = param_results[name]['metrics']
                if 'accuracy' in m:
                    print(f"  [{param}][{name}]  acc={m['accuracy']:.3f}")
                elif 'f1_binary' in m:
                    print(f"  [{param}][{name}]  "
                          f"f1={m.get('f1_binary', float('nan')):.3f}  "
                          f"auroc={m.get('auroc', float('nan')):.3f}  "
                          f"bal_acc={m.get('balanced_accuracy', float('nan')):.3f}  "
                          f"[imbalanced={majority_frac:.0%}]")
                else:
                    print(f"  [{param}][{name}]  "
                          f"f1={m.get('f1_macro', float('nan')):.3f}  "
                          f"recall={m.get('recall_macro', float('nan')):.3f}  "
                          f"auroc={m.get('auroc_macro', float('nan')):.3f}")

            clf_results[param] = param_results

            if out_root:
                pdir = os.path.join(out_root, 'classification', param)
                os.makedirs(pdir, exist_ok=True)
                _plot_classification_param(param, param_results, pdir, le)
        else:  # continuous regression
            param_results = {}
            for name, mdl in _regression_models():
                if methods and name not in methods:
                    continue
                param_results[name] = _eval_regression(mdl, X_tr, y_tr, X_te, y_te)
                m = param_results[name]['metrics']
                print(f"  [{param}][{name}]  R²={m['r2']:.3f}")

            reg_results[param] = param_results

            if out_root:
                pdir = os.path.join(out_root, 'regression', param)
                os.makedirs(pdir, exist_ok=True)
                _plot_regression_param(param, param_results, pdir)
                _save_metrics_json(param_results, pdir)

    # ── Summary plots ─────────────────────────────────────────────────────────
    if out_root:
        if reg_results:
            sdir = os.path.join(out_root, 'regression', '_summary')
            os.makedirs(sdir, exist_ok=True)
            _plot_summary_regression(reg_results, sdir)
        if clf_results:
            sdir = os.path.join(out_root, 'classification', '_summary')
            os.makedirs(sdir, exist_ok=True)
            _plot_summary_classification(clf_results, sdir)

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
    class_label_key: str = 'class',
    tag: str | None = None,
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
      latent_umap.png           — UMAP 2-D: predicted | ground-truth panels
      latent_tsne.png           — t-SNE 2-D: predicted | ground-truth panels
      silhouette.png            — per-cluster silhouette bar chart
      param_epsilon_squared.png — effect-size bar chart across parameters
      cluster_{k}/
        param_distributions.png — violin plots of continuous params in this cluster
    """
    out_dir = None
    if out_root:
        subdir = f'{latent_key}_{tag}' if tag else latent_key
        out_dir = os.path.join(out_root, 'clustering', subdir)
        os.makedirs(out_dir, exist_ok=True)

    is_medalcare = dataset_name.lower() == 'medalcare-xl'

    # ── 1. Extract latent arrays ──────────────────────────────────────────────
    X_tr = train_latents[latent_key].astype(np.float64)
    X_te = test_latents[latent_key].astype(np.float64)

    # ── 2. Standardise ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr        = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)   # kept for UMAP/t-SNE embedding (full dim)

    # ── 3. PCA (fit on train) ─────────────────────────────────────────────────
    actual_pca_dim = min(pca_dim, X_tr.shape[1], X_tr.shape[0])
    pca = PCA(n_components=actual_pca_dim, random_state=42)
    Z_tr = pca.fit_transform(X_tr)
    Z_te = pca.transform(X_te_scaled)
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
        raw_true = [m.get('labels', {}).get(class_label_key) for m in test_metadata]
        if any(v is not None for v in raw_true):
            le_cls = LabelEncoder()
            valid_mask = np.array([v is not None and str(v) not in ('', 'None')
                                   for v in raw_true])
            true_enc  = le_cls.fit_transform([str(v) for v, ok in zip(raw_true, valid_mask) if ok])
            pred_valid = labels_te[valid_mask]
            ari = float(adjusted_rand_score(true_enc, pred_valid))
            true_classes_te = np.array([str(v) if ok else 'unknown'
                                         for v, ok in zip(raw_true, valid_mask)])
            print(f"  [GMM/{latent_key}] ARI vs '{class_label_key}': {ari:.4f}")

    # ── 8. Per-cluster label statistics + Kruskal-Wallis ε² + Chi-square ────────
    _SKIP_LABEL_KEYS = {'class', 'patient_id'}

    # ── Continuous parameters ──────────────────────────────────────────────────
    all_params: set[str] = set()
    for m in test_metadata:
        for k, v in m.get('labels', {}).items():
            if k in _SKIP_LABEL_KEYS:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                all_params.add(k)

    param_values: dict[str, np.ndarray] = {}
    for param in all_params:
        arr = np.array([
            float(m.get('labels', {}).get(param, float('nan')))
            if isinstance(m.get('labels', {}).get(param), (int, float))
            else float('nan')
            for m in test_metadata
        ])
        # Only keep params with >2 distinct non-NaN values (binary params go to chi-square)
        distinct_vals = set(v for v in arr if not np.isnan(v))
        if len(distinct_vals) > 2:
            param_values[param] = arr

    # ── Categorical parameters ─────────────────────────────────────────────────
    # Mirror the same logic as run_linear_probes:
    #   - str/bool values → categorical
    #   - numeric with exactly 2 distinct values → binary (treated as categorical)
    _raw_label_vals: dict[str, list] = {}
    for m in test_metadata:
        for k, v in m.get('labels', {}).items():
            if k in _SKIP_LABEL_KEYS:
                continue
            _raw_label_vals.setdefault(k, []).append(v)

    cat_params: set[str] = set()
    for k, vals in _raw_label_vals.items():
        non_null = [v for v in vals
                    if v is not None
                    and not (isinstance(v, float) and np.isnan(v))
                    and str(v).lower() not in ('nan', 'none', '')]
        if not non_null:
            continue
        if isinstance(non_null[0], bool) or isinstance(non_null[0], str):
            cat_params.add(k)
        else:
            # numeric: binary if exactly 2 distinct values (same as is_binary in probes)
            if len(set(non_null)) == 2:
                cat_params.add(k)

    # param → list[str | None] aligned with test samples
    cat_param_values: dict[str, list] = {}
    for param in cat_params:
        vals: list = []
        for m in test_metadata:
            v = m.get('labels', {}).get(param)
            if v is None or (isinstance(v, float) and np.isnan(v)) \
                    or str(v).lower() in ('nan', 'none', ''):
                vals.append(None)
            else:
                vals.append(str(v))
        cat_param_values[param] = vals

    # ── Per-cluster statistics (continuous + categorical) ──────────────────────
    cluster_stats: dict[int, dict] = {}
    for k in range(n_clusters):
        mask = labels_te == k
        cstats: dict = {'n': int(mask.sum()), 'continuous': {}, 'categorical': {}}

        for param, arr in param_values.items():
            vals = arr[mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                cstats['continuous'][param] = {'mean': None, 'std': None, 'n_valid': 0}
            else:
                cstats['continuous'][param] = {
                    'mean':    round(float(valid.mean()), 6),
                    'std':     round(float(valid.std()),  6),
                    'median':  round(float(np.median(valid)), 6),
                    'n_valid': int(len(valid)),
                }

        for param, vals_list in cat_param_values.items():
            cluster_vals = [v for v, in_k in zip(vals_list, mask) if in_k and v is not None]
            from collections import Counter as _Counter
            counts = dict(_Counter(cluster_vals))
            total  = sum(counts.values())
            freqs  = {cat: round(cnt / total, 4) for cat, cnt in counts.items()} if total else {}
            cstats['categorical'][param] = {'counts': counts, 'freq': freqs, 'n_valid': total}

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

    # ── 9a. Chi-square + Cramér's V for categorical parameters ───────────────
    # Contingency table: rows = clusters, cols = unique categories.
    # Cramér's V = sqrt(χ² / (n · min(r-1, c-1))) — analogous to ε² for KW.
    chi2_results: dict[str, dict] = {}
    for param, vals_list in cat_param_values.items():
        categories = sorted({v for v in vals_list if v is not None})
        if len(categories) < 2:
            chi2_results[param] = {'chi2': None, 'dof': None, 'p_value': None,
                                   'cramers_v': None}
            continue
        print(f'Performing CHI SQUARE for {param}')
        cat_idx = {c: i for i, c in enumerate(categories)}
        table = np.zeros((n_clusters, len(categories)), dtype=int)
        for v, lbl in zip(vals_list, labels_te):
            if v is not None:
                table[lbl, cat_idx[v]] += 1

        # Remove all-zero rows/cols before running the test
        row_mask = table.sum(axis=1) > 0
        col_mask = table.sum(axis=0) > 0
        table_trimmed = table[np.ix_(row_mask, col_mask)]

        if table_trimmed.shape[0] < 2 or table_trimmed.shape[1] < 2:
            chi2_results[param] = {'chi2': None, 'dof': None, 'p_value': None,
                                   'cramers_v': None}
            print(f'SKipping chi square')
            continue

        try:
            chi2_stat, pval, dof, _ = sp_chi2(table_trimmed)
            n       = int(table_trimmed.sum())
            min_dim = min(table_trimmed.shape[0] - 1, table_trimmed.shape[1] - 1)
            cramers_v = float(np.sqrt(chi2_stat / (n * min_dim))) \
                if min_dim > 0 and n > 0 else float('nan')
            chi2_results[param] = {
                'chi2':      round(float(chi2_stat), 6),
                'dof':       int(dof),
                'p_value':   round(float(pval),      8),
                'cramers_v': round(cramers_v,         6),
            }
        except Exception as e:
            print(f'Error {e}')
            chi2_results[param] = {'chi2': None, 'dof': None, 'p_value': None,
                                   'cramers_v': None, 'error': str(e)}

    # ── 9b. OLS regression of each continuous param on cluster one-hot ────────
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
        if args.plot_latent_reduct:
            _plot_latent_embed(X_te_scaled, labels_te, true_classes_te, n_clusters, out_dir)
        _plot_gmm_silhouette(sil_sample, labels_te, n_clusters, sil_global, out_dir)
        _plot_gmm_epsilon_squared(eps_sq_results, out_dir)
        _plot_gmm_cramers_v(chi2_results, out_dir)
        _plot_gmm_cluster_violins(param_values, labels_te, n_clusters, out_dir)

    # ── 11. Wandb logging ─────────────────────────────────────────────────────
    if run is not None:
        _log_gmm_to_wandb(run, latent_key, sil_global, sil_per_cluster, ari,
                          eps_sq_results, chi2_results, out_dir)

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
        'chi_square':          chi2_results,
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

def _scatter_panel(ax, E2d, labels, unique_labels, cmap_fn, title, xlabel, ylabel,
                   is_cluster: bool = True) -> None:
    """Draw one scatter panel onto *ax*. Shared by UMAP and t-SNE figures."""
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        label_str = f'C{lbl} (n={mask.sum()})' if is_cluster else str(lbl)
        ax.scatter(E2d[mask, 0], E2d[mask, 1], s=5, alpha=0.5,
                   color=cmap_fn(i), label=label_str, rasterized=True)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ncol = max(1, len(unique_labels) // 10)
    ax.legend(fontsize=5, markerscale=2, ncol=ncol, framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)


def _plot_latent_embed(X: np.ndarray, labels_pred: np.ndarray,
                       true_classes, n_clusters: int, out_dir: str) -> None:
    """UMAP and t-SNE 2-D embedding of the scaled latent space.

    Produces two figures, each with side-by-side predicted / ground-truth panels:
      latent_umap.png
      latent_tsne.png

    *true_classes* may be None (skips the ground-truth panel).
    """
    cmap_pred = mpl_cm.get_cmap('tab20', n_clusters)
    unique_pred = sorted(set(labels_pred.tolist()))

    has_true = true_classes is not None
    unique_true = sorted(set(true_classes.tolist())) if has_true else []
    cmap_true = mpl_cm.get_cmap('tab20', max(len(unique_true), 1))

    n_panels = 2 if has_true else 1

    for method_name, E2d in _embed_2d(X):
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)

        _scatter_panel(
            axes[0, 0], E2d, labels_pred, unique_pred,
            cmap_pred,
            title=f'GMM predicted — {method_name}',
            xlabel=f'{method_name} 1', ylabel=f'{method_name} 2',
            is_cluster=True,
        )
        if has_true:
            _scatter_panel(
                axes[0, 1], E2d, true_classes, unique_true,
                cmap_true,
                title=f'Ground truth — {method_name}',
                xlabel=f'{method_name} 1', ylabel=f'{method_name} 2',
                is_cluster=False,
            )

        fig.tight_layout()
        fname = f'latent_{method_name.lower()}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)
        print(f"  Saved {fname}")


def _embed_2d(X: np.ndarray):
    """Yield (method_name, 2-D embedding) for UMAP (if available) and t-SNE."""
    if HAS_UMAP:
        try:
            emb = _UMAP(n_components=2, random_state=42,
                        n_neighbors=min(15, len(X) - 1)).fit_transform(X)
            yield 'UMAP', emb
        except Exception as e:
            print(f"  UMAP failed: {e}")

    perp = min(30, max(5, len(X) // 100))
    try:
        # max_iter replaced n_iter in scikit-learn 1.4; try both
        try:
            emb = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                       random_state=42).fit_transform(X)
        except TypeError:
            emb = TSNE(n_components=2, perplexity=perp, n_iter=1000,  # type: ignore[call-arg]
                       random_state=42).fit_transform(X)
        yield 'tSNE', emb
    except Exception as e:
        print(f"  t-SNE failed: {e}")


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


def _plot_gmm_cramers_v(chi2_results: dict, out_dir: str) -> None:
    """Bar chart of Cramér's V effect sizes for categorical parameters."""
    params = [p for p, v in chi2_results.items()
              if v.get('cramers_v') is not None and not math.isnan(v['cramers_v'])]
    if not params:
        return

    params     = sorted(params, key=lambda p: chi2_results[p]['cramers_v'], reverse=True)
    cramers_vals = [chi2_results[p]['cramers_v'] for p in params]
    pvals        = [chi2_results[p]['p_value']   for p in params]

    fig, ax = plt.subplots(figsize=(max(6, len(params) * 0.55 + 2), 4))
    colours = ['#d62728' if p is not None and p < 0.05 else '#aec7e8' for p in pvals]
    bars = ax.bar(range(len(params)), cramers_vals, color=colours, edgecolor='white')

    for bar, p in zip(bars, pvals):
        if p is None:
            continue
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
    ax.set_ylabel("Cramér's V (χ² effect size)", fontsize=9)
    ax.set_title("Cramér's V per categorical parameter across GMM clusters",
                 fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=[mpl_patches.Patch(color='#d62728', label='p < 0.05'),
                       mpl_patches.Patch(color='#aec7e8', label='p ≥ 0.05')],
              fontsize=7, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'param_cramers_v.png'), dpi=130)
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

    chi2 = results.get('chi_square', {})
    valid_chi2 = {p: v for p, v in chi2.items()
                  if v.get('cramers_v') is not None and not math.isnan(v['cramers_v'])}
    if valid_chi2:
        print(f"\n  Top-5 parameters by Cramér's V (chi-square):")
        for p, v in sorted(valid_chi2.items(),
                           key=lambda x: x[1]['cramers_v'], reverse=True)[:5]:
            pval = v['p_value']
            sig  = '*' if pval is not None and pval < 0.05 else ''
            pstr = f"{pval:.4g}" if pval is not None else 'N/A'
            print(f"    {p:30s}  V={v['cramers_v']:.4f}  "
                  f"χ²={v['chi2']:.2f}  p={pstr}{sig}")
    print(f"{'─'*60}\n")


def _log_gmm_to_wandb(run, latent_key: str, sil_global: float,
                       sil_per_cluster: dict, ari: float | None,
                       eps_sq_results: dict, chi2_results: dict,
                       out_dir: str | None) -> None:
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

    for param, v in chi2_results.items():
        cv = v.get('cramers_v')
        if cv is not None and not math.isnan(cv):
            log_dict[f'{panel}/{latent_key}/cramers_v/{param}']   = cv
            pval = v.get('p_value')
            if pval is not None:
                log_dict[f'{panel}/{latent_key}/chi2_pvalue/{param}'] = pval

    if out_dir:
        for fname, key_suffix in [
            ('latent_umap.png',           'embed_umap'),
            ('latent_tsne.png',           'embed_tsne'),
            ('silhouette.png',            'silhouette'),
            ('param_epsilon_squared.png', 'epsilon_squared'),
            ('param_cramers_v.png',       'cramers_v'),
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

    z0_list, z0_sample_list, m_list, ztL_list, metadata = [], [], [], [], []
    has_m = True   # set False if model returns m=None
    not_found = 0

    with torch.no_grad():
        for batch, batch_y, mask in tqdm(dataloader, desc="Collecting latents"):
            batch = batch.to(device)
            mask  = mask.to(device)

            # returns (z0_mean, z0_sample, m, ztL)
            z0, z0_samp, m, ztL = model(batch, 1, mask=mask)
            if m is None:
                has_m = False
            # ztL: [L, N, T, q] — mean over MC samples → [N, T, q]
            ztL = ztL.mean(0).detach().cpu()

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
                    z0_sample_list.append(z0_samp[i].detach().cpu().numpy())
                    ztL_list.append(ztL[i].numpy())
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
                    z0_sample_list.append(z0_samp[i].detach().cpu().numpy())
                    ztL_list.append(ztL[i].numpy())
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
                    z0_sample_list.append(z0_samp[i].detach().cpu().numpy())
                    ztL_list.append(ztL[i].numpy())
                    if has_m:
                        m_list.append(m[i].detach().cpu().numpy())
                    metadata.append({'uid': pid, 'patient_id': pid, 'labels': labels})

    if not_found:
        if aladin_metadata is not None:
            print(f"  {not_found} UIDs not found in ALADIN metadata — labels set to {{}}.")
        else:
            print(f"  {not_found} patient IDs not found in phenotype targets — skipped.")

    latents = {'z0': np.stack(z0_list, axis=0)}
    if z0_sample_list:
        latents['z0_sample'] = np.stack(z0_sample_list, axis=0)
    if has_m and m_list:
        latents['m'] = np.stack(m_list, axis=0)
    if ztL_list:
        # Variable T across batches — pad to global max T with zeros
        max_T = max(arr.shape[0] for arr in ztL_list)
        q_dim = ztL_list[0].shape[-1]
        ztL_padded = np.zeros((len(ztL_list), max_T, q_dim), dtype=ztL_list[0].dtype)
        for idx, arr in enumerate(ztL_list):
            ztL_padded[idx, :arr.shape[0], :] = arr
        latents['ztL'] = ztL_padded   # [N, max_T, q]

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
                      metrics (binary: accuracy; multi-class: macro F1 / Recall / AUROC)
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

            # Overall metrics bar chart
            # Binary: accuracy only; multi-class: f1_macro / recall_macro / auroc_macro
            sample_metrics = next(iter(model_results.values()))['metrics']
            if 'accuracy' in sample_metrics:
                metric_keys = ['accuracy']
            else:
                metric_keys = ['f1_macro', 'recall_macro', 'auroc_macro']
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


def run_trajectory_analysis(eval_latents: dict, eval_metadata: list,
                             dataset_name: str, out_root: str,
                             seg_type: str | None = None) -> None:
    """PCA decomposition of latent trajectories + per-class phase portrait and speed.

    Parameters
    ----------
    eval_latents  : dict with key 'ztL' → ndarray [N, T, q]
    eval_metadata : list of N metadata dicts with 'labels' key
    dataset_name  : used to gate MedalCare-XL-specific plots
    out_root      : directory under which 'trajectory/' sub-folder is created
    seg_type      : 'atrial' | 'ventricular' | None — used only for plot titles
    """
    if 'ztL' not in eval_latents:
        print("  [trajectory] No 'ztL' in eval_latents — skipping trajectory analysis.")
        return

    from sklearn.decomposition import PCA as _PCA

    zt_mean = eval_latents['ztL']          # [N, T, q]  (already numpy)
    N, T, q = zt_mean.shape
    out_dir = os.path.join(out_root, 'trajectory')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  [trajectory] N={N}  T={T}  q={q}  → fitting PCA(2) on {N*T} points")

    # ── PCA on all (N*T) latent points ───────────────────────────────────────
    zt_flat = zt_mean.reshape(N * T, q)
    pca     = _PCA(n_components=2)
    zt_2d   = pca.fit_transform(zt_flat).reshape(N, T, 2)   # [N, T, 2]
    var_exp = pca.explained_variance_ratio_
    print(f"  [trajectory] PCA explained variance: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}")

    # ── Speed in original latent space ────────────────────────────────────────
    zt_velocity = np.diff(zt_mean, axis=1)              # [N, T-1, q]
    zt_speed    = np.linalg.norm(zt_velocity, axis=-1)  # [N, T-1]

    # ── Class labels ──────────────────────────────────────────────────────────
    classes = np.array([m.get('labels', {}).get('class', None) for m in eval_metadata])

    # ── Balanced sinus resampling (MedalCare-XL only) ────────────────────────
    if dataset_name == 'medalcare-xl':
        _, bal_idx = _resample_sinus_balanced_idx(classes)
        zt_2d_bal   = zt_2d[bal_idx]
        speed_bal   = zt_speed[bal_idx]
        classes_bal = classes[bal_idx]
    else:
        zt_2d_bal   = zt_2d
        speed_bal   = zt_speed
        classes_bal = classes

    unique_classes = [c for c in sorted(set(classes_bal)) if c is not None]
    cmap           = plt.cm.tab20
    n_cls          = len(unique_classes)
    cls_colours    = {cls: cmap(i / max(n_cls - 1, 1)) for i, cls in enumerate(unique_classes)}

    # ── Plot 1: Phase portrait (PC1 vs PC2, time as colour) per class ────────
    time_cmap = plt.cm.plasma
    t_norm    = plt.Normalize(vmin=0, vmax=T - 1)

    fig, axes = plt.subplots(
        1, n_cls,
        figsize=(max(4 * n_cls, 8), 4),
        squeeze=False,
    )
    fig.suptitle(
        f"Phase Portrait — mean latent trajectory per class\n"
        f"({seg_type or dataset_name})  "
        f"PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}",
        fontsize=11, fontweight='bold',
    )

    for ci, cls in enumerate(unique_classes):
        ax   = axes[0][ci]
        mask = classes_bal == cls
        traj = zt_2d_bal[mask]          # [n_cls, T, 2]
        mean_traj = traj.mean(axis=0)   # [T, 2]

        # Draw individual trajectories faintly
        for n in range(min(len(traj), 50)):   # cap at 50 to avoid overplotting
            ax.plot(traj[n, :, 0], traj[n, :, 1],
                    color=cls_colours[cls], alpha=0.08, linewidth=0.6, zorder=1)

        # Draw mean trajectory with time-coloured scatter
        sc = ax.scatter(mean_traj[:, 0], mean_traj[:, 1],
                        c=np.arange(T), cmap=time_cmap, norm=t_norm,
                        s=20, zorder=3, edgecolors='none')
        ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                color=cls_colours[cls], linewidth=1.8, zorder=2, alpha=0.85)

        # Start / end markers
        ax.scatter(*mean_traj[0],  marker='o', s=60, color='black',  zorder=5, label='start')
        ax.scatter(*mean_traj[-1], marker='X', s=60, color='red',    zorder=5, label='end')

        ax.set_title(f"{cls}\n(n={mask.sum()})", fontsize=9, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=8)
        if ci == 0:
            ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    # Shared colourbar for time
    sm = plt.cm.ScalarMappable(cmap=time_cmap, norm=t_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], shrink=0.7, pad=0.02)
    cbar.set_label('Time step', fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'phase_portrait.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [trajectory] Saved {out_path}")

    # ── Plot 2: Mean latent speed per timepoint per class ────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    t_axis = np.arange(T - 1)

    for cls in unique_classes:
        mask      = classes_bal == cls
        cls_speed = speed_bal[mask]          # [n_cls, T-1]
        mean_spd  = cls_speed.mean(axis=0)   # [T-1]
        std_spd   = cls_speed.std(axis=0)

        colour = cls_colours[cls]
        ax2.plot(t_axis, mean_spd, color=colour, linewidth=1.8,
                 label=f"{cls} (n={mask.sum()})", zorder=3)
        ax2.fill_between(t_axis,
                         mean_spd - std_spd,
                         mean_spd + std_spd,
                         color=colour, alpha=0.15, zorder=2)

    ax2.set_xlabel('Time step', fontsize=10)
    ax2.set_ylabel('Latent speed  ‖Δz‖₂', fontsize=10)
    ax2.set_title(
        f"Mean latent speed per class  ({seg_type or dataset_name})",
        fontsize=11, fontweight='bold',
    )
    ax2.legend(fontsize=8, framealpha=0.85, ncol=2)
    ax2.spines[['top', 'right']].set_visible(False)
    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, 'latent_speed.png')
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"  [trajectory] Saved {out_path2}")


def _resample_sinus_balanced_idx(classes: np.ndarray,
                                  seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (full_idx_array, balanced_idx_array) for the balanced sinus set.

    Mirrors the logic of _resample_sinus_balanced but works on raw class arrays
    and returns indices rather than resampled arrays, so it can be applied to
    any derived array (zt_2d, zt_speed, …).
    """
    rng = np.random.default_rng(seed)
    sinus_idx     = np.where(classes == 'sinus')[0]
    non_sinus_idx = np.where(classes != 'sinus')[0]

    if len(non_sinus_idx) == 0 or len(sinus_idx) == 0:
        all_idx = np.arange(len(classes))
        return all_idx, all_idx

    cls_counts = {}
    for c in classes[non_sinus_idx]:
        cls_counts[c] = cls_counts.get(c, 0) + 1
    target_n = max(cls_counts.values())

    if len(sinus_idx) >= target_n:
        sel_sinus = np.sort(sinus_idx)[:target_n]
    else:
        extra = rng.choice(non_sinus_idx, size=target_n - len(sinus_idx), replace=True)
        sel_sinus = np.concatenate([sinus_idx, extra])

    all_idx = np.sort(np.concatenate([sel_sinus, non_sinus_idx]))
    return np.arange(len(classes)), all_idx


def run_post_training_probes(args, model, device, trainset, testset, task_params, run,
                              validset=None):
    """Load best checkpoint, collect latents, run OLS linear probes, log to wandb.

    Evaluation is done on the **combined valid + test** set (when validset is
    provided); otherwise test only.

    Results are saved to:
        {args.save}/latents/           — z0/m arrays + metadata JSON
        {args.save}/final_finetune_results/  — per-param metrics.json + plots

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

    va_latents: dict = {}
    va_metadata: list = []

    with np.errstate(all='ignore'):
        print("Collecting train latents...")
        tr_latents, tr_metadata = collect_latents(
            trainset, model, task_params, args, device,
            aladin_metadata=_load_aladin_meta('train'))
        if validset is not None:
            print("Collecting valid latents...")
            va_latents, va_metadata = collect_latents(
                validset, model, task_params, args, device,
                aladin_metadata=_load_aladin_meta('valid'))
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
    if validset is not None:
        np.savez(os.path.join(latents_dir, 'valid_latents.npz'), **va_latents)
        with open(os.path.join(latents_dir, 'valid_metadata.json'), 'w') as f:
            json.dump(va_metadata, f, indent=2)
    print(f"  Saved latents to {latents_dir}")

    # Remap MedalCare-XL class labels for the segment type
    if dataset_name == 'medalcare-xl' and seg_type:
        tr_metadata = _remap_metadata(tr_metadata, seg_type)
        te_metadata = _remap_metadata(te_metadata, seg_type)
        if validset is not None:
            va_metadata = _remap_metadata(va_metadata, seg_type)

    # Combine valid + test into a single evaluation split
    if validset is not None:
        eval_latents = {}
        for k in te_latents:
            arrs = [va_latents[k], te_latents[k]]
            if arrs[0].ndim == 3:
                global_max_T = max(a.shape[1] for a in arrs)
                padded = []
                for a in arrs:
                    if a.shape[1] < global_max_T:
                        a = np.pad(a, [(0, 0), (0, global_max_T - a.shape[1]), (0, 0)])
                    padded.append(a)
                eval_latents[k] = np.concatenate(padded, axis=0)
            else:
                eval_latents[k] = np.concatenate(arrs, axis=0)
        eval_metadata = va_metadata + te_metadata
        print(f"  Eval set: valid ({len(va_metadata)}) + test ({len(te_metadata)}) "
              f"= {len(eval_metadata)} samples")
    else:
        eval_latents = te_latents
        eval_metadata = te_metadata

    # Build combined latent key when modulator is present
    has_m      = 'm' in tr_latents and 'm' in eval_latents
    has_sample = 'z0_sample' in tr_latents and 'z0_sample' in eval_latents
    if has_m:
        tr_latents['z0_m']   = np.concatenate([tr_latents['z0'],   tr_latents['m']],   axis=1)
        eval_latents['z0_m'] = np.concatenate([eval_latents['z0'], eval_latents['m']], axis=1)
    if has_sample and has_m:
        tr_latents['z0_sample_m']   = np.concatenate([tr_latents['z0_sample'],   tr_latents['m']],   axis=1)
        eval_latents['z0_sample_m'] = np.concatenate([eval_latents['z0_sample'], eval_latents['m']], axis=1)

    run_label     = seg_type if seg_type else 'all_classes'
    finetune_root = os.path.join(args.save, 'final_finetune_results', run_label)

    # Params to skip in linear probing (patient_id is not a useful probe target)
    probe_skip = {'patient_id'} if dataset_name == 'medalcare-xl' else None

    # n_clusters: inferred from classes for MedalCare-XL, fixed 8 for UK Biobank
    gmm_n_clusters = None if dataset_name == 'medalcare-xl' else 8

    latent_keys = ['z0'] + (['m', 'z0_m'] if has_m else [])
    if has_sample:
        latent_keys += ['z0_sample'] + (['z0_sample_m'] if has_m else [])
    for lkey in latent_keys:
        print(f"\n=== Linear probes ({lkey}) ===")
        with np.errstate(all='ignore'):
            probe_results = run_linear_probes(
                tr_latents,   tr_metadata,
                eval_latents, eval_metadata,
                latent_key=lkey,
                out_root=os.path.join(finetune_root, lkey),
                methods={'ols'},
                skip_params=probe_skip,
                balance_sinus=(dataset_name == 'medalcare-xl'),
            )
        log_probe_metrics(probe_results, lkey, seg_type, run)

        # ── GMM clustering ────────────────────────────────────────────────────
        print(f"\n=== GMM clustering ({lkey}) ===")
        with np.errstate(all='ignore'):
            run_gmm_clustering(
                tr_latents,   tr_metadata,
                eval_latents, eval_metadata,
                latent_key=lkey,
                dataset_name=dataset_name,
                n_clusters=gmm_n_clusters,
                pca_dim=10,
                out_root=finetune_root,
                run=run,
            )

        # ── MedalCare-XL extra clusterings ───────────────────────────────────
        if dataset_name == 'medalcare-xl':
            print(f"\n=== Patient-ID clustering ({lkey}) ===")
            with np.errstate(all='ignore'):
                run_gmm_clustering(
                    tr_latents,   tr_metadata,
                    eval_latents, eval_metadata,
                    latent_key=lkey,
                    dataset_name=dataset_name,
                    n_clusters=2,
                    pca_dim=10,
                    out_root=finetune_root,
                    run=run,
                    class_label_key='patient_id',
                    tag='patient_id',
                )

            print(f"\n=== GMM clustering k=10 ({lkey}) ===")
            with np.errstate(all='ignore'):
                run_gmm_clustering(
                    tr_latents,   tr_metadata,
                    eval_latents, eval_metadata,
                    latent_key=lkey,
                    dataset_name=dataset_name,
                    n_clusters=10,
                    pca_dim=10,
                    out_root=finetune_root,
                    run=run,
                    tag='k10',
                )

    # ── Latent trajectory analysis ────────────────────────────────────────────
    print("\n=== Latent trajectory analysis ===")
    run_trajectory_analysis(
        eval_latents, eval_metadata,
        dataset_name=dataset_name,
        out_root=finetune_root,
        seg_type=seg_type,
    )

    print("========== Post-training probes complete ==========\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run linear probes on pre-saved latents.')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Directory containing the saved latent .npz and metadata .json files.')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='Splits to load; first is training, rest are combined for eval '
                             '(default: train valid test).')
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
    parser.add_argument('--dataset', type=str, default='medalcare-xl',
                        choices=['medalcare-xl', 'uk-biobank'],
                        help='Dataset name — controls GMM cluster count and patient_id handling.')
    parser.add_argument('--plot_latent_reduct', type=bool, default=False,
                        help='Plot UMP and T-SNE')
    args = parser.parse_args()

    root_dir = args.root_dir
    methods  = set(args.methods) if args.methods else {'ols'}

    # Output directory: named by seg_type when provided so runs don't overwrite each other;
    # otherwise use a timestamp to guarantee uniqueness.
    if args.seg_type:
        run_label = args.seg_type
    else:
        run_label = 'all_classes'

    finetune_root = os.path.normpath(os.path.join(root_dir, 'final_finetune_results', run_label))

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
    eval_splits = args.splits[1:]   # e.g. ['valid', 'test']

    all_probe_splits = [train_split] + eval_splits
    has_m      = all('m' in latents_dict[s]['latents'] for s in all_probe_splits)
    has_sample = all('z0_sample' in latents_dict[s]['latents'] for s in all_probe_splits)

    if has_m:
        for split in all_probe_splits:
            lats = latents_dict[split]['latents']
            lats['z0_m'] = np.concatenate([lats['z0'], lats['m']], axis=1)
    if has_sample and has_m:
        for split in all_probe_splits:
            lats = latents_dict[split]['latents']
            lats['z0_sample_m'] = np.concatenate([lats['z0_sample'], lats['m']], axis=1)

    # Combine all eval splits (valid + test, or just test if only one)
    def _combine(splits):
        lats_list = [latents_dict[s]['latents'] for s in splits]
        meta_list = [latents_dict[s]['metadata'] for s in splits]
        keys = lats_list[0].keys()
        combined_lats = {}
        for k in keys:
            arrs = [l[k] for l in lats_list]
            if arrs[0].ndim == 3:
                # Variable-T latents (e.g. ztL): pad each split to the global max_T first
                global_max_T = max(a.shape[1] for a in arrs)
                padded = []
                for a in arrs:
                    if a.shape[1] < global_max_T:
                        pad_width = [(0, 0), (0, global_max_T - a.shape[1]), (0, 0)]
                        a = np.pad(a, pad_width)
                    padded.append(a)
                combined_lats[k] = np.concatenate(padded, axis=0)
            else:
                combined_lats[k] = np.concatenate(arrs, axis=0)
        combined_meta = [entry for m in meta_list for entry in m]
        return combined_lats, combined_meta

    eval_latents, eval_metadata = _combine(eval_splits)
    tr_latents  = latents_dict[train_split]['latents']
    tr_metadata = latents_dict[train_split]['metadata']

    n_eval = len(eval_metadata)
    print(f"\nEval set: {' + '.join(eval_splits)} = {n_eval} samples")
    print(f"Output directory: {finetune_root}")
    print(f"Methods: {sorted(methods)}\n")

    dataset_name   = args.dataset.lower()
    probe_skip     = {'patient_id'} if dataset_name == 'medalcare-xl' else None
    gmm_n_clusters = None if dataset_name == 'medalcare-xl' else 8

    latent_keys = ['z0'] + (['m', 'z0_m'] if has_m else [])
    if has_sample:
        latent_keys += ['z0_sample'] + (['z0_sample_m'] if has_m else [])
    for lkey in latent_keys:
        print(f"\n=== Linear probes ({lkey}) ===")
        run_linear_probes(
            tr_latents,   tr_metadata,
            eval_latents, eval_metadata,
            latent_key=lkey,
            out_root=os.path.join(finetune_root, lkey),
            methods=methods,
            skip_params=probe_skip,
            balance_sinus=(dataset_name == 'medalcare-xl'),
        )

        print(f"\n=== GMM clustering ({lkey}) ===")
        with np.errstate(all='ignore'):
            run_gmm_clustering(
                tr_latents,   tr_metadata,
                eval_latents, eval_metadata,
                latent_key=lkey,
                dataset_name=dataset_name,
                n_clusters=gmm_n_clusters,
                pca_dim=10,
                out_root=finetune_root,
            )

        if dataset_name == 'medalcare-xl':
            print(f"\n=== Patient-ID clustering ({lkey}) ===")
            with np.errstate(all='ignore'):
                run_gmm_clustering(
                    tr_latents,   tr_metadata,
                    eval_latents, eval_metadata,
                    latent_key=lkey,
                    dataset_name=dataset_name,
                    n_clusters=2,
                    pca_dim=10,
                    out_root=finetune_root,
                    class_label_key='patient_id',
                    tag='patient_id',
                )

            print(f"\n=== GMM clustering k=10 ({lkey}) ===")
            with np.errstate(all='ignore'):
                run_gmm_clustering(
                    tr_latents,   tr_metadata,
                    eval_latents, eval_metadata,
                    latent_key=lkey,
                    dataset_name=dataset_name,
                    n_clusters=10,
                    pca_dim=10,
                    out_root=finetune_root,
                    tag='k10',
                )
