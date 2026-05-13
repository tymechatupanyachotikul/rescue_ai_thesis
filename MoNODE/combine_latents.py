#!/usr/bin/env python3
"""
combine_latents.py — Concatenate latents from two pre-trained models and run probes.

Loads pre-saved latent files (latent_tensors_{split}.npz / latent_meta_dict_{split}.json)
from two separate model run directories, matches samples by patient_id, averages within
each patient (handles multiple beats per patient), then concatenates the specified
latent key across models before running the same classification/regression probes
as finetune.py.

Summary JSON is saved in the same format as summarize_results.py with
segment_type='combined', ready to be consumed by compare_disentangle.py.

Directory layout written by this script
----------------------------------------
{output_dir}/
  final_finetune_results/
    combined/
      {latent_key}/
        classification/{param}/metrics.json    (+ confusion matrix PNGs)
        regression/{param}/metrics.json        (+ scatter PNGs)
  training_metrics.json   (placeholder — no ODE training MSE for combined model)
  args.json

Usage:
  python combine_latents.py \\
      --model1_latents_dir results/ecg/node/atrial_run/latents \\
      --model2_latents_dir results/ecg/node/ventricular_run/latents \\
      --latent_key z0 \\
      --output_dir results/combined/node_atrial_ventricular \\
      --model node \\
      --dataset medalcare-xl \\
      --model1_segment atrial \\
      --model2_segment ventricular \\
      --summary_output_dir results/summaries
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

# Reuse probing infrastructure from finetune.py
sys.path.insert(0, os.path.dirname(__file__))
from finetune import (
    _load_split, run_linear_probes,
    run_ptbxl_multilabel_probes, run_ptbxl_label_efficiency_probes,
)
from summarize_results import save_run_summary


# ─── mutual information ───────────────────────────────────────────────────────

def _safe_log_pseudodet(A: np.ndarray, eps: float = 1e-9) -> float:
    """Log pseudo-determinant: sum of log of positive eigenvalues."""
    eigvals = np.linalg.eigvalsh(A)
    pos     = eigvals[eigvals > eps]
    return float(np.sum(np.log(pos))) if len(pos) > 0 else float('-inf')


def compute_mutual_information(
    X1: np.ndarray,
    X2: np.ndarray,
    pca_dim: int = 10,
    n_cca: int   = 5,
    random_state: int = 42,
) -> dict:
    """Estimate mutual information between two latent spaces.

    Three complementary estimators are computed:

    1. **Gaussian MI** (full-dim + PCA-reduced)
       Assumes both spaces are jointly Gaussian:
         MI(X1; X2) = 0.5 * (log|Σ1| + log|Σ2| - log|Σ_joint|)
       Uses log pseudo-determinant for robustness when D >> N.

    2. **Feature-wise MI** (sklearn k-NN estimator, averaged)
       For each pair (feature_i in X1, feature_j in X2), estimate MI via the
       Kraskov k-NN method (`mutual_info_regression`).  Reports mean, max, and
       the D1×D2 matrix (saved but not printed in full).

    3. **CCA canonical correlations**
       Canonical Correlation Analysis finds the directions of maximal linear
       correlation.  The canonical correlations serve as a linear lower bound on
       the shared information.

    Parameters
    ----------
    X1, X2      : matched latent arrays, shape [N, D1] and [N, D2]
    pca_dim     : dimensionality to reduce to before Gaussian MI estimation
    n_cca       : number of canonical components
    random_state: seed for sklearn estimators

    Returns
    -------
    dict with all MI estimates, safe to JSON-serialise (no ndarray values).
    """
    N, D1 = X1.shape
    _,  D2 = X2.shape

    # ── Standardise ───────────────────────────────────────────────────────────
    X1s = StandardScaler().fit_transform(X1)
    X2s = StandardScaler().fit_transform(X2)

    results: dict = {
        'n_samples': N,
        'dim1': D1,
        'dim2': D2,
        'pca_dim_used': None,
    }

    # ── 1. Gaussian MI (full dimensionality) ─────────────────────────────────
    try:
        Sigma1 = np.cov(X1s.T)
        Sigma2 = np.cov(X2s.T)
        SigmaJ = np.cov(np.concatenate([X1s, X2s], axis=1).T)

        ld1 = _safe_log_pseudodet(np.atleast_2d(Sigma1))
        ld2 = _safe_log_pseudodet(np.atleast_2d(Sigma2))
        ldJ = _safe_log_pseudodet(SigmaJ)
        mi_gauss_full = 0.5 * (ld1 + ld2 - ldJ)
        results['gaussian_mi_full'] = round(float(mi_gauss_full), 6)
    except Exception as e:
        results['gaussian_mi_full'] = None
        results['gaussian_mi_full_error'] = str(e)

    # ── 2. Gaussian MI (PCA-reduced) ──────────────────────────────────────────
    try:
        actual_pca = min(pca_dim, D1, D2, N - 1)
        results['pca_dim_used'] = actual_pca

        pca1 = PCA(n_components=actual_pca, random_state=random_state)
        pca2 = PCA(n_components=actual_pca, random_state=random_state)
        Z1   = pca1.fit_transform(X1s)
        Z2   = pca2.fit_transform(X2s)

        S1 = np.cov(Z1.T)
        S2 = np.cov(Z2.T)
        SJ = np.cov(np.concatenate([Z1, Z2], axis=1).T)

        ld1 = _safe_log_pseudodet(np.atleast_2d(S1))
        ld2 = _safe_log_pseudodet(np.atleast_2d(S2))
        ldJ = _safe_log_pseudodet(SJ)
        mi_gauss_pca = 0.5 * (ld1 + ld2 - ldJ)
        results['gaussian_mi_pca'] = round(float(mi_gauss_pca), 6)

        # Fraction of variance explained by PCA
        results['pca_var_explained_1'] = round(float(pca1.explained_variance_ratio_.sum()), 4)
        results['pca_var_explained_2'] = round(float(pca2.explained_variance_ratio_.sum()), 4)
    except Exception as e:
        results['gaussian_mi_pca'] = None
        results['gaussian_mi_pca_error'] = str(e)

    # ── 3. Feature-wise MI (Kraskov k-NN, averaged over all feature pairs) ───
    try:
        # For each feature j in X2, compute MI with every feature of X1 → [D2, D1]
        mi_matrix = np.zeros((D2, D1))
        for j in range(D2):
            mi_matrix[j] = mutual_info_regression(
                X1s, X2s[:, j], random_state=random_state
            )
        results['feature_mi_mean'] = round(float(mi_matrix.mean()), 6)
        results['feature_mi_max']  = round(float(mi_matrix.max()),  6)
        results['feature_mi_sum']  = round(float(mi_matrix.sum()),  6)
        # Per-feature summary (mean MI of each X2 feature with all X1 features)
        results['feature_mi_per_dim2'] = [round(float(v), 6) for v in mi_matrix.mean(axis=1)]
    except Exception as e:
        results['feature_mi_mean'] = None
        results['feature_mi_error'] = str(e)

    # ── 4. CCA canonical correlations ─────────────────────────────────────────
    try:
        n_comp = min(n_cca, D1, D2, N // 5)
        if n_comp >= 1:
            cca = CCA(n_components=n_comp)
            Z1c, Z2c = cca.fit_transform(X1s, X2s)
            corrs = [
                round(float(np.corrcoef(Z1c[:, i], Z2c[:, i])[0, 1]), 6)
                for i in range(n_comp)
            ]
            results['cca_correlations'] = corrs
            results['cca_mean_correlation'] = round(float(np.mean(corrs)), 6)
        else:
            results['cca_correlations'] = []
            results['cca_mean_correlation'] = None
    except Exception as e:
        results['cca_correlations'] = []
        results['cca_mean_correlation'] = None
        results['cca_error'] = str(e)

    return results


def print_mi_summary(mi: dict) -> None:
    print("\n── Mutual Information between latent spaces ──")
    print(f"  Samples: {mi['n_samples']}   Dim1: {mi['dim1']}   Dim2: {mi['dim2']}")
    if mi.get('gaussian_mi_full') is not None:
        print(f"  Gaussian MI (full dim) : {mi['gaussian_mi_full']:.4f} nats")
    if mi.get('gaussian_mi_pca') is not None:
        print(f"  Gaussian MI (PCA-{mi['pca_dim_used']}d)  : {mi['gaussian_mi_pca']:.4f} nats"
              f"  [var_exp: {mi.get('pca_var_explained_1', '?'):.1%} / "
              f"{mi.get('pca_var_explained_2', '?'):.1%}]")
    if mi.get('feature_mi_mean') is not None:
        print(f"  Feature-wise MI (mean) : {mi['feature_mi_mean']:.4f} nats  "
              f"max={mi['feature_mi_max']:.4f}")
    if mi.get('cca_correlations'):
        corrs_str = '  '.join(f'{c:.3f}' for c in mi['cca_correlations'])
        print(f"  CCA correlations       : {corrs_str}")
        print(f"  CCA mean correlation   : {mi['cca_mean_correlation']:.4f}")


# ─── PTB-XL data statistics ───────────────────────────────────────────────────

_PTBXL_SUPERCLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def _get_pids(metadata: list) -> set:
    return {str(e.get('uid') or e.get('patient_id', '')) for e in metadata}


def _ptbxl_class_counts(metadata: list) -> dict:
    """Count patients per PTB-XL superclass (multi-hot) plus totals."""
    counts = {cls: 0 for cls in _PTBXL_SUPERCLASS_NAMES}
    counts['_total']    = len(metadata)
    counts['_no_class'] = 0
    for entry in metadata:
        sc = entry.get('labels', {}).get('superclass')
        if not isinstance(sc, list):
            counts['_no_class'] += 1
            continue
        has_any = False
        for i, v in enumerate(sc):
            if v and i < len(_PTBXL_SUPERCLASS_NAMES):
                counts[_PTBXL_SUPERCLASS_NAMES[i]] += 1
                has_any = True
        if not has_any:
            counts['_no_class'] += 1
    return counts


def _compute_ptbxl_statistics(
    tr_meta1: list, tr_meta2: list, tr_meta_combined: list,
    ev_meta1: list, ev_meta2: list, te_meta_combined: list,
) -> dict:
    """Compute per-class counts and discard statistics for PTB-XL."""
    pids_tr1      = _get_pids(tr_meta1)
    pids_tr2      = _get_pids(tr_meta2)
    pids_tr_match = _get_pids(tr_meta_combined)

    pids_ev1      = _get_pids(ev_meta1)
    pids_ev2      = _get_pids(ev_meta2)
    pids_te_match = _get_pids(te_meta_combined)

    return {
        'train': {
            'n_patients_model1':        len(pids_tr1),
            'n_patients_model2':        len(pids_tr2),
            'n_patients_matched':       len(pids_tr_match),
            'n_discarded_model1_only':  len(pids_tr1 - pids_tr2),
            'n_discarded_model2_only':  len(pids_tr2 - pids_tr1),
            'per_class':                _ptbxl_class_counts(tr_meta_combined),
        },
        'eval': {
            'n_patients_model1':        len(pids_ev1),
            'n_patients_model2':        len(pids_ev2),
            'n_patients_matched':       len(pids_te_match),
            'n_discarded_model1_only':  len(pids_ev1 - pids_ev2),
            'n_discarded_model2_only':  len(pids_ev2 - pids_ev1),
            'per_class':                _ptbxl_class_counts(te_meta_combined),
        },
    }


def plot_pca_explained_variance(
    latents: dict,
    out_dir: str,
    max_components: int | None = None,
    keys_to_plot: list[str] | None = None,
) -> None:
    """Plot cumulative PCA explained variance vs number of components.

    Plots z0 and m (and z0_m if present) on the same axes.
    Saves:
      pca_explained_variance.npz  — cumvar + evr arrays per key
      pca_explained_variance.png  — single figure, one curve per key
    """
    os.makedirs(out_dir, exist_ok=True)

    plot_keys = keys_to_plot or [k for k in ('z0', 'm', 'z0_m') if k in latents]
    if not plot_keys:
        print("  PCA explained variance: no latent keys found, skipping.")
        return

    _style: dict[str, dict] = {
        'z0':   {'color': '#1F77B4', 'ls': '-',  'label': 'z₀'},
        'm':    {'color': '#FF7F0E', 'ls': '--', 'label': 'm'},
        'z0_m': {'color': '#2CA02C', 'ls': ':',  'label': 'z₀ + m'},
    }

    npz_data: dict[str, np.ndarray] = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    max_n = 0

    for key in plot_keys:
        X = latents[key].astype(np.float64)
        n_comp = min(X.shape[0] - 1, X.shape[1])
        if max_components is not None:
            n_comp = min(n_comp, max_components)
        if n_comp < 1:
            continue

        pca = PCA(n_components=n_comp)
        pca.fit(StandardScaler().fit_transform(X))

        evr    = pca.explained_variance_ratio_
        cumvar = np.cumsum(evr)
        comp   = np.arange(1, len(cumvar) + 1)
        max_n  = max(max_n, len(cumvar))

        st = _style.get(key, {'color': 'gray', 'ls': '-', 'label': key})
        ax.plot(comp, cumvar, color=st['color'], ls=st['ls'], lw=2, label=st['label'])

        npz_data[f'{key}_cumvar']       = cumvar
        npz_data[f'{key}_evr']          = evr
        npz_data[f'{key}_n_components'] = comp

    for thresh in (0.90, 0.95):
        ax.axhline(thresh, color='gray', lw=0.8, ls=':', alpha=0.7)
        ax.text(max_n * 0.02, thresh + 0.006, f'{thresh:.0%}',
                fontsize=8, color='gray', va='bottom')

    ax.set_xlabel('Number of principal components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title('PCA explained variance — combined latents')
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.7)
    fig.tight_layout()

    png_path = os.path.join(out_dir, 'pca_explained_variance.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  PCA explained variance plot → {png_path}")

    npz_path = os.path.join(out_dir, 'pca_explained_variance.npz')
    np.savez(npz_path, **npz_data)
    print(f"  PCA explained variance data → {npz_path}")


def _print_ptbxl_statistics(stats: dict) -> None:
    for split in ('train', 'eval'):
        s = stats[split]
        print(f"\n  [{split}]")
        print(f"    model1 patients : {s['n_patients_model1']}")
        print(f"    model2 patients : {s['n_patients_model2']}")
        print(f"    matched         : {s['n_patients_matched']}")
        print(f"    discarded (m1 only) : {s['n_discarded_model1_only']}")
        print(f"    discarded (m2 only) : {s['n_discarded_model2_only']}")
        pc = s['per_class']
        print(f"    per-class counts (matched):")
        for cls in _PTBXL_SUPERCLASS_NAMES:
            print(f"      {cls:6s}: {pc.get(cls, 0)}")
        if pc.get('_no_class', 0):
            print(f"      (no superclass label): {pc['_no_class']}")


# ─── matching helpers ─────────────────────────────────────────────────────────

def _group_by_patient_all(latents: dict, metadata: list):
    """Group ALL latent keys by patient_id in a single pass, return per-patient means.

    Handles base keys (z0, m, …) present in `latents` and synthesises z0_m from
    z0 + m if both exist but z0_m is not explicitly stored.

    Returns:
        sorted_pids   : sorted list of patient_ids
        arrs          : {key: np.ndarray [N_patients, D]} for every available key
        labels_by_pid : {patient_id: labels dict}
    """
    from collections import defaultdict

    base_keys = list(latents.keys())
    synth_z0m = ('z0' in latents and 'm' in latents and 'z0_m' not in latents)
    all_keys  = base_keys + (['z0_m'] if synth_z0m else [])

    pid_latents: dict = {k: defaultdict(list) for k in all_keys}
    pid_labels:  dict = {}

    for i, entry in enumerate(metadata):
        pid = str(entry.get('uid'))
        pid_labels[pid] = dict(entry.get('labels', {}))
        for k in base_keys:
            pid_latents[k][pid].append(latents[k][i])
        if synth_z0m:
            pid_latents['z0_m'][pid].append(
                np.concatenate([latents['z0'][i], latents['m'][i]])
            )

    sorted_pids = sorted(pid_labels.keys())
    arrs = {
        k: np.stack([np.mean(pid_latents[k][pid], axis=0) for pid in sorted_pids])
        for k in all_keys
    }
    return sorted_pids, arrs, {pid: pid_labels[pid] for pid in sorted_pids}


def _match_and_combine_all(
    lat1: dict, meta1: list,
    lat2: dict, meta2: list,
    prefer_labels: int = 1,
) -> tuple[dict, list]:
    """Match patients across two models and concatenate ALL shared latent keys.

    Produces combined arrays for every key that exists in both models, plus z0_m
    synthesised from z0+m when needed.

    Returns
    -------
    combined_latents : {key: np.ndarray [N_matched, D1+D2]}  for each shared key
    combined_metadata : list of N_matched dicts with 'patient_id' and 'labels'
    """
    pids1, arrs1, labels1 = _group_by_patient_all(lat1, meta1)
    pids2, arrs2, labels2 = _group_by_patient_all(lat2, meta2)

    shared_keys = sorted(set(arrs1) & set(arrs2))
    if not shared_keys:
        raise ValueError("No common latent keys between the two models.")

    set1, set2 = set(pids1), set(pids2)
    common     = sorted(set1 & set2)
    if not common:
        raise ValueError(
            f"No common patient_ids between the two latent sets "
            f"({len(set1)} vs {len(set2)} patients). "
            "Check that both models were trained on the same dataset split."
        )

    n_only1 = len(set1 - set2)
    n_only2 = len(set2 - set1)
    print(f"  Matched {len(common)} patients "
          f"(model1-only={n_only1}, model2-only={n_only2})")

    idx1 = {pid: i for i, pid in enumerate(pids1)}
    idx2 = {pid: i for i, pid in enumerate(pids2)}

    combined_latents: dict = {}
    for key in shared_keys:
        rows1 = np.stack([arrs1[key][idx1[pid]] for pid in common])
        rows2 = np.stack([arrs2[key][idx2[pid]] for pid in common])
        combined_latents[key] = np.concatenate([rows1, rows2], axis=1)

    combined_metadata = []
    for pid in common:
        if prefer_labels == 1:
            merged = {**labels2[pid], **labels1[pid]}
        else:
            merged = {**labels1[pid], **labels2[pid]}
        combined_metadata.append({'patient_id': pid, 'uid': pid, 'labels': merged})

    return combined_latents, combined_metadata



# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Combine latents from two models and run linear probes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    parser.add_argument('--model1_latents_dir', required=True,
                        help='Directory with latent_tensors_{split}.npz / '
                             'latent_meta_dict_{split}.json for model 1')
    parser.add_argument('--model2_latents_dir', required=True,
                        help='Directory with latent files for model 2')
    parser.add_argument('--latent_key', default='z0',
                        help='Latent key used for MI analysis (probes run for all '
                             'available keys: z0, m, z0_m)')
    parser.add_argument('--model1_segment', default='atrial',
                        help='Segment type of model 1 (for logging)')
    parser.add_argument('--model2_segment', default='ventricular',
                        help='Segment type of model 2 (for logging)')
    parser.add_argument('--prefer_labels', type=int, default=1, choices=[1, 2],
                        help='Which model\'s labels take precedence when both have a key')
    # Probe config
    parser.add_argument('--balance_sinus', type=eval, default=False,
                        help='Resample sinus class in eval set (MedalCare-XL)')
    parser.add_argument('--skip_params', nargs='*', default=None,
                        help='Label parameters to skip during probing')
    # ALADIN metadata
    parser.add_argument('--aladin_metadata_dir', default=None,
                        help='Directory containing {train,valid,test}_metadata.json '
                             'produced by aladin_preprocess.py')
    # Output
    parser.add_argument('--output_dir', required=True,
                        help='Root directory for combined probe outputs')
    parser.add_argument('--model', required=True,
                        help='Model architecture name (e.g. node, vae)')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (e.g. medalcare-xl, uk_biobank)')
    parser.add_argument('--summary_output_dir', default=None,
                        help='Where to write the summary JSON and MI analysis. '
                             'Filenames: combined_{model}_{dataset_slug}.json, '
                             'mi_{model}_{dataset_slug}.json')
    parser.add_argument('--pca_dim', type=int, default=10,
                        help='PCA dimensionality for Gaussian MI estimation')
    parser.add_argument('--n_cca', type=int, default=5,
                        help='Number of CCA canonical components')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Save args ─────────────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── Load ALADIN metadata (optional) ──────────────────────────────────────
    aladin_meta: dict | None = None
    if args.aladin_metadata_dir:
        # Merge train+valid+test into one lookup dict
        aladin_meta = {}
        for split in ('train', 'valid', 'test'):
            p = os.path.join(args.aladin_metadata_dir, f'{split}_metadata.json')
            if os.path.exists(p):
                with open(p) as f:
                    aladin_meta.update(json.load(f))
        print(f"  Loaded ALADIN metadata: {len(aladin_meta)} entries")

    # ── Load latents for each split ──────────────────────────────────────────
    print(f"\n── Model 1 ({args.model1_segment}) ──")
    tr_lat1, tr_meta1 = _load_split(args.model1_latents_dir, 'train', aladin_meta)
    va_lat1, va_meta1 = _load_split(args.model1_latents_dir, 'valid', aladin_meta)
    te_lat1, te_meta1 = _load_split(args.model1_latents_dir, 'test',  aladin_meta)

    print(f"\n── Model 2 ({args.model2_segment}) ──")
    tr_lat2, tr_meta2 = _load_split(args.model2_latents_dir, 'train', aladin_meta)
    va_lat2, va_meta2 = _load_split(args.model2_latents_dir, 'valid', aladin_meta)
    te_lat2, te_meta2 = _load_split(args.model2_latents_dir, 'test',  aladin_meta)

    # Merge valid + test into a single evaluation split per model
    common_keys1 = set(va_lat1) & set(te_lat1)
    common_keys2 = set(va_lat2) & set(te_lat2)
    ev_lat1 = {k: np.concatenate([va_lat1[k], te_lat1[k]], axis=0) for k in common_keys1}
    ev_lat2 = {k: np.concatenate([va_lat2[k], te_lat2[k]], axis=0) for k in common_keys2}
    ev_meta1 = va_meta1 + te_meta1
    ev_meta2 = va_meta2 + te_meta2

    # ── Match and concatenate ALL latent keys ────────────────────────────────
    print(f"\n── Matching train splits ──")
    tr_combined, tr_meta = _match_and_combine_all(
        tr_lat1, tr_meta1, tr_lat2, tr_meta2,
        prefer_labels=args.prefer_labels,
    )
    print(f"\n── Matching eval splits (valid + test) ──")
    te_combined, te_meta = _match_and_combine_all(
        ev_lat1, ev_meta1, ev_lat2, ev_meta2,
        prefer_labels=args.prefer_labels,
    )

    combined_keys = sorted(tr_combined.keys())
    n_train = next(iter(tr_combined.values())).shape[0]
    n_eval  = next(iter(te_combined.values())).shape[0]
    print(f"\n  Combined latent keys : {combined_keys}")
    for key in combined_keys:
        print(f"    {key:8s}: dim={tr_combined[key].shape[1]}")
    print(f"  Train samples        : {n_train}")
    print(f"  Eval  samples        : {n_eval}"
          f"  (valid={len(va_meta1 + va_meta2) // 2}, test={len(te_meta1 + te_meta2) // 2})")

    # ── PCA explained variance ────────────────────────────────────────────────
    print(f"\n── PCA explained variance ──")
    plot_pca_explained_variance(tr_combined, os.path.join(args.output_dir, 'pca'))

    # ── Mutual information between the two latent spaces ─────────────────────
    # Use args.latent_key (default z0) — most interpretable for MI
    mi_key = args.latent_key if args.latent_key in tr_combined else combined_keys[0]

    def _key_dim_single(lat: dict, key: str) -> int:
        if key in lat:
            return lat[key].shape[1]
        if key == 'z0_m' and 'z0' in lat and 'm' in lat:
            return lat['z0'].shape[1] + lat['m'].shape[1]
        raise KeyError(f"Latent key '{key}' not found. Available: {list(lat.keys())}")

    d1 = _key_dim_single(tr_lat1, mi_key)

    print(f"\n── Computing mutual information (key: {mi_key}) ──")
    X_combined_tr = tr_combined[mi_key]   # [N, D1+D2]
    X1_tr = X_combined_tr[:, :d1]
    X2_tr = X_combined_tr[:, d1:]

    mi_results = compute_mutual_information(
        X1_tr, X2_tr,
        pca_dim=args.pca_dim,
        n_cca=args.n_cca,
    )
    mi_results['model']          = args.model
    mi_results['dataset']        = args.dataset
    mi_results['latent_key']     = mi_key
    mi_results['model1_segment'] = args.model1_segment
    mi_results['model2_segment'] = args.model2_segment
    print_mi_summary(mi_results)

    # Save MI results locally
    mi_local_path = os.path.join(args.output_dir, 'mi_analysis.json')
    with open(mi_local_path, 'w') as f:
        json.dump(mi_results, f, indent=2)
    print(f"  Saved: {mi_local_path}")

    # Save MI results to summary output dir (for compare_disentangle.py)
    if args.summary_output_dir:
        os.makedirs(args.summary_output_dir, exist_ok=True)
        dataset_slug = args.dataset.lower().replace('-', '_').replace(' ', '_')
        mi_summary_path = os.path.join(
            args.summary_output_dir,
            f'mi_{args.model}_{dataset_slug}.json',
        )
        with open(mi_summary_path, 'w') as f:
            json.dump(mi_results, f, indent=2)
        print(f"  Saved: {mi_summary_path}")

    # ── Dataset flags ─────────────────────────────────────────────────────────
    dataset_lower = args.dataset.lower()
    is_medalcare  = 'medalcare' in dataset_lower
    is_ptbxl      = 'ptb' in dataset_lower
    is_ukbb       = 'uk' in dataset_lower and 'biobank' in dataset_lower

    # ── Run probes for each combined latent key ───────────────────────────────
    # probe_out_parent: {output_dir}/final_finetune_results/combined/
    # PTB-XL OVR functions append /{latent_key}/ themselves.
    # run_linear_probes receives the full path including latent_key.
    probe_out_parent = os.path.join(
        args.output_dir, 'final_finetune_results', 'combined')
    os.makedirs(probe_out_parent, exist_ok=True)

    _skip = set(args.skip_params or []) | {'patient_id'}

    if is_ptbxl:
        # PTB-XL data statistics (computed once — same patients for all keys)
        print(f"\n── PTB-XL data statistics ──")
        ptbxl_stats = _compute_ptbxl_statistics(
            tr_meta1, tr_meta2, tr_meta,
            ev_meta1, ev_meta2, te_meta,
        )
        _print_ptbxl_statistics(ptbxl_stats)
        stats_path = os.path.join(args.output_dir, 'ptbxl_data_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(ptbxl_stats, f, indent=2)
        print(f"  Saved: {stats_path}")

        for key in combined_keys:
            print(f"\n── Running PTB-XL multi-label OVR probes  [{key}] ──")
            run_ptbxl_multilabel_probes(
                tr_latents=tr_combined,
                tr_metadata=tr_meta,
                eval_latents=te_combined,
                eval_metadata=te_meta,
                latent_key=key,
                out_root=probe_out_parent,
            )
            print(f"\n── Running PTB-XL label efficiency OVR probes  [{key}] ──")
            run_ptbxl_label_efficiency_probes(
                tr_latents=tr_combined,
                tr_metadata=tr_meta,
                eval_latents=te_combined,
                eval_metadata=te_meta,
                latent_key=key,
                out_root=probe_out_parent,
                fractions=[0.01, 0.10, 0.50, 1.00],
            )
    else:
        for key in combined_keys:
            print(f"\n── Running linear probes  [{key}] ──")
            probe_out_root = os.path.join(probe_out_parent, key)
            os.makedirs(probe_out_root, exist_ok=True)
            run_linear_probes(
                train_latents=tr_combined,
                train_metadata=tr_meta,
                test_latents=te_combined,
                test_metadata=te_meta,
                latent_key=key,
                out_root=probe_out_root,
                skip_params=_skip,
                methods={'ols'},
                balance_sinus=(is_medalcare and args.balance_sinus),
                use_target_scaling=is_ukbb,
            )

    # ── Write placeholder training_metrics.json ───────────────────────────────
    training_stub = {
        'note': 'No ODE training MSE for combined latent model.',
        'model1_segment': args.model1_segment,
        'model2_segment': args.model2_segment,
        'combined_keys':  combined_keys,
        'latent_dims':    {k: int(tr_combined[k].shape[1]) for k in combined_keys},
        'n_train':        int(n_train),
        'n_eval':         int(n_eval),
        'eval_note':      'valid + test combined',
    }
    with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(training_stub, f, indent=2)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    if args.summary_output_dir:
        save_run_summary(
            run_dir=args.output_dir,
            output_dir=args.summary_output_dir,
            model=args.model,
            dataset=args.dataset,
            segment_type='combined',
            original_dir=None
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
