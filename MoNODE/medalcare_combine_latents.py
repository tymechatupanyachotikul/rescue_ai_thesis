#!/usr/bin/env python3
"""
medalcare_combine_latents.py — Combine atrial + ventricular latents for MedalCare-XL
and run classification / regression probes.

Workflow
--------
1. Load pre-saved train latents from both model directories (atrial, ventricular).
2. Override labels from ALADIN metadata JSON files; UIDs are derived from the
   original file-path stem stored in each metadata entry.
3. Merge valid + test splits into a single evaluation set.
4. Match patients across atrial and ventricular models, then concatenate their
   latent vectors (identical logic to combine_latents.py).
5. Run the same probes as finetune.py / combine_latents.py.
6. Write output in the same directory layout and summary-JSON format as
   summarize_results.py so it can be consumed by analyse_results.py.

Pre-requisites
--------------
Both model dirs must already contain a `latents/` sub-directory produced by
finetune.py or run_post_training_probes() with {split}_latents.npz /
{split}_metadata.json for train, valid, and test.

Usage
-----
  python medalcare_combine_latents.py \\
      --atrial_model_dir      results/ecg/node/atrial_run \\
      --ventricular_model_dir results/ecg/node/ventricular_run \\
      --aladin_metadata_dir   /path/to/aladin_output \\
      --output_dir            results/combined/medalcare_node \\
      --model                 node \\
      --latent_key            z0 \\
      --summary_output_dir    results/summaries
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from finetune import _load_split, run_linear_probes
from combine_latents import _match_and_combine, compute_mutual_information, print_mi_summary
from summarize_results import save_run_summary


# ─── MedalCare-XL uid helpers ─────────────────────────────────────────────────

def _uid_from_mc_path(path: str) -> str:
    """Derive the MedalCare-XL uid from an original file path.

    Mirrors _parse_medalcare_ids() in aladin_preprocess.py:
      uid = {run_id}_{session_id}_{class}
    where:
      run_id     = parts[-2].split('_')[1]
      session_id = parts[-1].split('_')[0]
      class      = parts[-4]
    """
    parts = path.replace('\\', '/').split('/')
    try:
        run_id     = parts[-2].split('_')[1]
        session_id = parts[-1].split('_')[0]
        cls        = parts[-4]
        return f'{run_id}_{session_id}_{cls}'
    except IndexError:
        # Fallback: use file stem
        return os.path.splitext(os.path.basename(path))[0]


def _load_aladin_metadata(metadata_dir: str, splits: tuple[str, ...] = ('train', 'valid', 'test')) -> dict:
    """Load and merge ALADIN metadata JSONs into a single uid-keyed dict.

    Each JSON is either:
      - A dict  keyed by uid  (aladin_preprocess.py output)
      - A list  of entries with 'uid' field  (finetune.py style)
    """
    merged: dict = {}
    for split in splits:
        p = os.path.join(metadata_dir, f'{split}_metadata.json')
        if not os.path.exists(p):
            print(f"  [warn] ALADIN metadata not found: {p}")
            continue
        with open(p) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            merged.update(raw)
        else:
            for entry in raw:
                uid = entry.get('uid') or entry.get('patient_id')
                if uid:
                    merged[str(uid)] = entry
        print(f"  ALADIN [{split}]: {len(raw) if isinstance(raw, dict) else len(raw)} entries")
    print(f"  Total ALADIN entries: {len(merged)}")
    return merged


# ─── split helpers ────────────────────────────────────────────────────────────

def _concat_splits(lat_a: dict, meta_a: list,
                   lat_b: dict, meta_b: list) -> tuple[dict, list]:
    """Concatenate two splits. Only keys present in both dicts are kept."""
    common_keys = set(lat_a.keys()) & set(lat_b.keys())
    combined_lat  = {k: np.concatenate([lat_a[k], lat_b[k]], axis=0) for k in common_keys}
    combined_meta = meta_a + meta_b
    return combined_lat, combined_meta


def _load_model_splits(
    model_dir: str,
    segment: str,
    aladin_meta: dict | None,
) -> tuple[dict, list, dict, list]:
    """Load train and eval (valid+test combined) latents for one model.

    Returns (tr_lat, tr_meta, eval_lat, eval_meta).
    """
    latents_dir = os.path.join(model_dir, 'latents')

    print(f"\n── {segment.capitalize()} model ({model_dir}) ──")
    tr_lat, tr_meta = _load_split(latents_dir, 'train', aladin_meta)

    va_lat, va_meta = _load_split(latents_dir, 'valid', aladin_meta)
    te_lat, te_meta = _load_split(latents_dir, 'test',  aladin_meta)

    print(f"  Merging valid ({len(va_meta)}) + test ({len(te_meta)}) → eval set")
    eval_lat, eval_meta = _concat_splits(va_lat, va_meta, te_lat, te_meta)
    print(f"  Eval set: {eval_lat[next(iter(eval_lat))].shape[0]} samples")

    return tr_lat, tr_meta, eval_lat, eval_meta


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Combine MedalCare-XL atrial + ventricular latents and run probes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model directories
    parser.add_argument('--atrial_model_dir',      required=True,
                        help='Run directory for the atrial model '
                             '(must contain latents/ sub-dir)')
    parser.add_argument('--ventricular_model_dir', required=True,
                        help='Run directory for the ventricular model '
                             '(must contain latents/ sub-dir)')

    # Data roots (used for path documentation / future inference; not required for
    # loading pre-saved latents)
    parser.add_argument('--atrial_data_root',      default=None,
                        help='Root directory of atrial ECG files (for documentation)')
    parser.add_argument('--ventricular_data_root', default=None,
                        help='Root directory of ventricular ECG files (for documentation)')

    # ALADIN metadata — provides ground-truth labels keyed by UID
    parser.add_argument('--aladin_metadata_dir',   default=None,
                        help='Directory with {train,valid,test}_metadata.json from '
                             'aladin_preprocess.py.  Labels are re-derived from this source.')

    # Probe config
    parser.add_argument('--latent_key', default='z0',
                        help='Latent key to concatenate across models (z0, m, z0_m, …)')
    parser.add_argument('--prefer_labels', type=int, default=1, choices=[1, 2],
                        help='Which model supplies labels when both have the same key')
    parser.add_argument('--balance_sinus', type=eval, default=False,
                        help='Resample sinus class in eval set (recommended for MedalCare-XL)')
    parser.add_argument('--skip_params', nargs='*', default=None,
                        help='Label parameters to skip during probing')

    # MI analysis
    parser.add_argument('--pca_dim', type=int, default=10,
                        help='PCA dimensionality for Gaussian MI estimation')
    parser.add_argument('--n_cca', type=int, default=5,
                        help='Number of CCA canonical components')

    # Output
    parser.add_argument('--output_dir',        required=True,
                        help='Root directory for combined probe outputs')
    parser.add_argument('--model',             required=True,
                        help='Model architecture name (node, vae, hbnode, …)')
    parser.add_argument('--dataset',           default='medalcare-xl')
    parser.add_argument('--summary_output_dir', default=None,
                        help='Directory for summary JSON (for analyse_results.py)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── ALADIN metadata ───────────────────────────────────────────────────────
    aladin_meta: dict | None = None
    if args.aladin_metadata_dir:
        print("\n── Loading ALADIN metadata ──")
        aladin_meta = _load_aladin_metadata(args.aladin_metadata_dir)

    # ── Load latents ──────────────────────────────────────────────────────────
    tr_lat_a, tr_meta_a, ev_lat_a, ev_meta_a = _load_model_splits(
        args.atrial_model_dir, 'atrial', aladin_meta,
    )
    tr_lat_v, tr_meta_v, ev_lat_v, ev_meta_v = _load_model_splits(
        args.ventricular_model_dir, 'ventricular', aladin_meta,
    )

    # ── Match patients and concatenate across models ──────────────────────────
    print(f"\n── Matching train splits (atrial × ventricular) ──")
    tr_combined, tr_meta = _match_and_combine(
        tr_lat_a, tr_meta_a, tr_lat_v, tr_meta_v,
        latent_key=args.latent_key,
        prefer_labels=args.prefer_labels,
    )

    print(f"\n── Matching eval splits (valid+test, atrial × ventricular) ──")
    ev_combined, ev_meta = _match_and_combine(
        ev_lat_a, ev_meta_a, ev_lat_v, ev_meta_v,
        latent_key=args.latent_key,
        prefer_labels=args.prefer_labels,
    )

    d_a = tr_lat_a[args.latent_key].shape[1]
    d_v = tr_lat_v[args.latent_key].shape[1]
    print(f"\n  Combined latent dim : {d_a} (atrial) + {d_v} (ventricular) = {d_a + d_v}")
    print(f"  Train samples       : {tr_combined[args.latent_key].shape[0]}")
    print(f"  Eval  samples       : {ev_combined[args.latent_key].shape[0]}")

    # ── Save combined latents for inspection ─────────────────────────────────
    latents_out_dir = os.path.join(args.output_dir, 'latents')
    os.makedirs(latents_out_dir, exist_ok=True)

    np.savez(os.path.join(latents_out_dir, 'train_latents.npz'), **tr_combined)
    np.savez(os.path.join(latents_out_dir, 'eval_latents.npz'),  **ev_combined)
    with open(os.path.join(latents_out_dir, 'train_metadata.json'), 'w') as f:
        json.dump(tr_meta, f, indent=2)
    with open(os.path.join(latents_out_dir, 'eval_metadata.json'), 'w') as f:
        json.dump(ev_meta, f, indent=2)
    print(f"\n  Saved combined latents → {latents_out_dir}")

    # ── Mutual information between the two latent spaces (train split) ────────
    print(f"\n── Computing mutual information (atrial vs ventricular, train) ──")
    X_comb_tr = tr_combined[args.latent_key]
    X1_tr     = X_comb_tr[:, :d_a]
    X2_tr     = X_comb_tr[:, d_a:]

    mi_results = compute_mutual_information(X1_tr, X2_tr,
                                            pca_dim=args.pca_dim,
                                            n_cca=args.n_cca)
    mi_results.update({
        'model':            args.model,
        'dataset':          args.dataset,
        'latent_key':       args.latent_key,
        'model1_segment':   'atrial',
        'model2_segment':   'ventricular',
    })
    print_mi_summary(mi_results)

    mi_local = os.path.join(args.output_dir, 'mi_analysis.json')
    with open(mi_local, 'w') as f:
        json.dump(mi_results, f, indent=2)
    print(f"  Saved: {mi_local}")

    if args.summary_output_dir:
        os.makedirs(args.summary_output_dir, exist_ok=True)
        slug = args.dataset.lower().replace('-', '_').replace(' ', '_')
        mi_summary_path = os.path.join(args.summary_output_dir, f'mi_{args.model}_{slug}.json')
        with open(mi_summary_path, 'w') as f:
            json.dump(mi_results, f, indent=2)
        print(f"  Saved: {mi_summary_path}")

    # ── Run probes ────────────────────────────────────────────────────────────
    # Output matches what summarize_results._walk_probe_results expects:
    #   {output_dir}/final_finetune_results/combined/{latent_key}/
    probe_out_root = os.path.join(
        args.output_dir, 'final_finetune_results', 'combined', args.latent_key,
    )
    os.makedirs(probe_out_root, exist_ok=True)

    print(f"\n── Running linear probes ──")
    print(f"  Train: {tr_combined[args.latent_key].shape[0]} samples")
    print(f"  Eval : {ev_combined[args.latent_key].shape[0]} samples  (valid + test)")
    run_linear_probes(
        train_latents=tr_combined,
        train_metadata=tr_meta,
        test_latents=ev_combined,
        test_metadata=ev_meta,
        latent_key=args.latent_key,
        out_root=probe_out_root,
        skip_params=args.skip_params,
        balance_sinus=args.balance_sinus,
    )

    # ── Placeholder training_metrics.json ─────────────────────────────────────
    training_stub = {
        'note':             'No ODE training MSE for combined latent model.',
        'model1_segment':   'atrial',
        'model2_segment':   'ventricular',
        'latent_key':       args.latent_key,
        'latent_dim':       d_a + d_v,
        'n_train':          int(tr_combined[args.latent_key].shape[0]),
        'n_eval':           int(ev_combined[args.latent_key].shape[0]),
        'eval_note':        'valid + test combined',
    }
    with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(training_stub, f, indent=2)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    if args.summary_output_dir:
        save_run_summary(
            run_dir=args.output_dir,
            output_dir=args.summary_output_dir,
            model=args.model,
            dataset=args.dataset,
            segment_type='combined',
            original_dir=None,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
