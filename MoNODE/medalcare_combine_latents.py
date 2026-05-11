#!/usr/bin/env python3
"""
medalcare_combine_latents.py — Combine atrial + ventricular latents for MedalCare-XL
and run classification / regression probes.

Workflow
--------
1. Load model checkpoint from {model_dir}/model.pth; reconstruct architecture
   from the saved {model_dir}/args.json.
2. Build ECGDataset loaders by scanning the ALADIN preprocessed output directory
   (structure: {data_root}/{split}/{seg_type}/{beat_type}/*.pth).
3. Run inference to collect z0 (and m) latents for train / valid / test.
   Results are cached in {model_dir}/latents/ — re-running skips inference.
4. Override labels from ALADIN metadata JSON files.
5. Merge valid + test splits into a single evaluation set.
6. Match patients across atrial and ventricular models, then concatenate their
   latent vectors (identical logic to combine_latents.py).
7. Run OLS probes (classification + regression).
8. Write output in the same directory layout and summary-JSON format as
   summarize_results.py so it can be consumed by analyse_results.py.

Usage
-----
  python medalcare_combine_latents.py \\
      --atrial_model_dir      results/ecg/node/atrial_run \\
      --ventricular_model_dir results/ecg/node/ventricular_run \\
      --atrial_data_root      /projects/data/aladin_atrial \\
      --ventricular_data_root /projects/data/aladin_ventricular \\
      --aladin_metadata_dir   /projects/data/aladin_atrial \\
      --output_dir            results/combined/medalcare_node \\
      --model                 node \\
      --latent_key            z0 \\
      --summary_output_dir    results/summaries
"""

import argparse
import glob as _glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from finetune import (
    run_linear_probes, collect_latents,
    run_trajectory_analysis, _medalcare_uid_from_stem,
)
from combine_latents import (
    _match_and_combine_all, _group_by_patient_all,
    compute_mutual_information, print_mi_summary,
)
from summarize_results import save_run_summary
from data.data_utils import ECGDataset, pad_collate
from model.build_model import build_model, build_simclr_model, build_byol_model


# ─── ALADIN metadata helpers ──────────────────────────────────────────────────

def _load_aladin_metadata(
    metadata_dir: str,
    splits: tuple[str, ...] = ('train', 'valid', 'test'),
) -> dict:
    """Load and merge ALADIN metadata JSONs into a single uid-keyed dict."""
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
        n = len(raw) if isinstance(raw, dict) else len(raw)
        print(f"  ALADIN [{split}]: {n} entries")
    print(f"  Total ALADIN entries: {len(merged)}")
    return merged


# ─── data helpers ─────────────────────────────────────────────────────────────

def _infer_inp_dim(data_root: str, split: str, seg_type: str, beat_type: str) -> int:
    """Read one .pth file to determine the number of input leads."""
    split_dir = os.path.join(data_root, split, seg_type, beat_type)
    files = _glob.glob(os.path.join(split_dir, '*.pth'))
    if not files:
        raise FileNotFoundError(f"No .pth files found in {split_dir}")
    sample = torch.load(files[0], map_location='cpu', weights_only=False)
    return int(sample.shape[-1])  # [T, D] → D


def _build_loader(
    data_root: str,
    split: str,
    seg_type: str,
    beat_type: str,
    batch_size: int,
    dtype: torch.dtype,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """Scan an ALADIN output directory and build an ECGDataset DataLoader.

    Expected directory structure:
        {data_root}/{split}/{seg_type}/{beat_type}/{uid}.pth

    For MedalCare-XL the uid is {run_id}_{session_id}_{class}.
    """
    split_dir = os.path.join(data_root, split, seg_type, beat_type)
    file_paths = sorted(_glob.glob(os.path.join(split_dir, '*.pth')))
    print(file_paths[:10])
    if not file_paths:
        raise FileNotFoundError(f"No .pth files in {split_dir}")

    labels:  list[str] = []
    run_ids: list[str] = []
    for fp in file_paths:
        stem  = os.path.splitext(os.path.basename(fp))[0]
        parts = stem.split('_')
        run_ids.append(parts[0])
        labels.append('_'.join(parts[2:]) if len(parts) > 2 else 'unknown')

    ds = ECGDataset(
        file_paths=file_paths,
        labels=labels,
        run_id=run_ids,
        dtype=dtype,
        dataset='medalcare-xl',
        return_file_path=True,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=min(batch_size, len(file_paths)),
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        collate_fn=pad_collate,
        pin_memory=False,
    )


# ─── model helpers ────────────────────────────────────────────────────────────

def _build_and_load_model(
    model_dir: str,
    inp_dim: int,
    out_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, argparse.Namespace]:
    """Reconstruct model from {model_dir}/args.json and load {model_dir}/model.pth."""
    args_path = os.path.join(model_dir, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {model_dir}")

    with open(args_path) as f:
        saved = json.load(f)

    ns = argparse.Namespace(**saved)

    is_simclr = getattr(ns, 'simclr_pretrain', False)
    is_byol   = getattr(ns, 'byol_pretrain',   False)

    if is_simclr or is_byol:
        # SSL encoder-only models — safe defaults for fields that may be absent
        for attr, default in [
            ('proj_dim', 64), ('enc_H', 50), ('order', 1),
        ]:
            if not hasattr(ns, attr):
                setattr(ns, attr, default)
        if is_simclr:
            model = build_simclr_model(ns, device, dtype, inp_dim)
        else:
            model = build_byol_model(ns, device, dtype, inp_dim)
    else:
        # NODE / HBNODE / VAE / MoNODE
        for attr, default in [
            ('Nobj', 1), ('sobolev_weight', 0), ('l_w', 0),
            ('rnn_hidden_dec', None), ('content_dim', 0),
        ]:
            if not hasattr(ns, attr):
                setattr(ns, attr, default)
        config = {
            'inp_dim': inp_dim,
            'out_dim': out_dim,
            'w_dt':    ns.sobolev_weight,
            'l_w':     ns.l_w,
        }
        model = build_model(ns, device, dtype, **config)

    model.to(device).to(dtype)

    ckpt_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model, ns


# ─── inference + caching ──────────────────────────────────────────────────────

def _collect_and_cache(
    model_dir: str,
    model: torch.nn.Module,
    loaders: dict[str, torch.utils.data.DataLoader],
    model_args: argparse.Namespace,
    device: torch.device,
    aladin_meta: dict | None,
    seg_type: str,
) -> dict[str, tuple[dict, list]]:
    """Run inference for each split; cache to {model_dir}/latents/.

    Returns {split: (latents_dict, metadata_list)}.
    """
    latents_dir = os.path.join(model_dir, 'latents')
    os.makedirs(latents_dir, exist_ok=True)

    task_params = {'dataset': 'medalcare-xl', 'beat_type': seg_type}

    def _cached(split: str) -> bool:
        return (
            os.path.exists(os.path.join(latents_dir, f'{split}_latents.npz')) and
            os.path.exists(os.path.join(latents_dir, f'{split}_metadata.json'))
        )

    def _load_cached(split: str) -> tuple[dict, list]:
        lat = dict(np.load(os.path.join(latents_dir, f'{split}_latents.npz')))
        with open(os.path.join(latents_dir, f'{split}_metadata.json')) as f:
            meta = json.load(f)
        return lat, meta

    result: dict[str, tuple[dict, list]] = {}
    for split, loader in loaders.items():
        if False:
            print(f"  [{split}] Cached — loading from disk.")
            result[split] = _load_cached(split)
        else:
            print(f"  [{split}] Running inference ({len(loader.dataset)} samples) …")
            with torch.no_grad(), np.errstate(all='ignore'):
                lat, meta = collect_latents(
                    loader, model, task_params, model_args, device,
                    aladin_metadata=aladin_meta,
                )
            np.savez(os.path.join(latents_dir, f'{split}_latents.npz'), **lat)
            with open(os.path.join(latents_dir, f'{split}_metadata.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"    Saved → {latents_dir}/{split}_latents.npz  ({lat['z0'].shape[0]} samples)")
            result[split] = (lat, meta)

    return result


# ─── zTL collection + caching ────────────────────────────────────────────────

_SEG_T_CUTOFF = {'atrial': 45, 'ventricular': 180}


def _collect_and_cache_zTL(
    model_dir: str,
    model: torch.nn.Module,
    loaders: dict,
    device: torch.device,
) -> dict[str, tuple[np.ndarray, list]]:
    """Run inference for each split and collect per-patient mean zTL trajectories.

    Caches results to {model_dir}/latents/{split}_traj.npz + {split}_traj_uids.json.
    Returns {split: (zTL [N_patients, T, q], uid_list [N_patients])}.
    """
    from collections import defaultdict
    from tqdm import tqdm

    latents_dir = os.path.join(model_dir, 'latents')
    os.makedirs(latents_dir, exist_ok=True)

    model.eval()
    model.return_latent = True

    results: dict = {}
    for split, loader in loaders.items():
        traj_npz  = os.path.join(latents_dir, f'{split}_traj.npz')
        uids_json = os.path.join(latents_dir, f'{split}_traj_uids.json')

        if os.path.exists(traj_npz) and os.path.exists(uids_json):
            print(f"  [{split}] zTL cached — loading from disk.")
            arr = np.load(traj_npz)
            with open(uids_json) as f:
                uids = json.load(f)
            results[split] = (arr['zTL'], uids)
            continue

        print(f"  [{split}] Collecting zTL ({len(loader.dataset)} samples) …")
        pid_zTL: dict = defaultdict(list)

        with torch.no_grad():
            for batch, batch_y, mask in tqdm(loader, desc=f"zTL [{split}]"):
                batch = batch.to(device)
                mask  = mask.to(device)

                _z0, _z0s, _m, ztL = model(batch, 1, mask=mask)
                del _z0, _z0s, _m
                ztL_cpu = ztL.mean(0).detach().cpu().numpy()   # [N, T, q]
                del ztL

                for i in range(batch.shape[0]):
                    file_path = batch_y[i][2]
                    stem      = os.path.splitext(os.path.basename(file_path))[0]
                    uid       = _medalcare_uid_from_stem(stem)
                    pid_zTL[uid].append(ztL_cpu[i])

        # Per-patient mean trajectory; pad variable T with zeros
        sorted_uids = sorted(pid_zTL.keys())
        per_pid: list = [np.mean(pid_zTL[uid], axis=0) for uid in sorted_uids]
        max_T  = max(a.shape[0] for a in per_pid)
        q      = per_pid[0].shape[-1]
        padded = np.zeros((len(per_pid), max_T, q), dtype=per_pid[0].dtype)
        for idx, arr in enumerate(per_pid):
            padded[idx, :arr.shape[0], :] = arr

        np.savez(traj_npz, zTL=padded)
        with open(uids_json, 'w') as f:
            json.dump(sorted_uids, f)
        print(f"    Saved → {traj_npz}  (patients={len(sorted_uids)}, T={max_T}, q={q})")
        results[split] = (padded, sorted_uids)

    model.return_latent = False
    return results


# ─── combined trajectory analysis ────────────────────────────────────────────

def run_trajectory_analysis_combined(
    traj_a: dict,
    traj_v: dict,
    matched_uids: list,
    eval_meta: list,
    out_root: str,
    dataset: str,
) -> None:
    """Concatenate atrial + ventricular zTL for matched patients and run analysis.

    traj_a / traj_v : {split: (zTL [N_pat, T, q], uid_list)} as returned by
                      _collect_and_cache_zTL.  Uses the 'eval' split (valid+test).
    matched_uids    : ordered list of patient UIDs present in ev_combined.
    eval_meta       : combined eval metadata list (same order as matched_uids).
    """
    # Retrieve eval trajectories and uid→index maps
    zTL_a_full, uids_a = traj_a.get('eval', (None, None))
    zTL_v_full, uids_v = traj_v.get('eval', (None, None))

    if zTL_a_full is None or zTL_v_full is None:
        print("  [trajectory] zTL not available for one/both models — skipping.")
        return

    idx_a = {uid: i for i, uid in enumerate(uids_a)}
    idx_v = {uid: i for i, uid in enumerate(uids_v)}

    # Keep only matched uids that exist in BOTH trajectory sets
    valid = [uid for uid in matched_uids if uid in idx_a and uid in idx_v]
    if not valid:
        print("  [trajectory] No matched uids found in both zTL sets — skipping.")
        return
    print(f"\n── Trajectory analysis: {len(valid)} matched patients ──")

    # Filter metadata to valid patients and collapse MI subclasses → 'mi'
    uid_to_meta = {m['uid']: m for m in eval_meta}
    filtered_meta = []
    for uid in valid:
        if uid not in uid_to_meta:
            continue
        entry  = dict(uid_to_meta[uid])
        labels = dict(entry.get('labels', {}))
        cls    = labels.get('class', '')
        if isinstance(cls, str) and cls.startswith(('LAD_', 'LCX_', 'RCA_')):
            labels['class'] = 'mi'
        entry['labels'] = labels
        filtered_meta.append(entry)

    # Build per-patient arrays from matched set
    rows_a = np.stack([zTL_a_full[idx_a[uid]] for uid in valid])   # [N, T_a, q_a]
    rows_v = np.stack([zTL_v_full[idx_v[uid]] for uid in valid])   # [N, T_v, q_v]

    # Trim both to the minimum usable T (atrial is the bottleneck at 45 steps)
    T_min = min(
        _SEG_T_CUTOFF['atrial'],
        rows_a.shape[1],
        rows_v.shape[1],
    )
    rows_a = rows_a[:, :T_min, :]
    rows_v = rows_v[:, :T_min, :]

    # Concatenate along latent dimension → [N, T_min, q_a + q_v]
    zTL_combined = np.concatenate([rows_a, rows_v], axis=2)
    print(f"  Combined zTL shape: {zTL_combined.shape}  "
          f"(T={T_min}, q_a={rows_a.shape[2]}, q_v={rows_v.shape[2]})")

    run_trajectory_analysis(
        eval_latents={'zTL': zTL_combined},
        eval_metadata=filtered_meta,
        dataset_name=dataset,
        out_root=out_root,
        seg_type=None,          # pre-trimmed; no further cutoff needed
    )


# ─── split concat helper ──────────────────────────────────────────────────────

def _concat_splits(
    lat_a: dict, meta_a: list,
    lat_b: dict, meta_b: list,
) -> tuple[dict, list]:
    common = set(lat_a) & set(lat_b)
    return (
        {k: np.concatenate([lat_a[k], lat_b[k]], axis=0) for k in common},
        meta_a + meta_b,
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Combine MedalCare-XL atrial + ventricular latents and run probes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model directories ─────────────────────────────────────────────────────
    parser.add_argument('--atrial_model_dir',      required=True,
                        help='Run directory for the atrial model (contains model.pth + args.json)')
    parser.add_argument('--ventricular_model_dir', required=True,
                        help='Run directory for the ventricular model')

    # ── Data roots (ALADIN preprocessed output) ───────────────────────────────
    parser.add_argument('--atrial_data_root',      required=True,
                        help='Root of ALADIN-preprocessed atrial data. '
                             'Expected layout: {root}/{split}/atrial/{beat_type}/*.pth')
    parser.add_argument('--ventricular_data_root', required=True,
                        help='Root of ALADIN-preprocessed ventricular data. '
                             'Expected layout: {root}/{split}/ventricular/{beat_type}/*.pth')

    # ── ALADIN metadata (labels) ──────────────────────────────────────────────
    parser.add_argument('--aladin_metadata_dir',   default=None,
                        help='Directory with {train,valid,test}_metadata.json from '
                             'aladin_preprocess.py. Labels are re-derived from this source.')

    # ── Data / inference config ───────────────────────────────────────────────
    parser.add_argument('--beat_type',  default='median', choices=['median', 'sampled'],
                        help='Beat type sub-directory in the data root')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dim',    type=int, default=None,
                        help='Decoder output leads. Defaults to inp_dim (no lead exclusion). '
                             'Set to 8 if 4 leads were excluded from output during training.')

    # ── Probe config ──────────────────────────────────────────────────────────
    parser.add_argument('--latent_key',    default='z0',
                        help='Latent key to concatenate (z0, m, z0_m)')
    parser.add_argument('--prefer_labels', type=int, default=1, choices=[1, 2])
    parser.add_argument('--balance_sinus', type=eval, default=False,
                        help='Resample sinus class in eval set')
    parser.add_argument('--skip_params',   nargs='*', default=None,
                        help='Label parameters to skip during probing')

    # ── MI analysis ───────────────────────────────────────────────────────────
    parser.add_argument('--pca_dim', type=int, default=10)
    parser.add_argument('--n_cca',   type=int, default=5)

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument('--output_dir',        required=True)
    parser.add_argument('--model',             required=True,
                        help='Model architecture name (node, vae, hbnode, …)')
    parser.add_argument('--dataset',           default='medalcare-xl')
    parser.add_argument('--summary_output_dir', default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float64
    print(f"Device: {device}")

    # ── ALADIN metadata ───────────────────────────────────────────────────────
    aladin_meta: dict | None = None
    if args.aladin_metadata_dir:
        print("\n── Loading ALADIN metadata ──")
        aladin_meta = _load_aladin_metadata(args.aladin_metadata_dir)

    # ── Infer input dims from data ────────────────────────────────────────────
    inp_dim_a = 12
    inp_dim_v = 12
    out_dim_a = 12
    out_dim_v = 12
    print(f"\n  Atrial  inp_dim={inp_dim_a}  out_dim={out_dim_a}")
    print(f"  Ventricular inp_dim={inp_dim_v}  out_dim={out_dim_v}")

    # ── Build models and load checkpoints ─────────────────────────────────────
    print("\n── Building atrial model ──")
    model_a, margs_a = _build_and_load_model(
        args.atrial_model_dir, inp_dim_a, out_dim_a, device, dtype,
    )

    print("\n── Building ventricular model ──")
    model_v, margs_v = _build_and_load_model(
        args.ventricular_model_dir, inp_dim_v, out_dim_v, device, dtype,
    )

    # ── Build data loaders ────────────────────────────────────────────────────
    print("\n── Building data loaders ──")
    loaders_a: dict[str, torch.utils.data.DataLoader] = {}
    loaders_v: dict[str, torch.utils.data.DataLoader] = {}
    for split in ('train', 'valid', 'test'):
        loaders_a[split] = _build_loader(
            args.atrial_data_root, split, 'atrial',
            args.beat_type, args.batch_size, dtype,
        )
        loaders_v[split] = _build_loader(
            args.ventricular_data_root, split, 'ventricular',
            args.beat_type, args.batch_size, dtype,
        )
        print(f"  [{split}] atrial={len(loaders_a[split].dataset)}  "
              f"ventricular={len(loaders_v[split].dataset)}")

    # ── Collect latents (with per-model caching) ──────────────────────────────
    print("\n── Collecting atrial latents ──")
    splits_a = _collect_and_cache(
        args.atrial_model_dir, model_a, loaders_a, margs_a, device, aladin_meta, 'atrial',
    )

    print("\n── Collecting ventricular latents ──")
    splits_v = _collect_and_cache(
        args.ventricular_model_dir, model_v, loaders_v, margs_v, device, aladin_meta, 'ventricular',
    )

    # ── Collect zTL trajectories (before freeing models) ─────────────────────
    print("\n── Collecting atrial zTL trajectories ──")
    traj_a = _collect_and_cache_zTL(
        args.atrial_model_dir, model_a, loaders_a, device,
    )
    print("\n── Collecting ventricular zTL trajectories ──")
    traj_v = _collect_and_cache_zTL(
        args.ventricular_model_dir, model_v, loaders_v, device,
    )

    # Merge valid+test for trajectory eval split
    zTL_ev_a, uids_ev_a = traj_a.get('valid', (np.empty((0,)), []))
    zTL_te_a, uids_te_a = traj_a.get('test',  (np.empty((0,)), []))
    zTL_ev_v, uids_ev_v = traj_v.get('valid', (np.empty((0,)), []))
    zTL_te_v, uids_te_v = traj_v.get('test',  (np.empty((0,)), []))

    def _merge_traj(z1, u1, z2, u2):
        if z1.ndim < 3 or z2.ndim < 3:
            return z1 if z2.ndim < 3 else z2, u1 + u2
        q = z1.shape[2]
        T = max(z1.shape[1], z2.shape[1])
        merged = np.zeros((len(u1) + len(u2), T, q), dtype=z1.dtype)
        merged[:len(u1), :z1.shape[1], :] = z1
        merged[len(u1):, :z2.shape[1], :] = z2
        return merged, u1 + u2

    zTL_eval_a, uids_eval_a = _merge_traj(zTL_ev_a, uids_ev_a, zTL_te_a, uids_te_a)
    zTL_eval_v, uids_eval_v = _merge_traj(zTL_ev_v, uids_ev_v, zTL_te_v, uids_te_v)
    traj_a['eval'] = (zTL_eval_a, uids_eval_a)
    traj_v['eval'] = (zTL_eval_v, uids_eval_v)

    # Free GPU memory before running probes
    del model_a, model_v
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Merge valid + test → single eval set per model ────────────────────────
    tr_lat_a, tr_meta_a = splits_a['train']
    tr_lat_v, tr_meta_v = splits_v['train']

    ev_lat_a, ev_meta_a = _concat_splits(
        splits_a['valid'][0], splits_a['valid'][1],
        splits_a['test'][0],  splits_a['test'][1],
    )
    ev_lat_v, ev_meta_v = _concat_splits(
        splits_v['valid'][0], splits_v['valid'][1],
        splits_v['test'][0],  splits_v['test'][1],
    )
    print(f"\n  Eval set: atrial={len(ev_meta_a)}  ventricular={len(ev_meta_v)}")

    # ── Match patients and concatenate ALL latent keys ────────────────────────
    print(f"\n── Matching train splits (atrial × ventricular) ──")
    tr_combined, tr_meta = _match_and_combine_all(
        tr_lat_a, tr_meta_a, tr_lat_v, tr_meta_v,
        prefer_labels=args.prefer_labels,
    )

    print(f"\n── Matching eval splits (valid+test, atrial × ventricular) ──")
    ev_combined, ev_meta = _match_and_combine_all(
        ev_lat_a, ev_meta_a, ev_lat_v, ev_meta_v,
        prefer_labels=args.prefer_labels,
    )

    combined_keys = sorted(tr_combined.keys())
    n_train = next(iter(tr_combined.values())).shape[0]
    n_eval  = next(iter(ev_combined.values())).shape[0]
    print(f"\n  Combined latent keys : {combined_keys}")
    for key in combined_keys:
        print(f"    {key:8s}: dim={tr_combined[key].shape[1]}")
    print(f"  Train samples        : {n_train}")
    print(f"  Eval  samples        : {n_eval}")

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

    # ── Mutual information (z0 key, train set) ────────────────────────────────
    mi_key = args.latent_key if args.latent_key in tr_combined else combined_keys[0]
    d_a = tr_lat_a['z0'].shape[1] if 'z0' in tr_lat_a else tr_lat_a[mi_key].shape[1]
    print(f"\n── Computing mutual information (atrial vs ventricular, key={mi_key}) ──")
    X_comb_tr = tr_combined[mi_key]
    mi_results = compute_mutual_information(
        X_comb_tr[:, :d_a], X_comb_tr[:, d_a:],
        pca_dim=args.pca_dim, n_cca=args.n_cca,
    )
    mi_results.update({
        'model': args.model, 'dataset': args.dataset,
        'latent_key': mi_key,
        'model1_segment': 'atrial', 'model2_segment': 'ventricular',
    })
    print_mi_summary(mi_results)
    mi_local = os.path.join(args.output_dir, 'mi_analysis.json')
    with open(mi_local, 'w') as f:
        json.dump(mi_results, f, indent=2)

    if args.summary_output_dir:
        os.makedirs(args.summary_output_dir, exist_ok=True)
        slug = args.dataset.lower().replace('-', '_').replace(' ', '_')
        with open(os.path.join(args.summary_output_dir, f'mi_{args.model}_{slug}.json'), 'w') as f:
            json.dump(mi_results, f, indent=2)

    # ── Run probes for each combined key ──────────────────────────────────────
    probe_out_parent = os.path.join(args.output_dir, 'final_finetune_results', 'combined')
    _skip = set(args.skip_params or []) | {'patient_id'}
    for key in combined_keys:
        probe_out_root = os.path.join(probe_out_parent, key)
        os.makedirs(probe_out_root, exist_ok=True)
        print(f"\n── Running linear probes (OLS)  [{key}] ──")
        print(f"  Train: {n_train}  Eval: {n_eval}")
        run_linear_probes(
            train_latents=tr_combined,
            train_metadata=tr_meta,
            test_latents=ev_combined,
            test_metadata=ev_meta,
            latent_key=key,
            out_root=probe_out_root,
            skip_params=_skip,
            methods={'ols'},
            balance_sinus=args.balance_sinus,
        )

    # ── Latent trajectory analysis (combined atrial + ventricular zTL) ────────
    print(f"\n── Running combined trajectory analysis ──")
    matched_uids = [m['uid'] for m in ev_meta]
    run_trajectory_analysis_combined(
        traj_a=traj_a,
        traj_v=traj_v,
        matched_uids=matched_uids,
        eval_meta=ev_meta,
        out_root=os.path.join(args.output_dir, 'final_finetune_results', 'combined'),
        dataset=args.dataset,
    )

    # ── training_metrics.json placeholder ─────────────────────────────────────
    with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
        json.dump({
            'note':           'No ODE training MSE for combined latent model.',
            'model1_segment': 'atrial',
            'model2_segment': 'ventricular',
            'combined_keys':  combined_keys,
            'latent_dims':    {k: int(tr_combined[k].shape[1]) for k in combined_keys},
            'n_train':        int(n_train),
            'n_eval':         int(n_eval),
            'eval_note':      'valid + test combined',
        }, f, indent=2)

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
