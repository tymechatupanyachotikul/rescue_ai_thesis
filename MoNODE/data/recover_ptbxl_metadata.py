"""Recover PTB-XL metadata JSON + quarantine anomalous-length segments.

Steps
-----
1. Rebuild metadata from saved .pth files + the original split CSV.
2. Per seg_type (median beat only): run an iterative Grubbs' test (two-sided,
   α = 0.05) on segment lengths to identify statistically anomalous samples.
3. Move anomalous .pth files to  {out_dir}/anomaly/{uid}_{seg_type}.pth.
4. Write the cleaned  {split}_metadata.json  and a
   {split}_anomaly_report.json  that documents what was removed.

Usage
-----
    python recover_ptbxl_metadata.py \\
        --out_dir    /path/to/aladin_out/train \\
        --csv_path   /path/to/ptb_xl/data_split/ptb-xl_train.csv \\
        --split      train \\
        --seg_types  atrial ventricular whole \\
        --beat_types median sampled \\
        --fs         500
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from scipy import stats as sp_stats  # type: ignore[import-untyped]
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _to_serialisable(obj):
    if isinstance(obj, (dict, defaultdict)):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _uid_from_path(data_path: str) -> str:
    return os.path.splitext(os.path.basename(str(data_path)))[0]


def _parse_label_str(s) -> list[int]:
    """'1,0,0,0,0' -> [1, 0, 0, 0, 0]"""
    return [int(x) for x in str(s).split(',')]


# ---------------------------------------------------------------------------
# Two-tailed z-test anomaly detection
# ---------------------------------------------------------------------------

def find_anomalies(
    time_list: list[tuple[str, int]],
    seg_type: str,
) -> tuple[list[str], float | None, float | None]:
    """Two-tailed z-test: flag UIDs whose segment length is a statistical
    outlier (p < 0.05).  Cutoff times are mean ± z_crit * std.

    Parameters
    ----------
    time_list : list of (uid, length_in_samples)
    seg_type  : segment type (e.g., 'atrial', 'ventricular')

    Returns
    -------
    anomaly_uids : list of UID strings that are outliers
    lower        : lower cutoff (mean - z_crit * std)
    upper        : upper cutoff (mean + z_crit * std)
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required. Install with: pip install scipy")
    if len(time_list) < 2:
        return [], None, None

    anomaly_uids = []
    if seg_type == 'atrial':
        lower = 20 
        upper = 91 
    elif seg_type == 'ventricular':
        lower = 150 
        upper = 250
    else:
        return [], None, None

    for uid, time in time_list:
        if time < lower or time > upper:
            anomaly_uids.append(uid)
    
    return anomaly_uids, lower, upper


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

_SAMPLED_RE = re.compile(r'^(.+)_(\d+)\.pth$')


def _scan_dir(scan_dir: str, beat_type: str
              ) -> dict[str, list[tuple[int | None, str]]]:
    """Return {base_uid: [(segment_idx_or_None, file_path), ...]}."""
    uid_files: dict[str, list] = defaultdict(list)
    if not os.path.isdir(scan_dir):
        return uid_files

    for fname in os.listdir(scan_dir):
        if not fname.endswith('.pth'):
            continue
        fpath = os.path.join(scan_dir, fname)
        m = _SAMPLED_RE.match(fname)
        if m and beat_type == 'sampled':
            uid_files[m.group(1)].append((int(m.group(2)), fpath))
        else:
            uid_files[fname[:-4]].append((None, fpath))

    return uid_files


# ---------------------------------------------------------------------------
# Metadata recovery
# ---------------------------------------------------------------------------

def recover_metadata(
    out_dir: str,
    seg_types: list[str],
    beat_types: list[str],
    split: str,
) -> dict:
    """Rebuild metadata from .pth files and CSV labels.

    Returns
    -------
    file_index : {(seg_type, beat_type): {uid: [(idx_or_None, fpath)]}}
    """

    # 2. Scan saved .pth files
    file_index: dict = {}

    for seg_type in seg_types:
        for beat_type in beat_types:
            scan_dir  = os.path.join(out_dir, split, seg_type, beat_type)
            uid_files = _scan_dir(scan_dir, beat_type)
            file_index[(seg_type, beat_type)] = dict(uid_files)

            n_files = sum(len(v) for v in uid_files.values())
            print(f"  {scan_dir}: {len(uid_files)} UIDs, {n_files} files")

    return file_index


# ---------------------------------------------------------------------------
# Anomaly detection and quarantine
# ---------------------------------------------------------------------------

def detect_and_quarantine(
    metadata:   dict,
    file_index: dict,
    out_dir:    str,
    seg_types:  list[str],
    split:      str,
    fs:         int = 500,
) -> tuple[dict, dict]:
    """Two-tailed z-test per seg_type (median beat only), move anomalous files,
    update metadata, and return (clean_metadata, report).

    Anomalous files are moved to:
        {out_dir}/anomaly/{uid}_{seg_type}.pth

    Any UID anomalous in ANY seg_type is removed from metadata entirely.
    """
    anomaly_dir = os.path.join(out_dir, 'anomaly')
    os.makedirs(anomaly_dir, exist_ok=True)

    clean_metadata      = {uid: dict(entry) for uid, entry in metadata.items()}
    all_anomalous_uids: set[str] = set()

    report: dict = {
        'split':       split,
        'alpha':       0.05,
        'fs_hz':       fs,
        'anomaly_dir': anomaly_dir,
        'results':     {},
    }

    for seg_type in seg_types:
        uid_map = file_index.get((seg_type, 'median'), {})
        if not uid_map:
            print(f"  [{seg_type}/median] no files found — skipping")
            continue

        # Build (uid, length, src_file_path) for every median segment
        records: list[tuple[str, float, str]] = []
        for uid, entries in uid_map.items():
            sl = metadata.get(uid, {}).get('segment_lengths', {}).get(seg_type)
            if sl is None:
                continue
            # Median always stores an int, but guard for unexpected list
            length = float(np.mean(sl) if isinstance(sl, list) else sl)
            # entries = [(None, fpath)] for median
            fpath = entries[0][1]
            records.append((uid, length, fpath))

        if len(records) < 2:
            print(f"  [{seg_type}/median] <2 samples — skipping z-test")
            continue

        # ── Two-tailed z-test ─────────────────────────────────────────────────
        time_list     = [(uid, int(length)) for uid, length, _ in records]
        anomaly_uids, lower, upper = find_anomalies(time_list, seg_type)
        anomaly_set   = set(anomaly_uids)

        uid_to_record = {uid: (length, fpath) for uid, length, fpath in records}
        kept_lengths  = np.array([r[1] for r in records if r[0] not in anomaly_set],
                                 dtype=float)
        n_kept        = len(kept_lengths)

        final_mean = float(kept_lengths.mean()) if n_kept else float('nan')
        final_std  = float(kept_lengths.std())  if n_kept else float('nan')

        print(f"  [{seg_type}/median] "
              f"n={len(records)}  outliers={len(anomaly_uids)}  "
              f"kept=[{kept_lengths.min():.0f}, {kept_lengths.max():.0f}]  "
              f"cutoff=[{lower:.1f}, {upper:.1f}]")

        # ── Move anomalous files → {out_dir}/anomaly/{uid}_{seg_type}.pth ───
        removed_entries = []
        for uid in anomaly_uids:
            all_anomalous_uids.add(uid)
            length, src = uid_to_record[uid]

            dst = os.path.join(anomaly_dir, f'{uid}_{seg_type}.pth')
            if os.path.exists(src):
                shutil.move(src, dst)
            else:
                dst = None   # file was already missing

            removed_entries.append({
                'uid':            uid,
                'length_samples': int(length),
                'length_sec':     round(length / fs, 4),
                'src':            src,
                'dst':            dst,
            })

        # ── Section report ───────────────────────────────────────────────────
        report['results'][seg_type] = {
            'n_total':   len(records),
            'n_removed': len(anomaly_uids),
            'n_kept':    n_kept,
            'cutoff_samples': {
                'lower': round(lower, 2) if lower is not None else None,
                'upper': round(upper, 2) if upper is not None else None,
            },
            'cutoff_sec': {
                'lower': round(lower / fs, 4) if lower is not None else None,
                'upper': round(upper / fs, 4) if upper is not None else None,
            },
            'kept_length_stats': {
                'mean':   round(final_mean, 2),
                'std':    round(final_std, 2),
                'median': round(float(np.median(kept_lengths)), 2),
                'min':    int(kept_lengths.min()) if n_kept else None,
                'max':    int(kept_lengths.max()) if n_kept else None,
                'p5':     round(float(np.percentile(kept_lengths,  5)), 2) if n_kept else None,
                'p95':    round(float(np.percentile(kept_lengths, 95)), 2) if n_kept else None,
            },
            'removed': removed_entries,
        }

    # ── Strip anomalous UIDs from metadata ───────────────────────────────────
    print(f"\nTotal UIDs quarantined: {len(all_anomalous_uids)}")
    for uid in all_anomalous_uids:
        clean_metadata.pop(uid, None)

    report['total_uids_quarantined'] = len(all_anomalous_uids)
    report['quarantined_uids']       = sorted(all_anomalous_uids)

    return clean_metadata, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recover PTB-XL metadata and quarantine anomalous segments")
    parser.add_argument('--out_dir',        required=True,
                        help="Root output dir from aladin_preprocess (e.g. .../train)")
    parser.add_argument('--metadata_path',       required=True,
                        help="Split CSV (e.g. ptb-xl_train.csv)")
    parser.add_argument('--split',          default='train',
                        help="Split name for output filenames (default: train)")
    parser.add_argument('--seg_types',  nargs='+',
                        default=['atrial', 'ventricular', 'whole'],
                        help="Segment types to process")
    parser.add_argument('--beat_types', nargs='+',
                        default=['median', 'sampled'],
                        help="Beat types to process")
    parser.add_argument('--fs',    type=int, default=500,
                        help="ECG sampling frequency in Hz (default: 500)")
    parser.add_argument('--skip_anomaly_detection', action='store_true',
                        help="Only recover metadata; skip z-test quarantine step")
    args = parser.parse_args()

    print(f"Recovering metadata from: {args.out_dir}")
    file_index = recover_metadata(
        out_dir    = args.out_dir,
        seg_types  = args.seg_types,
        beat_types = args.beat_types,
        split = args.split
    )
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"  Recovered {len(metadata)} UIDs")

    if not args.skip_anomaly_detection:
        print("\nRunning z-test anomaly detection (two-tailed α=0.05)...")
        clean_metadata, report = detect_and_quarantine(
            metadata   = metadata,
            file_index = file_index,
            out_dir    = args.out_dir,
            seg_types  = args.seg_types,
            split      = args.split,
            fs         = args.fs,
        )

        report_path = os.path.join(args.out_dir, 'errors', f'{args.split}_anomaly_report.json')
        with open(report_path, 'w') as f:
            json.dump(_to_serialisable(report), f, indent=2)
        print(f"Anomaly report → {report_path}")

if __name__ == '__main__':
    main()
