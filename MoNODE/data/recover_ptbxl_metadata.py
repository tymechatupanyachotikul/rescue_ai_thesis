"""Recover metadata JSON for a PTB-XL aladin_preprocess.py run that crashed
during json.dump (ndarray not serialisable).

The preprocessing saved .pth segment files successfully; this script
reconstructs the metadata_dict from those files + the original split CSV.

Usage:
    python recover_ptbxl_metadata.py \
        --out_dir   /path/to/aladin_out/train \
        --csv_path  /path/to/ptb_xl/data_split/ptb-xl_train.csv \
        --split     train \
        --seg_types atrial ventricular whole \
        --beat_types median sampled
"""

import argparse
import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Helpers
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


def _uid_from_path(data_path: str) -> str:
    return os.path.splitext(os.path.basename(str(data_path)))[0]


def _parse_label_str(s):
    """'1,0,0,0,0' -> [1, 0, 0, 0, 0]"""
    return [int(x) for x in str(s).split(',')]


# ---------------------------------------------------------------------------
# Main recovery logic
# ---------------------------------------------------------------------------

def recover_metadata(out_dir: str, csv_path: str, split: str,
                     seg_types: list[str], beat_types: list[str]) -> dict:
    # 1. Load CSV and build uid → labels mapping
    df = pd.read_csv(csv_path)
    uid_to_labels: dict[str, dict] = {}
    for _, row in df.iterrows():
        uid = _uid_from_path(row.data_path)
        labels: dict = {'patient_id': uid}
        for col in ['superclass', 'subclass', 'form', 'rhythm']:
            if col in df.columns and pd.notna(row[col]):
                labels[col] = _parse_label_str(row[col])
            else:
                labels[col] = None
        uid_to_labels[uid] = labels

    print(f"CSV loaded: {len(uid_to_labels)} unique UIDs")

    # 2. Scan .pth files and collect segment info per base_uid
    #    Structure: {out_dir}/{seg_type}/{beat_type}/{uid}.pth   (median)
    #               {out_dir}/{seg_type}/{beat_type}/{uid}_N.pth (sampled)
    #
    # metadata key = base_uid
    # segment_lengths[seg_type] = int (median) or list[int] (sampled)

    metadata: dict[str, dict] = {}

    sampled_re = re.compile(r'^(.+)_(\d+)\.pth$')

    for seg_type in seg_types:
        for beat_type in beat_types:
            scan_dir = os.path.join(out_dir, seg_type, beat_type)
            if not os.path.isdir(scan_dir):
                print(f"  [skip] {scan_dir} not found")
                continue

            # Collect files, grouping sampled indices by base_uid
            uid_files: dict[str, list[tuple[int | None, str]]] = defaultdict(list)
            for fname in os.listdir(scan_dir):
                if not fname.endswith('.pth'):
                    continue
                fpath = os.path.join(scan_dir, fname)
                m = sampled_re.match(fname)
                if m and beat_type == 'sampled':
                    base_uid, idx = m.group(1), int(m.group(2))
                    uid_files[base_uid].append((idx, fpath))
                else:
                    base_uid = fname[:-4]  # strip .pth
                    uid_files[base_uid].append((None, fpath))

            n_files = sum(len(v) for v in uid_files.values())
            print(f"  {scan_dir}: {len(uid_files)} base UIDs, {n_files} files")

            for base_uid, entries in uid_files.items():
                if base_uid not in metadata:
                    metadata[base_uid] = {
                        'labels':           uid_to_labels.get(base_uid, {'patient_id': base_uid}),
                        'p_wave_estimated': None,  # not recoverable
                        'segment_lengths':  {},
                    }

                if beat_type == 'sampled':
                    # Sort by index, load each tensor to get length
                    entries_sorted = sorted(entries, key=lambda x: x[0] if x[0] is not None else 0)
                    lengths = []
                    for _, fpath in entries_sorted:
                        try:
                            t = torch.load(fpath, map_location='cpu')
                            lengths.append(int(t.shape[0]))
                        except Exception as e:
                            print(f"    [warn] could not load {fpath}: {e}")
                    metadata[base_uid]['segment_lengths'][seg_type] = lengths
                else:
                    # Median: single file, single int length
                    _, fpath = entries[0]
                    try:
                        t = torch.load(fpath, map_location='cpu')
                        metadata[base_uid]['segment_lengths'][seg_type] = int(t.shape[0])
                    except Exception as e:
                        print(f"    [warn] could not load {fpath}: {e}")
                        metadata[base_uid]['segment_lengths'][seg_type] = None

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Recover PTB-XL metadata JSON")
    parser.add_argument('--out_dir',    required=True,
                        help="Root output dir used in aladin_preprocess (e.g. .../train)")
    parser.add_argument('--csv_path',   required=True,
                        help="Split CSV (e.g. ptb-xl_train.csv)")
    parser.add_argument('--split',      default='train',
                        help="Split name used in the output filename (default: train)")
    parser.add_argument('--seg_types',  nargs='+',
                        default=['atrial', 'ventricular', 'whole'],
                        help="Segment types to scan")
    parser.add_argument('--beat_types', nargs='+',
                        default=['median', 'sampled'],
                        help="Beat types to scan")
    args = parser.parse_args()

    print(f"Recovering metadata from: {args.out_dir}")
    metadata = recover_metadata(
        out_dir    = args.out_dir,
        csv_path   = args.csv_path,
        split      = args.split,
        seg_types  = args.seg_types,
        beat_types = args.beat_types,
    )

    out_path = os.path.join(args.out_dir, f'{args.split}_metadata.json')
    with open(out_path, 'w') as f:
        json.dump(_to_serialisable(metadata), f, indent=2)

    print(f"\nSaved {len(metadata)} entries → {out_path}")


if __name__ == '__main__':
    main()
