import json
import random
import pandas as pd
import os
import gc
import wfdb
from tqdm import tqdm
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from aladin import ALADIN
from aladin.core import Record

import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import butter, filtfilt, medfilt

FS = 500  # All ECG records are written/read at 500 Hz

MEDALCARE_XL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
UK_BB_LEADS        = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MIMIC_IV_LEADS     = ['I', 'II', 'III', 'aVF', 'aVR', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

LEADS_DICT = {
    'medalcare-xl': MEDALCARE_XL_LEADS,
    'ukbb':         UK_BB_LEADS,
    'mimic-iv':     MIMIC_IV_LEADS,
    'ptb-xl':       UK_BB_LEADS,
}

# Segment types required for each top-level mode, and whether the mode is atomic.
_SEG_MODE: dict[str, tuple[list[str], bool]] = {
    'all':          (['whole', 'atrial', 'ventricular'], True),
    'both':         (['atrial', 'ventricular'],          False),
    'whole':        (['whole'],                          False),
    'atrial':       (['atrial'],                         False),
    'ventricular':  (['ventricular'],                    False),
}


# ---------------------------------------------------------------------------
# Identifier and label helpers
# ---------------------------------------------------------------------------

def _parse_medalcare_ids(path: str) -> tuple[str, str, str]:
    """Extract (run_id, session_id) from a MedalCare-XL file path."""

    print(path)
    parts = path.split('/')
    run_id     = parts[-2].split('_')[1]
    session_id = parts[-1].split('_')[0]
    _cls = parts[-4]

    return run_id, session_id, _cls


def _uid_from_row(row, dataset: str) -> str:
    """Derive the base UID from a CSV row without loading the ECG file.

    Mirrors get_unique_id() but works on raw path strings.
    """
    path = str(row.data_path)
    if dataset == 'medalcare-xl':
        parts = path.split('/')
        run_id     = parts[-2].split('_')[1]
        session_id = parts[-1].split('_')[0]
        _cls       = parts[-4]
        return f'{run_id}_{session_id}_{_cls}'
    return os.path.splitext(os.path.basename(path))[0]


def get_unique_id(record, dataset: str, idx: int | None = None) -> str:
    """Stable, human-readable identifier for one ECG recording.

    MedalCare-XL : {run_id}_{session_id}
    UK Biobank / MIMIC-IV : stem of the original filename (EID)

    Append _{idx} for sampled beat type (multiple segments per recording).
    """
    path = str(record.original_file_path)
    if dataset == 'medalcare-xl':
        run_id, session_id, _cls = _parse_medalcare_ids(path)
        uid = f'{run_id}_{session_id}_{_cls}'
    else:
        uid = os.path.splitext(os.path.basename(path))[0]
    return f'{uid}_{idx}' if idx is not None else uid


def get_labels(record, dataset: str, phenotype_data: dict | None = None) -> dict:
    """Build the labels dict for the metadata file.

    MedalCare-XL : {'class': <groundtruth>, 'patient_id': <run_id>}
    UK Biobank / MIMIC-IV : {'patient_id': <eid>} + all phenotype columns
                            when phenotype_data is provided.
    """
    if dataset == 'medalcare-xl':
        run_id, _, _ = _parse_medalcare_ids(str(record.original_file_path))

        return {
            'class':      record.groundtruth if hasattr(record, 'groundtruth') else None,
            'patient_id': run_id,
        }        
    
    pid = os.path.splitext(os.path.basename(str(record.original_file_path)))[0]
    labels: dict[str, object] = {'patient_id': pid}
    if dataset == 'ptb-xl':
        labels.update({
            'superclass': record.superclass if hasattr(record, 'superclass') else None,
            'subclass':   record.subclass   if hasattr(record, 'subclass')   else None,
            'form':       record.form       if hasattr(record, 'form')       else None,
            'rhythm':     record.rhythm     if hasattr(record, 'rhythm')     else None,
        })
        return labels
    
    if phenotype_data is not None:
        eid_to_idx = phenotype_data['eid_to_idx']
        idx = eid_to_idx.get(str(pid))
        if idx is None and pid.isdigit():
            idx = eid_to_idx.get(int(pid))
        if idx is not None:
            # Vectorised: extract entire row at once, then zip with column names.
            row = phenotype_data['targets'][idx].tolist()
            labels.update(zip(phenotype_data['columns'], row))

    return labels


# ---------------------------------------------------------------------------
# ECG loading and WFDB conversion
# ---------------------------------------------------------------------------

def _write_wfdb(case: str, ecg: np.ndarray, directory_path: str, dataset: str):
    wfdb.wrsamp(
        record_name=case,
        write_dir=directory_path,
        fs=FS,
        units=['mV'] * ecg.shape[1],
        sig_name=LEADS_DICT[dataset][:ecg.shape[1]],
        p_signal=ecg,
        fmt=['16'] * ecg.shape[1],
    )


def convert_ecg_to_wfdb(filename: str, ecg_path: str, directory_path: str, dataset: str):
    case = os.path.splitext(filename)[0]
    if filename.endswith('.csv'):
        ecg = pd.read_csv(ecg_path, header=None, dtype=np.float32).to_numpy()
    else:  # .npy
        ecg = np.load(ecg_path)
    if ecg.shape[0] < ecg.shape[1]:
        ecg = ecg.T
    np.nan_to_num(ecg, nan=0.0, copy=False)
    _write_wfdb(case, ecg, directory_path, dataset)


def load_and_convert_case(row, dataset: str):
    """Load one ECG file, converting to WFDB format if needed.

    Returns (Record, wfdb_rec).
    """
    MIN_LENGTH = 100
    ecg_path               = str(row.data_path)
    directory_path, fname  = os.path.split(ecg_path)
    case                   = os.path.splitext(fname)[0]
    filepath               = os.path.join(directory_path, case)

    if not os.path.exists(filepath + '.dat') or not os.path.exists(filepath + '.hea'):
        convert_ecg_to_wfdb(fname, ecg_path, directory_path, dataset)

    try:
        rec = wfdb.rdrecord(filepath)
        if dataset == 'mimic-iv' and rec.sig_name != LEADS_DICT['mimic-iv']:
            rec.sig_name = LEADS_DICT['mimic-iv']
            wfdb.wrsamp(
                record_name=case, write_dir=directory_path,
                fs=rec.fs, units=rec.units,
                sig_name=rec.sig_name, p_signal=rec.p_signal, fmt=rec.fmt,
            )
    except Exception:
        print(f"  [conversion warning] WFDB read failed for {filepath}; attempting conversion and reload.")
        convert_ecg_to_wfdb(fname, ecg_path, directory_path, dataset)
        rec = wfdb.rdrecord(filepath)

    if rec.p_signal.shape[0] < MIN_LENGTH:
        raise ValueError(f"ECG too short: {rec.p_signal.shape[0]} samples ({filepath})")

    # Reorder columns to match UK_BB_LEADS so all datasets share a consistent lead layout.
    # Comparison is case-insensitive; sig_name is normalised to the UK_BB_LEADS casing.
    current   = list(rec.sig_name)
    current_l = [l.lower() for l in current]
    target_l  = [l.lower() for l in UK_BB_LEADS]

    if current_l != target_l:
        in_order  = [l for l in UK_BB_LEADS if l.lower() in set(current_l)]
        remainder = [current[i] for i, l in enumerate(current_l) if l not in set(target_l)]
        target    = in_order + remainder
        idx       = [current_l.index(l.lower()) for l in target]
        rec.p_signal = rec.p_signal[:, idx]
        rec.sig_name = target
    
    target_out = []
    for sig in rec.sig_name:
        if sig.lower() == 'avr':
            target_out.append('aVR')
        elif sig.lower() == 'avl':
            target_out.append('aVL')
        elif sig.lower() == 'avf':
            target_out.append('aVF')
        else:
            target_out.append(sig)

    rec.sig_name = target_out

    print(f'AFTER DDDEBBUGGGG')
    print(rec.p_signal.shape, rec.sig_name)
    print(f'AFTER DDDEBBUGGGG')

    ecg_dict = {name: rec.p_signal[:, i] for i, name in enumerate(rec.sig_name)}
    record = Record(ecg_dict, rec.fs, "DEMO", case)
    if dataset == 'ptb-xl':
        record.superclass = row.superclass if hasattr(row, 'superclass') else None
        record.subclass = row.subclass if hasattr(row, 'subclass') else None
        record.form = row.form if hasattr(row, 'form') else None
        record.rhythm = row.rhythm if hasattr(row, 'rhythm') else None

    if hasattr(row, 'label'):
        record.groundtruth = row.label
    if hasattr(row, 'hash'):
        record.hash = row.hash
    record.original_file_path = row.data_path

    npy_path = filepath + '.npy'
    if os.path.exists(npy_path):
        os.remove(npy_path)

    return record, rec


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _is_valid(val) -> bool:
    return val is not None and not np.isnan(val)


def get_ecg_segments_idx(record, segment_type: str, beat_type: str) -> tuple[list, bool]:
    """Compute (start, end) sample index pairs for one segment type.

    For 'atrial' / median beat, applies a two-tier estimation strategy when
    P-wave delineations are partially or fully missing:

        If P_onset  is missing → P_onset  = QRS_onset − 200 ms (100 samples at 500 Hz)
        If P_offset is missing → P_offset = QRS_onset −  20 ms  (10 samples at 500 Hz)
        If QRS onset is also unknown → cannot estimate → returns empty list.

    Returns
    -------
    segments    : list of (start, end) tuples
    p_estimated : True when P-wave boundaries were estimated
    """
    if segment_type == 'ventricular':
        if beat_type == 'sampled':
            qrst_idx = [
                (int(beat.onset), int(beat.t.offset))
                for beat in record.qrs
                if beat.t is not None
            ]
            k = min(2, len(qrst_idx))
            return (random.sample(qrst_idx, k) if k > 0 else []), False

        # median
        onset  = record.median_beat.delineations.qrs.onset
        offset = record.median_beat.delineations.t.offset
        if _is_valid(onset) and _is_valid(offset):
            return [(int(onset), int(offset))], False
        return [], False

    if segment_type == 'atrial':
        if beat_type == 'sampled':
            k = min(2, len(record.p))
            sampled_p = random.sample(record.p, k) if k > 0 else []
            return [(int(s.onset), int(s.offset)) for s in sampled_p], False

        # median
        p_onset   = record.median_beat.delineations.p.onset
        p_offset  = record.median_beat.delineations.p.offset
        qrs_onset = record.median_beat.delineations.qrs.onset

        if _is_valid(p_onset) and _is_valid(p_offset):
            return [(int(p_onset), int(p_offset))], False

        if _is_valid(qrs_onset):
            qrs_i      = int(qrs_onset)
            est_onset  = int(p_onset)  if _is_valid(p_onset)  else qrs_i - int(0.200 * FS)
            est_offset = int(p_offset) if _is_valid(p_offset) else qrs_i - int(0.020 * FS)
            est_onset  = max(0, est_onset)
            est_offset = max(est_onset + 1, est_offset)
            if est_onset < est_offset:
                return [(est_onset, est_offset)], True

        return [], False  # QRS onset unknown → cannot estimate

    # whole — full median beat, no sub-segmentation
    beat_len = record.median_beat.ecg.shape[1]  # shape: [leads, T]
    return [(0, beat_len)], False


# ---------------------------------------------------------------------------
# Saving  (directories are pre-created in the main loop)
# ---------------------------------------------------------------------------

def save_ecg_segment(
    segments: list[tuple[int, int]],
    raw_ecg: np.ndarray,
    base_uid: str,
    segment_type: str,
    save_dir: str,
    beat_type: str,
):
    """Slice raw_ecg into segments, normalise (except 'whole'), and save.

    Normalisation: per-lead z-score computed on the segment itself.
    'whole' segments are saved without any normalisation.
    Directories must be pre-created by the caller.
    """
    normalize = segment_type != 'whole'

    for i, (start, end) in enumerate(segments):
        segment = raw_ecg[start:end, :]  # view — no copy; normalization creates new array

        if normalize:
            mu     = np.mean(segment, axis=0, keepdims=True)
            sigma  = np.std(segment,  axis=0, keepdims=True)
            segment = (segment - mu) / (sigma + 1e-8)

        print(f'DDDEBBUGGGG')
        print(segment.shape)

        uid       = f'{base_uid}_{i}' if beat_type == 'sampled' else base_uid
        save_path = os.path.join(save_dir, f'{uid}.pth')
        torch.save(torch.from_numpy(segment.astype(np.float32)), save_path)


# ---------------------------------------------------------------------------
# Per-record processing (lock-free — returns result dict for caller to merge)
# ---------------------------------------------------------------------------

def process_and_save_segments(
    record,
    original_record,
    segment_type: str,
    out_dir: str,
    beat_type: str,
    dataset: str,
    phenotype_data: dict | None = None,
    out_beat_type: str | None = None,
) -> dict:
    """Segment and save one ECG record.  No shared mutable state is touched.

    Returns a result dict containing:
        uid            : str | None — base unique ID (None if nothing was saved)
        metadata       : dict | None
        seg_failures   : list of (path, seg_type, beat_type) to append to error_dict
        seg_counts     : dict (seg_type, beat_type) -> int for segmentation failure counts
        p_estimated    : bool
        discarded      : bool
        groundtruth    : str | None (MedalCare-XL class, for class-distribution stats)
        path           : str (original file path, for error tracking)
    """
    path = str(record.original_file_path)
    gt   = getattr(record, 'groundtruth', None)

    result: dict = {
        'uid':          None,
        'metadata':     None,
        'seg_failures': [],
        'seg_counts':   {},
        'p_estimated':  False,
        'discarded':    False,
        'groundtruth':  gt,
        'path':         path,
    }

    seg_types, atomic = _SEG_MODE[segment_type]

    # Guard: median beat must be available
    if beat_type == 'median' and getattr(record, 'median_beat', None) is None:
        result['discarded'] = True
        return result

    raw_ecg = (original_record.p_signal
               if beat_type == 'sampled'
               else record.median_beat.ecg.T)  # [T, n_leads]
    
    print(f'RAW ECG DDDEBBUGGGG')
    print(raw_ecg.shape)
    # ── Collect segments ──────────────────────────────────────────────────────
    collected: dict[str, tuple[list, bool]] = {}

    for seg_type in seg_types:
        try:
            segs, p_est = get_ecg_segments_idx(record, seg_type, beat_type)
        except Exception as e:
            print(f"  [segmentation error] {seg_type} | {path}: {e}")
            segs, p_est = [], False

        if segs:
            collected[seg_type] = (segs, p_est)
            if p_est:
                result['p_estimated'] = True
        else:
            key = (seg_type, beat_type)
            result['seg_counts'][key] = result['seg_counts'].get(key, 0) + 1
            result['seg_failures'].append((path, seg_type, beat_type))

    # ── Completeness check ────────────────────────────────────────────────────
    if (atomic and len(collected) < len(seg_types)) or not collected:
        result['discarded'] = True
        return result

    # ── Save ─────────────────────────────────────────────────────────────────
    base_uid  = get_unique_id(record, dataset)
    _save_bt  = out_beat_type if out_beat_type is not None else beat_type

    for seg_type, (segs, _) in collected.items():
        save_dir = os.path.join(out_dir, seg_type, _save_bt)
        save_ecg_segment(segs, raw_ecg, base_uid, seg_type, save_dir, beat_type)

    # Segment lengths: int for median (one segment per type), list for sampled.
    seg_lengths = {
        seg_type: (segs[0][1] - segs[0][0] if len(segs) == 1
                   else [e - s for s, e in segs])
        for seg_type, (segs, _) in collected.items()
    }

    result['uid']      = base_uid
    result['metadata'] = {
        'labels':            get_labels(record, dataset, phenotype_data),
        'p_wave_estimated':  result['p_estimated'],
        'segment_lengths':   seg_lengths,
    }
    return result


def _merge_result(r: dict, metadata_dict: dict, error_dict: dict, dataset: str):
    """Merge one worker result into the shared (single-threaded) accumulators."""
    if r['uid']:
        metadata_dict[r['uid']] = r['metadata']

    # Segmentation failure counts
    for (seg_type, bt), cnt in r['seg_counts'].items():
        error_dict['segmentation'][seg_type][bt] += cnt
    error_dict['segmentation_failures'].extend(r['seg_failures'])

    path = r['path']
    gt   = r['groundtruth']
    is_mc = dataset == 'medalcare-xl'

    if r['discarded']:
        error_dict['discarded']['count'] += 1
        error_dict['discarded']['paths'].append(path)
        if is_mc and gt:
            error_dict['discarded']['class_distribution'][gt] += 1

    if r['p_estimated']:
        error_dict['p_wave_estimated']['count'] += 1
        error_dict['p_wave_estimated']['paths'].append(path)
        if is_mc and gt:
            error_dict['p_wave_estimated']['class_distribution'][gt] += 1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ecg(ecg: np.ndarray, out_path: str, f: int = 500):
    t, l = ecg.shape
    time_ax = np.arange(t) / f
    fig, axes = plt.subplots(l, 1, sharex=True, squeeze=False)
    axes_flat = axes.flatten()
    for j in range(l):
        axes_flat[j].plot(time_ax, ecg[:, j], linewidth=0.7)
        if j < l - 1:
            axes_flat[j].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Demo pipeline: filtering helpers
# ---------------------------------------------------------------------------

def _butterworth_lowpass(signal: np.ndarray, fs: int = FS, cutoff: float = 40.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, signal, axis=0)


def _median_baseline_removal(signal: np.ndarray, fs: int = FS, w1_ms: int = 200, w2_ms: int = 600) -> np.ndarray:
    w1 = int(w1_ms * fs / 1000)
    w2 = int(w2_ms * fs / 1000)
    w1 += (w1 % 2 == 0)  # enforce odd kernel size
    w2 += (w2 % 2 == 0)
    baseline = np.empty_like(signal)
    for col in range(signal.shape[1]):
        baseline[:, col] = medfilt(medfilt(signal[:, col], w1), w2)
    return signal - baseline


def _demo_filter(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Replicate the pipeline filter: Butterworth LP → double median baseline removal."""
    return _median_baseline_removal(_butterworth_lowpass(signal, fs), fs)


# ---------------------------------------------------------------------------
# Demo pipeline: figure helpers
# ---------------------------------------------------------------------------

_12LEAD_GRID = [
    ['I',   'aVR', 'V1', 'V4'],
    ['II',  'aVL', 'V2', 'V5'],
    ['III', 'aVF', 'V3', 'V6'],
]

_WAVE_FILL  = {'P': '#cce4f7', 'QRS': '#fde8e8', 'T': '#d5f5e3'}
_WAVE_EDGE  = {'P': '#4a90d9', 'QRS': '#e74c3c', 'T': '#27ae60'}
_SEG_COLOR  = {'atrial': '#4a90d9', 'ventricular': '#e74c3c'}
_SEG_TITLE  = {'atrial': 'Atrial segment (P-wave)', 'ventricular': 'Ventricular segment (QRST)'}


def _lead_map(leads: list) -> dict:
    return {l.upper(): i for i, l in enumerate(leads)}


def _plot_12lead_grid(
    signal: np.ndarray,
    leads: list,
    fs: int,
    out_path: str,
    title: str = '',
    wave_regions: dict | None = None,
    time_unit: str = 's',
):
    """Publication-quality 12-lead ECG in standard 3×4 clinical grid.

    wave_regions: {'P': (onset_samp, offset_samp), 'QRS': ..., 'T': ...}
    """
    lidx = _lead_map(leads)
    T    = signal.shape[0]
    scale = 1000.0 if time_unit == 'ms' else 1.0
    t    = np.arange(T) / fs * scale

    fig, axes = plt.subplots(
        3, 4, figsize=(14, 5.5), sharex=True,
        gridspec_kw={'hspace': 0.42, 'wspace': 0.38},
    )
    fig.patch.set_facecolor('white')
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    for row, row_leads in enumerate(_12LEAD_GRID):
        for col, lead_name in enumerate(row_leads):
            ax = axes[row, col]
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if lead_name not in lidx:
                ax.set_visible(False)
                continue

            sig = signal[:, lidx[lead_name]]

            if wave_regions:
                for wave, (onset, offset) in wave_regions.items():
                    if onset is not None and offset is not None:
                        ax.axvspan(onset / fs * scale, offset / fs * scale,
                                   alpha=0.30, color=_WAVE_FILL[wave], lw=0,
                                   label=wave if (row == 0 and col == 0) else '_')

            ax.plot(t, sig, linewidth=0.75, color='#1a1a2e')
            ax.set_title(lead_name, fontsize=9, pad=3, fontweight='semibold')
            ax.tick_params(labelsize=7)

            if col == 0:
                ax.set_ylabel('mV', fontsize=7)
            if row == 2:
                ax.set_xlabel(time_unit, fontsize=7)

    # Legend for wave shading (attach to first axis)
    if wave_regions:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=_WAVE_FILL[w], edgecolor=_WAVE_EDGE[w],
                         alpha=0.6, label=w)
                   for w in wave_regions if wave_regions[w][0] is not None]
        if handles:
            axes[0, 0].legend(handles=handles, fontsize=7, loc='upper left',
                              frameon=False)

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_segment_pair(
    segments: dict,
    leads: list,
    fs: int,
    out_path: str,
):
    """All 12 leads stacked, atrial and ventricular side by side."""
    seg_types = [k for k in ('atrial', 'ventricular') if k in segments]
    n_seg  = len(seg_types)
    n_lead = segments[seg_types[0]].shape[1] if seg_types else 12
    n_lead = min(n_lead, len(leads))

    fig, axes = plt.subplots(
        n_lead, n_seg,
        figsize=(4.5 * n_seg, n_lead * 0.65),
        squeeze=False,
        gridspec_kw={'wspace': 0.45},
    )
    fig.patch.set_facecolor('white')
    fig.suptitle('Phase 4: Segments (z-score normalised)', fontsize=12,
                 fontweight='bold', y=1.01)

    for col, seg_type in enumerate(seg_types):
        sig = segments[seg_type]   # [T, n_leads]
        T   = sig.shape[0]
        t   = np.arange(T) / fs * 1000  # ms
        axes[0, col].set_title(_SEG_TITLE[seg_type], fontsize=10,
                                fontweight='bold', pad=6)

        for row in range(n_lead):
            ax = axes[row, col]
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.plot(t, sig[:, row], linewidth=0.9, color=_SEG_COLOR[seg_type])
            ax.tick_params(labelsize=6, left=False, labelleft=False)
            if col == 0:
                lead_label = leads[row] if row < len(leads) else f'L{row}'
                ax.set_ylabel(lead_label, rotation=0, labelpad=28,
                              va='center', fontsize=8)
            if row == n_lead - 1:
                ax.set_xlabel('ms', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_demo_pipeline(
    record,
    orig_signal: np.ndarray,
    filtered_signal: np.ndarray,
    segment_type: str,
    uid: str,
    save_dir: str,
    leads: list,
    fs: int = FS,
):
    """Save figures + .npy data for each preprocessing pipeline phase.

    Outputs (all under save_dir/):
      phase1_original.{npy,png}
      phase2_filtered.{npy,png}
      phase3_median_beat.{npy,png}, phase3_delineations.json
      phase4_{atrial,ventricular}.npy, phase4_segments.png, phase4_segment_meta.json
    """
    os.makedirs(save_dir, exist_ok=True)
    px = os.path.join(save_dir, uid)

    # ── Phase 1: Original ECG ─────────────────────────────────────────────────
    np.save(f'{px}_phase1_original.npy', orig_signal)
    _plot_12lead_grid(orig_signal, leads, fs,
                      f'{px}_phase1_original.png',
                      title='Phase 1: Original ECG', time_unit='s')
    print(f"  [demo] Phase 1 → {uid}_phase1_original.{{npy,png}}")

    # ── Phase 2: Filtered ECG ─────────────────────────────────────────────────
    np.save(f'{px}_phase2_filtered.npy', filtered_signal)
    _plot_12lead_grid(filtered_signal, leads, fs,
                      f'{px}_phase2_filtered.png',
                      title='Phase 2: Filtered ECG  (Butterworth 40 Hz LP + Median Baseline Removal)',
                      time_unit='s')
    print(f"  [demo] Phase 2 → {uid}_phase2_filtered.{{npy,png}}")

    # ── Phases 3–4 require a median beat ──────────────────────────────────────
    if getattr(record, 'median_beat', None) is None:
        print(f"  [demo] Phases 3–4 skipped (no median beat) for {uid}")
        return

    median_ecg = record.median_beat.ecg.T  # [T, n_leads]
    delin      = record.median_beat.delineations

    def _safe(obj, attr):
        v = getattr(obj, attr, None) if obj is not None else None
        return float(v) if _is_valid(v) else None

    p_wav = getattr(delin, 'p',   None)
    qrs   = getattr(delin, 'qrs', None)
    t_wav = getattr(delin, 't',   None)

    delineations = {
        'p_onset':    _safe(p_wav, 'onset'),
        'p_offset':   _safe(p_wav, 'offset'),
        'qrs_onset':  _safe(qrs,   'onset'),
        'qrs_offset': _safe(qrs,   'offset'),
        't_onset':    _safe(t_wav, 'onset'),
        't_offset':   _safe(t_wav, 'offset'),
    }

    # ── Phase 3: Median beat + delineations ───────────────────────────────────
    np.save(f'{px}_phase3_median_beat.npy', median_ecg)
    with open(f'{px}_phase3_delineations.json', 'w') as fh:
        json.dump(delineations, fh, indent=2)

    wave_regions = {
        'P':   (delineations['p_onset'],   delineations['p_offset']),
        'QRS': (delineations['qrs_onset'], delineations['qrs_offset']),
        'T':   (delineations['t_onset'],   delineations['t_offset']),
    }
    _plot_12lead_grid(median_ecg, leads, fs,
                      f'{px}_phase3_median_beat.png',
                      title='Phase 3: Median Beat with Delineations',
                      wave_regions=wave_regions, time_unit='ms')
    print(f"  [demo] Phase 3 → {uid}_phase3_median_beat.{{npy,png,json}}")

    # ── Phase 4: Segments ─────────────────────────────────────────────────────
    segments: dict = {}
    seg_meta: dict = {}

    for seg_t in ('atrial', 'ventricular'):
        try:
            segs, _ = get_ecg_segments_idx(record, seg_t, 'median')
        except Exception:
            segs = []
        if not segs:
            continue
        s, e   = segs[0]
        raw    = median_ecg[s:e, :]
        mu     = raw.mean(axis=0, keepdims=True)
        sigma  = raw.std(axis=0,  keepdims=True)
        norm   = (raw - mu) / (sigma + 1e-8)
        np.save(f'{px}_phase4_{seg_t}.npy', norm)
        segments[seg_t] = norm
        seg_meta[seg_t] = {'start': int(s), 'end': int(e),
                            'duration_ms': round((e - s) / fs * 1000, 1)}

    with open(f'{px}_phase4_segment_meta.json', 'w') as fh:
        json.dump(seg_meta, fh, indent=2)

    if segments:
        _plot_segment_pair(segments, leads, fs, f'{px}_phase4_segments.png')
        print(f"  [demo] Phase 4 → {uid}_phase4_{{atrial,ventricular}}.{{npy,png}}")


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _to_serialisable(obj):
    if isinstance(obj, (defaultdict, dict)):
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
# Retry preflight
# ---------------------------------------------------------------------------

def _retry_preflight(
    df,
    dataset: str,
    out_dir: str,
    split: str,
    segment_type: str,
    beat_type: str,
    beat_type_save: str,
) -> set[str]:
    """Copy already-processed files to the retry directory; return UIDs to skip.

    For each CSV row we derive the base UID without loading the ECG file.
    If any output file for that UID exists in the original {beat_type} directory
    (under any required segment sub-folder), we copy ALL matching files for that
    UID into the {beat_type_save} directory and mark the UID to be skipped during
    processing.
    """
    import glob as _glob
    import shutil

    seg_types = _SEG_MODE[segment_type][0]
    skip_uids: set[str] = set()
    n_copied = 0
    n_to_process = 0

    # Pre-create all retry destination directories
    for st in seg_types:
        os.makedirs(os.path.join(out_dir, split, st, beat_type_save), exist_ok=True)

    print(f"\nRetry pre-flight: scanning {len(df)} records …")
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Checking cache"):
        uid = _uid_from_row(row, dataset)
        found_any = False

        for st in seg_types:
            src_dir = os.path.join(out_dir, split, st, beat_type)
            dst_dir = os.path.join(out_dir, split, st, beat_type_save)
            # glob covers both median ({uid}.pth) and sampled ({uid}_0.pth, …)
            matches = _glob.glob(os.path.join(src_dir, f'{uid}*.pth'))
            for src in matches:
                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
                found_any = True

        if found_any:
            skip_uids.add(uid)
            n_copied += 1
        else:
            n_to_process += 1

    print(f"  Found in original dir  : {n_copied:>6}  → copied to '{beat_type_save}/'")
    print(f"  Not found (will process): {n_to_process:>6}")
    return skip_uids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="ALADIN-based ECG beat extraction and segmentation")
    argparser.add_argument("--input_path",   type=str, required=True,
                           help="CSV file listing ECG records for one split")
    argparser.add_argument("--out_dir",      type=str, required=True,
                           help="Root output directory")
    argparser.add_argument("--beat_type",    type=str, required=True,
                           choices=['sampled', 'median'])
    argparser.add_argument("--segment_type", type=str, default=None,
                           choices=['atrial', 'ventricular', 'both', 'whole', 'all'],
                           help=(
                               "'whole'  — full median beat, no sub-segmentation, no normalisation. "
                               "'atrial' — P-wave with estimation fallback. "
                               "'ventricular' — QRS-to-T. "
                               "'both'   — atrial + ventricular independently. "
                               "'all'    — whole + atrial + ventricular, atomic."
                           ))
    argparser.add_argument("--label_path",   type=str, default=None,
                           help="phenotype_targets.pt for UK Biobank; embeds labels in metadata.")
    argparser.add_argument("--batch_size",   type=int, default=32)
    argparser.add_argument("--workers",      type=int, default=os.cpu_count())
    argparser.add_argument("--demo",         action='store_true',
                           help="Process only the first 3 records")
    argparser.add_argument("--split",    type=str, required=True,
                           choices=['train', 'valid', 'test'])
    argparser.add_argument("--plot_only",    action='store_true',
                           help="Generate diagnostic plots only, do not save segments")
    argparser.add_argument("--plot_dir",     type=str,
                           default='/home/tchatupanyacho/rescue_ai_thesis/results/plots')
    argparser.add_argument("--retry",        action='store_true',
                           help=(
                               "Retry mode: save to {beat_type}_retry directories. "
                               "Files already present in the original {beat_type} dir are "
                               "copied over and skipped; only missing records are re-processed."
                           ))
    args = argparser.parse_args()

    # ── Dataset and split detection ───────────────────────────────────────────
    dataset = (
        'medalcare-xl' if 'medalcare-xl' in args.input_path  or 'medalcare_xl' in args.input_path else
        'ukbb'         if 'ukbb'         in args.input_path else
        'ptb-xl'       if 'ptb-xl'       in args.input_path else
        os.path.basename(args.input_path).split('_')[0]
    )

    if args.segment_type is not None:
        segment_type = args.segment_type
    elif dataset == 'medalcare-xl':
        segment_type = os.path.splitext(args.input_path)[0].split('_')[-1]
    else:
        segment_type = 'both'

    beat_type = args.beat_type

    if segment_type in ('whole', 'all') and beat_type != 'median':
        raise ValueError(f"--segment_type {segment_type} requires --beat_type median")

    split = args.split

    beat_type_save = beat_type + '_retry' if args.retry else beat_type

    out_dir   = os.path.join(args.out_dir, split)
    error_dir = os.path.join(args.out_dir, 'errors')
    os.makedirs(out_dir,   exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    print(f"Dataset      : {dataset}")
    print(f"Split        : {split}")
    print(f"Segment type : {segment_type}")
    print(f"Beat type    : {beat_type}")
    if args.retry:
        print(f"Retry mode   : ON  (saving to '{beat_type_save}/')")

    # ── Pre-create all output directories ────────────────────────────────────
    # Done once here so workers never pay the makedirs syscall cost.
    if not args.plot_only:
        for st in _SEG_MODE[segment_type][0]:
            os.makedirs(os.path.join(out_dir, st, beat_type_save), exist_ok=True)

    # ── Optional: load UK Biobank phenotype targets ───────────────────────────
    phenotype_data: dict | None = None
    if args.label_path is not None:
        print(f"Loading phenotype targets from {args.label_path} ...")
        raw = torch.load(args.label_path, map_location='cpu', weights_only=False)
        eid_to_idx: dict = {}
        for i, eid in enumerate(raw['eids']):
            eid_to_idx[str(eid)] = i
            if str(eid).isdigit():
                eid_to_idx[int(eid)] = i
        phenotype_data = {
            'eids':       raw['eids'],
            'targets':    raw['targets'].cpu(),
            'columns':    raw['columns'],
            'eid_to_idx': eid_to_idx,
        }
        print(f"  {len(raw['eids'])} EIDs, {len(raw['columns'])} phenotypes.")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df     = pd.read_csv(args.input_path, nrows=3) if args.demo else pd.read_csv(args.input_path)
    def parse_label(s):
        return np.array(list(map(int, s.split(','))))
    
    if dataset == 'ptb-xl':
        for col in ['superclass', 'subclass', 'form', 'rhythm']:
            df[col] = df[col].apply(parse_label)

    chunks = [df.iloc[i:i + args.batch_size] for i in range(0, len(df), args.batch_size)]

    # ── Retry pre-flight: copy cached files, build skip set ──────────────────
    skip_uids: set[str] = set()
    if args.retry:
        skip_uids = _retry_preflight(
            df, dataset, args.out_dir, split, segment_type, beat_type, beat_type_save,
        )

    # ── Load ALADIN ────────────────────────────────────────────────────────────
    print("Loading ALADIN ...")
    aladin = ALADIN(
        modelpaths=["ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
        debug={"segmenter": False, "afibdetector": False, "reflection": False, "total": False},
    )

    print(f"Processing {len(df)} records in {len(chunks)} batches ({args.workers} workers) ...")

    # ── Error / stats tracking (accessed only by the main thread) ─────────────
    error_dict = {
        'convert_case':           defaultdict(int),
        'save_segment':           defaultdict(int),
        'median_beat_extraction': {'count': 0, 'paths': [], 'class_distribution': defaultdict(int)},
        'p_wave_estimated':       {'count': 0, 'paths': [], 'class_distribution': defaultdict(int)},
        'discarded':              {'count': 0, 'paths': [], 'class_distribution': defaultdict(int)},
        'segmentation': {
            'ventricular': {'sampled': 0, 'median': 0},
            'atrial':      {'sampled': 0, 'median': 0},
            'whole':       {'sampled': 0, 'median': 0},
        },
        'segmentation_failures': [],
    }

    metadata_dict: dict = {}
    n_loaded  = 0
    n_success = 0
    is_mc     = dataset == 'medalcare-xl'

    # ── Main processing loop ──────────────────────────────────────────────────
    for chunk in tqdm(chunks, desc="Batches"):

        # Load + convert ECG files in parallel (I/O-bound)
        loaded_data: list = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(load_and_convert_case, row, dataset): row
                for row in chunk.itertuples(index=False)
            }
            for future in as_completed(futures):
                try:
                    loaded_data.append(future.result())
                    n_loaded += 1
                except Exception as e:
                    error_dict['convert_case'][type(e).__name__] += 1
                    print(f"  [load error] {e}")

        if not loaded_data:
            continue

        records          = [d[0] for d in loaded_data]
        original_records = [d[1] for d in loaded_data]

        # In retry mode, drop records whose files were already copied
        if skip_uids:
            pairs = [(rec, orig) for rec, orig in zip(records, original_records)
                     if get_unique_id(rec, dataset) not in skip_uids]
            records, original_records = ([p[0] for p in pairs],
                                          [p[1] for p in pairs])
            if not records:
                del loaded_data
                continue

        # ALADIN batch segmentation + reflection (must be sequential)
        aladin.segmenter.batch(records)
        aladin.reflection.batch(records)

        # Median beat extraction (sequential; ALADIN may not be thread-safe)
        if beat_type == 'median':
            for record in tqdm(records, desc="Median beats", leave=False):
                try:
                    aladin.calculate_median(record, 0.4, 0.6, 0.1)
                except Exception as e:
                    ed = error_dict['median_beat_extraction']
                    ed['count'] += 1
                    ed['paths'].append(str(record.original_file_path))
                    if is_mc and hasattr(record, 'groundtruth'):
                        ed['class_distribution'][record.groundtruth] += 1
                    print(f"  [median beat error] {record.recordname}: {e}")

            # Demo pipeline figures — one set of phase plots per record
            if args.demo:
                demo_dir = os.path.join(args.plot_dir, 'pipeline_demo')
                os.makedirs(demo_dir, exist_ok=True)
                for record, orig in zip(records, original_records):
                    uid      = get_unique_id(record, dataset)
                    orig_sig = orig.p_signal                            # [T, n_leads]
                    filt_sig = _demo_filter(orig_sig, FS)
                    _leads   = list(orig.sig_name) if hasattr(orig, 'sig_name') else UK_BB_LEADS
                    save_demo_pipeline(
                        record, orig_sig, filt_sig, segment_type,
                        uid, demo_dir, _leads,
                    )

        if args.plot_only:
            os.makedirs(args.plot_dir, exist_ok=True)
            for record, orig in zip(records, original_records):
                uid = get_unique_id(record, dataset)
                if beat_type == 'sampled':
                    aladin.plot(record, name=os.path.join(args.plot_dir, f'{uid}_ecg'))
                elif getattr(record, 'median_beat', None) is not None:
                    plot_ecg(record.median_beat.ecg.T,
                             os.path.join(args.plot_dir, f'{uid}_median.png'))
                    plot_ecg(orig.p_signal,
                             os.path.join(args.plot_dir, f'{uid}_original.png'))
        else:
            # Submit save jobs; workers are lock-free and return result dicts.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                save_futures = [
                    pool.submit(
                        process_and_save_segments,
                        rec, orig, segment_type, out_dir, beat_type,
                        dataset, phenotype_data, beat_type_save,
                    )
                    for rec, orig in zip(records, original_records)
                ]
                for future in as_completed(save_futures):
                    try:
                        result = future.result()
                        _merge_result(result, metadata_dict, error_dict, dataset)
                        n_success += 1
                    except Exception as e:
                        error_dict['save_segment'][type(e).__name__] += 1
                        print(f"  [save error] {e}")

        del loaded_data, records, original_records
        gc.collect()

    # ── Save metadata ─────────────────────────────────────────────────────────
    metadata_path = os.path.join(args.out_dir, f'{split}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(_to_serialisable(metadata_dict), f, indent=2)
    print(f"\nMetadata → {metadata_path}  ({len(metadata_dict)} entries)")

    # ── Save statistics ───────────────────────────────────────────────────────
    stats = {
        'summary': {
            'total_records_in_csv':        len(df),
            'total_records_loaded':         n_loaded,
            'total_records_save_attempted': n_success,
            'total_patients_in_metadata':   len(metadata_dict),
            'load_success_rate_pct':        round(100 * n_loaded / max(len(df), 1), 2),
        },
        'median_beat_extraction_failures': _to_serialisable(error_dict['median_beat_extraction']),
        'p_wave_estimated':                _to_serialisable(error_dict['p_wave_estimated']),
        'discarded_samples':               _to_serialisable(error_dict['discarded']),
        'segmentation_failures_by_type':   _to_serialisable(error_dict['segmentation']),
        'segmentation_failure_paths':      error_dict['segmentation_failures'],
        'conversion_errors':               _to_serialisable(error_dict['convert_case']),
        'save_errors':                     _to_serialisable(error_dict['save_segment']),
    }
    stats_path = os.path.join(error_dir, f'{split}_{segment_type}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # ── Human-readable summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Records in CSV              : {len(df)}")
    print(f"  Successfully loaded         : {n_loaded}")
    print(f"  Patients saved to metadata  : {len(metadata_dict)}")
    print(f"  Median beat failures        : {error_dict['median_beat_extraction']['count']}")
    print(f"  P-wave boundaries estimated : {error_dict['p_wave_estimated']['count']}")
    print(f"  Samples discarded           : {error_dict['discarded']['count']}")
    if is_mc:
        pwe = dict(error_dict['p_wave_estimated']['class_distribution'])
        dis = dict(error_dict['discarded']['class_distribution'])
        if pwe:
            print(f"  P-wave estimated — class dist : {pwe}")
        if dis:
            print(f"  Discarded        — class dist : {dis}")
    print(f"\n  Statistics → {stats_path}")
    print(f"  Metadata   → {metadata_path}")

    if args.retry:
        print(f"\n  Files in '{beat_type_save}' directories:")
        for st in _SEG_MODE[segment_type][0]:
            d = os.path.join(out_dir, st, beat_type_save)
            n = len([f for f in os.listdir(d) if f.endswith('.pth')]) if os.path.isdir(d) else 0
            print(f"    {st:15s}: {n:>6} files  ({d})")

    print("=" * 60)
