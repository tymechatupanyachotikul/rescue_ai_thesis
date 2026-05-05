#!/usr/bin/env python3
"""
visualise_latents.py — UMAP and t-SNE visualisation of atrial/ventricular latent spaces.

Loads pre-saved latent files from two model run directories (atrial and
ventricular), combines all splits while preserving split order
(train → valid → test), and produces scatter plots coloured by diagnosis labels.

For each latent key present in the .npz files, six plots are produced:
  • atrial latents only     × {t-SNE, UMAP}
  • ventricular latents only × {t-SNE, UMAP}
  • combined (matched patients, atrial ∥ ventricular concatenated) × {t-SNE, UMAP}

Each projection is computed independently ("separately") for the three views.

Label colouring
---------------
  ptb_xl      — diagnostic superclass (multi-hot → active class names joined by '+')
  medalcare-xl — 'class' field, all MI subclasses collapsed to 'MI'
  uk_biobank   — labels used as-is; use --color_key to specify which label field

Output filenames
----------------
  {save_dir}/{model}_{dataset}_{latent_key}_{view}_{method}.png
  view ∈ {atrial, ventricular, combined}
  method ∈ {tsne, umap}

Usage
-----
  python visualise_latents.py \\
      --atrial_latents_dir      results/atrial_run/latents \\
      --ventricular_latents_dir results/ventricular_run/latents \\
      --model  node \\
      --dataset ptb_xl \\
      --save_dir results/visualisations
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from finetune import _load_split

try:
    from umap import UMAP as _UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed — UMAP plots will be skipped.")

# ── Label constants ────────────────────────────────────────────────────────────

PTBXL_SUPERCLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

_ATRIAL_CLASSES      = {'avblock', 'fam', 'iab', 'lae'}
_VENTRICULAR_CLASSES = {'mi', 'lbbb', 'rbbb'}
_ALL_KNOWN_CLASSES   = _ATRIAL_CLASSES | _VENTRICULAR_CLASSES | {'sinus'}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_splits(latents_dir: str) -> tuple[dict, list]:
    """Concatenate all available splits (train, valid, test) from *latents_dir*.

    Returns:
        latents  — dict[key → np.ndarray shape (N_total, D)]
        metadata — list of N_total entry dicts (each has 'uid', 'labels')
    """
    all_latents: dict[str, list[np.ndarray]] = defaultdict(list)
    all_metadata: list = []

    for split in ('train', 'valid', 'test'):
        try:
            lat, meta = _load_split(latents_dir, split)
            for k, v in lat.items():
                all_latents[k].append(v)
            all_metadata.extend(meta)
            print(f"    [{split}] {len(meta)} samples")
        except FileNotFoundError:
            pass

    if not all_metadata:
        raise FileNotFoundError(f"No latent splits found in: {latents_dir}")

    combined = {k: np.concatenate(vs, axis=0) for k, vs in all_latents.items()}
    print(f"    Total: {len(all_metadata)} samples | keys: {list(combined.keys())}")
    return combined, all_metadata


def _group_by_patient(
    latents: dict, metadata: list
) -> tuple[list, dict[str, np.ndarray], dict[str, dict]]:
    """Average latent vectors per patient UID across multiple beats.

    Returns:
        sorted_pids    — sorted list of unique patient IDs
        latent_arrays  — {key: ndarray [N_patients, D]}
        label_by_pid   — {pid: labels_dict}
    """
    pid_latents: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pid_labels: dict[str, dict] = {}

    for i, entry in enumerate(metadata):
        pid = str(entry.get('uid') or entry.get('patient_id', f'idx_{i}'))
        for k, arr in latents.items():
            pid_latents[pid][k].append(arr[i])
        pid_labels[pid] = dict(entry.get('labels', {}))

    sorted_pids = sorted(pid_latents.keys())
    latent_arrays = {
        k: np.stack([np.mean(pid_latents[pid][k], axis=0) for pid in sorted_pids])
        for k in latents
    }
    return sorted_pids, latent_arrays, pid_labels


def build_combined(
    atrial_latents: dict, atrial_meta: list,
    ventricular_latents: dict, ventricular_meta: list,
) -> tuple[dict, list]:
    """Match patients across modalities and horizontally concatenate their latents.

    Averages over multiple beats per patient before matching.  Only latent keys
    present in both atrial and ventricular sets are included in the output.

    Returns:
        combined_latents — {key: ndarray [N_matched, D_a + D_v]}
        combined_meta    — list of N_matched dicts with merged labels
    """
    pids_a, lat_a, labels_a = _group_by_patient(atrial_latents,      atrial_meta)
    pids_v, lat_v, labels_v = _group_by_patient(ventricular_latents, ventricular_meta)

    common = sorted(set(pids_a) & set(pids_v))
    if not common:
        raise ValueError("No common patient IDs between atrial and ventricular sets.")

    print(f"    Matched {len(common)} patients "
          f"(atrial-only={len(set(pids_a) - set(pids_v))}, "
          f"ventricular-only={len(set(pids_v) - set(pids_a))})")

    idx_a = {pid: i for i, pid in enumerate(pids_a)}
    idx_v = {pid: i for i, pid in enumerate(pids_v)}

    shared_keys = sorted(set(lat_a) & set(lat_v))
    combined_latents = {}
    for k in shared_keys:
        rows_a = np.stack([lat_a[k][idx_a[pid]] for pid in common])
        rows_v = np.stack([lat_v[k][idx_v[pid]] for pid in common])
        combined_latents[k] = np.concatenate([rows_a, rows_v], axis=1)

    combined_meta = [
        {
            'uid': pid, 'patient_id': pid,
            'labels': {**labels_v[pid], **labels_a[pid]},
        }
        for pid in common
    ]
    return combined_latents, combined_meta


# ── Label extraction ───────────────────────────────────────────────────────────

def extract_label(entry: dict, dataset: str, color_key: str | None) -> str:
    """Return a single categorical label string for one metadata entry."""
    labels = entry.get('labels', {})

    if color_key is not None:
        val = labels.get(color_key)
        return str(val) if val is not None else 'unknown'

    dl = dataset.lower()

    if 'ptb' in dl:
        sc = labels.get('superclass')
        if isinstance(sc, list):
            active = [
                PTBXL_SUPERCLASS_NAMES[i]
                for i, v in enumerate(sc)
                if i < len(PTBXL_SUPERCLASS_NAMES) and v
            ]
            return '+'.join(active) if active else 'NONE'
        return str(sc) if sc is not None else 'NONE'

    if 'medalcare' in dl or 'medal' in dl:
        cls = labels.get('class', 'unknown')
        # Collapse 'mi' and all MI subclasses (anything not in the known set) → 'MI'
        if cls == 'mi' or cls not in _ALL_KNOWN_CLASSES:
            return 'MI'
        return cls

    # UK Biobank / other: use as-is; try common categorical keys
    for key in ('class', 'diagnosis', 'label', 'pathology'):
        if key in labels:
            return str(labels[key])
    return 'unknown'


def get_labels(metadata: list, dataset: str, color_key: str | None) -> list[str]:
    return [extract_label(m, dataset, color_key) for m in metadata]


# ── Dimensionality reduction ───────────────────────────────────────────────────

def _embed_2d(
    X: np.ndarray,
    method: str,
    seed: int,
    perplexity: float,
    n_neighbors: int,
) -> np.ndarray | None:
    if method == 'tsne':
        perp = min(perplexity, len(X) - 1)
        return TSNE(
            n_components=2, random_state=seed, perplexity=perp,
            n_iter=1000, init='pca', learning_rate='auto',
        ).fit_transform(X)

    if method == 'umap':
        if not HAS_UMAP:
            return None
        try:
            return _UMAP(
                n_components=2, random_state=seed,
                n_neighbors=min(n_neighbors, len(X) - 1),
            ).fit_transform(X)
        except Exception as exc:
            print(f"    UMAP failed: {exc}")
            return None

    raise ValueError(f"Unknown method: {method!r}")


def _maybe_subsample(
    X: np.ndarray, labels: list[str], max_n: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    if len(labels) <= max_n:
        return X, labels
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(labels), size=max_n, replace=False))
    return X[idx], [labels[i] for i in idx]


# ── Plotting ───────────────────────────────────────────────────────────────────

def _save_scatter(
    E2d: np.ndarray,
    labels: list[str],
    title: str,
    save_path: str,
) -> None:
    unique = sorted(set(labels))
    n_cls  = len(unique)
    cmap   = mpl_cm.get_cmap('tab20' if n_cls <= 20 else 'hsv', n_cls)
    l2i    = {l: i for i, l in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(
            E2d[mask, 0], E2d[mask, 1],
            s=4, alpha=0.5, color=cmap(l2i[lbl]),
            label=f'{lbl} (n={mask.sum()})', rasterized=True,
        )

    ncol = max(1, n_cls // 12)
    ax.legend(
        fontsize=6, markerscale=2.5, ncol=ncol,
        framealpha=0.7, loc='best',
        handletextpad=0.3, columnspacing=0.5,
    )
    ax.set_xlabel('Component 1', fontsize=9)
    ax.set_ylabel('Component 2', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {os.path.basename(save_path)}")


# ── Main visualisation driver ──────────────────────────────────────────────────

def visualise_view(
    X: np.ndarray,
    labels: list[str],
    title_prefix: str,
    save_prefix: str,
    max_samples: int,
    seed: int,
    perplexity: float,
    n_neighbors: int,
) -> None:
    """Scale → subsample → project → save for all enabled methods."""
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    X_sub, labels_sub = _maybe_subsample(X_sc, labels, max_samples, seed)
    n_sub   = len(labels_sub)
    n_cls   = len(set(labels_sub))
    print(f"    {n_sub} samples | {n_cls} unique labels")

    methods = []
    if HAS_UMAP:
        methods.append('umap')
    methods.append('tsne')

    for method in methods:
        method_label = 'UMAP' if method == 'umap' else 't-SNE'
        title      = f'{title_prefix} — {method_label}'
        save_path  = f'{save_prefix}_{method}.png'
        print(f"    Running {method_label}...")
        try:
            E2d = _embed_2d(X_sub, method, seed, perplexity, n_neighbors)
            if E2d is None:
                continue
            _save_scatter(E2d, labels_sub, title, save_path)
        except Exception as exc:
            print(f"    {method_label} failed: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='UMAP / t-SNE visualisation of atrial and ventricular latent spaces.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--atrial_latents_dir', required=True,
                        help='Directory containing atrial latents '
                             '({split}_latents.npz / {split}_metadata.json)')
    parser.add_argument('--ventricular_latents_dir', required=True,
                        help='Directory containing ventricular latents')
    parser.add_argument('--model',   required=True, help='Model name (used in output filenames)')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (ptb_xl | medalcare-xl | uk_biobank)')
    parser.add_argument('--save_dir', required=True, help='Output directory for PNG files')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Max samples passed to t-SNE / UMAP (subsampled if exceeded)')
    parser.add_argument('--color_key', default=None,
                        help='Metadata labels key to use for colouring '
                             '(overrides dataset-specific defaults)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tsne_perplexity', type=float, default=30)
    parser.add_argument('--umap_neighbors',  type=int,   default=15)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    dataset_slug = args.dataset.lower().replace('-', '_').replace(' ', '_')

    # ── Load all splits ───────────────────────────────────────────────────────
    print(f"\n── Loading atrial latents ({args.atrial_latents_dir}) ──")
    atrial_lat, atrial_meta = load_all_splits(args.atrial_latents_dir)

    print(f"\n── Loading ventricular latents ({args.ventricular_latents_dir}) ──")
    vent_lat, vent_meta = load_all_splits(args.ventricular_latents_dir)

    # ── Build patient-matched combined set ────────────────────────────────────
    print(f"\n── Building combined (patient-matched) set ──")
    combined_lat, combined_meta = build_combined(
        atrial_lat, atrial_meta, vent_lat, vent_meta,
    )

    # ── Determine latent keys ─────────────────────────────────────────────────
    all_keys = sorted(set(atrial_lat) | set(vent_lat))
    print(f"\nLatent keys: {all_keys}")

    # ── Produce plots ─────────────────────────────────────────────────────────
    views = [
        ('atrial',      atrial_lat,  atrial_meta),
        ('ventricular', vent_lat,    vent_meta),
        ('combined',    combined_lat, combined_meta),
    ]

    for latent_key in all_keys:
        print(f"\n{'═' * 60}")
        print(f"  Latent key: {latent_key}")
        print(f"{'═' * 60}")

        for view_name, lat_dict, meta in views:
            if latent_key not in lat_dict:
                print(f"\n  [{view_name}] key not present — skipping")
                continue

            X      = lat_dict[latent_key]
            labels = get_labels(meta, args.dataset, args.color_key)
            print(f"\n  [{view_name}]  shape={X.shape}")

            title_prefix = (
                f'{args.model} | {args.dataset} | {latent_key} | {view_name}'
            )
            save_prefix = os.path.join(
                args.save_dir,
                f'{args.model}_{dataset_slug}_{latent_key}_{view_name}',
            )

            visualise_view(
                X, labels,
                title_prefix=title_prefix,
                save_prefix=save_prefix,
                max_samples=args.max_samples,
                seed=args.seed,
                perplexity=args.tsne_perplexity,
                n_neighbors=args.umap_neighbors,
            )

    print("\nDone.")


if __name__ == '__main__':
    main()
