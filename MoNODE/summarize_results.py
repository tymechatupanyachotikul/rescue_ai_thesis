"""
Summarise training + post-training results for a single model run.

Reads from the run directory saved by main.py / finetune.py:
  {run_dir}/training_metrics.json
  {run_dir}/final_finetune_results/{seg_type}/{latent_key}/classification/{param}/metrics.json
  {run_dir}/final_finetune_results/{seg_type}/{latent_key}/regression/{param}/metrics.json

Saves one summary JSON to:
  {output_dir}/{segment_type}_{model}_{dataset}.json

Usage (standalone):
  python summarize_results.py \\
      --run_dir  results/ecg/node/12_04_2026-10:00:00-0 \\
      --output_dir  results/summaries \\
      --model node \\
      --dataset medalcare-xl \\
      --segment_type atrial

Usage (called from main.py):
  from summarize_results import save_run_summary
  save_run_summary(run_dir=args.save, output_dir=args.summary_output_dir,
                   model=args.model, dataset=args.dataset,
                   segment_type=args.segment_type)
"""

import argparse
import json
import os
from pathlib import Path


# ─── helpers ─────────────────────────────────────────────────────────────────

def _load_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _walk_probe_results(finetune_root: str, seg_type: str) -> dict:
    """Walk final_finetune_results/{seg_type}/ and collect all metrics.json files.

    Returns a nested dict:
      {latent_key: {task: {param: {method: {metric: value, ...}}}}}

    For classification, also includes per_class_accuracy if present.
    """
    base = Path(finetune_root) / seg_type
    if not base.exists():
        return {}

    results: dict = {}

    for lkey_dir in sorted(base.iterdir()):
        if not lkey_dir.is_dir():
            continue
        lkey = lkey_dir.name
        results[lkey] = {}

        for task in ('classification', 'regression'):
            task_dir = lkey_dir / task
            if not task_dir.exists():
                continue
            results[lkey][task] = {}

            for param_dir in sorted(task_dir.iterdir()):
                if not param_dir.is_dir() or param_dir.name.startswith('_'):
                    continue
                param = param_dir.name
                mj = _load_json(param_dir / 'metrics.json')
                if mj is None:
                    continue

                # metrics.json structure differs by task:
                # classification: {method: {metrics: {...}, per_class_accuracy: {...}}}
                # regression:     {method: {metric: value, ...}}  (flat)
                results[lkey][task][param] = mj

    return results


def _flatten_probe_summary(probe_results: dict) -> dict:
    """Convert nested probe results into a clean flat-ish summary.

    Output shape:
    {
      latent_key: {
        classification: {
          param: {
            method: {
              accuracy: ..., f1: ..., auroc: ...,
              per_class_accuracy: {cls: value, ...}
            }
          }
        },
        regression: {
          param: {method: {r2: ..., mse: ...}}
        }
      }
    }
    """
    summary: dict = {}
    for lkey, tasks in probe_results.items():
        summary[lkey] = {}

        clf = tasks.get('classification', {})
        if clf:
            summary[lkey]['classification'] = {}
            for param, method_dict in clf.items():
                summary[lkey]['classification'][param] = {}
                for method, res in method_dict.items():
                    # Handle both flat and nested metrics.json formats
                    if 'metrics' in res:
                        metrics = res['metrics']
                        pca     = res.get('per_class_accuracy', {})
                    else:
                        metrics = res
                        pca     = {}

                    entry: dict = {}
                    for k in ('accuracy', 'balanced_accuracy', 'f1', 'f1_binary',
                              'f1_macro', 'auroc', 'auroc_macro', 'recall_macro'):
                        if k in metrics:
                            entry[k] = metrics[k]
                    if pca:
                        entry['per_class_accuracy'] = {str(k): v for k, v in pca.items()}
                    summary[lkey]['classification'][param][method] = entry

        reg = tasks.get('regression', {})
        if reg:
            summary[lkey]['regression'] = {}
            for param, method_dict in reg.items():
                summary[lkey]['regression'][param] = {}
                for method, res in method_dict.items():
                    if 'metrics' in res:
                        metrics = res['metrics']
                    else:
                        metrics = res
                    summary[lkey]['regression'][param][method] = {
                        k: metrics[k] for k in ('r2', 'mse', 'mae', 'r2_ci', 'mae_ci') if k in metrics
                    }

    return summary


# ─── label efficiency ─────────────────────────────────────────────────────────

def _walk_label_efficiency(finetune_root: str, seg_type: str) -> dict:
    """Load label_efficiency_summary.json for each latent key.

    Returns:
      {lkey: {fraction_str: {task: {param: {metric_mean: float, metric_std: float, ...}}}}}
    """
    base = Path(finetune_root) / seg_type
    results: dict = {}

    le_root = base / 'label_efficiency'
    if not le_root.exists():
        return {}

    for lkey_dir in sorted(le_root.iterdir()):
        if not lkey_dir.is_dir():
            continue
        summary_path = lkey_dir / 'label_efficiency_summary.json'
        data = _load_json(summary_path)
        if data is None:
            continue
        results[lkey_dir.name] = data   # fraction_str -> {regression: ..., classification: ...}

    return results


# ─── pearson correlations ──────────────────────────────────────────────────────

def _walk_pearson(finetune_root: str, seg_type: str) -> dict:
    """Load pearson_correlations.json for each latent key and summarise.

    For each param, reports the latent dimension with the largest |r| and its value.

    Returns:
      {lkey: {param: {max_abs_r: float, dim: int, r: float}}}
    """
    base = Path(finetune_root) / seg_type
    results: dict = {}

    for lkey_dir in sorted(base.iterdir()):
        if not lkey_dir.is_dir():
            continue
        pearson_path = lkey_dir / 'pearson' / 'pearson_correlations.json'
        data = _load_json(pearson_path)
        if data is None:
            continue

        lkey = lkey_dir.name
        results[lkey] = {}
        corrs = data.get('correlations', {})
        for param, r_list in corrs.items():
            arr = [r for r in r_list if r is not None and not (r != r)]  # drop NaN
            if not arr:
                continue
            max_abs_r = max(abs(r) for r in arr)
            best_dim  = next(i for i, r in enumerate(r_list)
                             if r is not None and abs(r) == max_abs_r)
            results[lkey][param] = {
                'max_abs_r': round(max_abs_r, 6),
                'dim':       best_dim,
                'r':         round(r_list[best_dim], 6),
            }

    return results


# ─── permutation test ────────────────────────────────────────────────────────

def _walk_permutation_test(finetune_root: str, seg_type: str) -> dict:
    """Load permutation_results.json for each latent key.

    Returns:
      {lkey: {param: {r2_mean, r2_std, mae_mean, mae_std, n_permutations}}}
    """
    base = Path(finetune_root) / seg_type
    results: dict = {}

    perm_root = base / 'permutation_test'
    if not perm_root.exists():
        return {}

    for lkey_dir in sorted(perm_root.iterdir()):
        if not lkey_dir.is_dir():
            continue
        data = _load_json(lkey_dir / 'permutation_results.json')
        if data is None:
            continue
        results[lkey_dir.name] = data

    return results


# ─── main entry point ─────────────────────────────────────────────────────────

def save_run_summary(
    run_dir: str,
    output_dir: str,
    model: str,
    dataset: str,
    segment_type: str | None,
    original_dir: str | None,
    filename: str | None = None,
    finetune_dir: str | None = None,
) -> str:
    """Compile and save a summary JSON for one run.

    Parameters
    ----------
    run_dir      : directory where main.py saved its outputs (args.save)
    output_dir   : directory to write the summary JSON
    model        : model name (node / vae / monode / hbnode)
    dataset      : dataset name (medalcare-xl / uk-biobank)
    segment_type : atrial | ventricular | whole | None
    finetune_dir: finetune result directory

    Returns the path to the saved JSON.
    """
    seg_label    = segment_type or 'all'
    dataset_slug = dataset.lower().replace('-', '_').replace(' ', '_')
    if filename is None:
        filename = f"{seg_label}_{model}_{dataset_slug}.json"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    summary: dict = {
        'model':        model,
        'dataset':      dataset,
        'segment_type': segment_type,
        'run_dir':      os.path.abspath(run_dir),
    }

    # ── Training metrics ──────────────────────────────────────────────────────
    train_mj = _load_json(os.path.join(run_dir, 'training_metrics.json'))
    if train_mj is not None:
        summary['training'] = train_mj
    else:
        if original_dir is not None:
            train_mj = _load_json(os.path.join(original_dir, 'training_metrics.json'))
        
        if train_mj is None:
            print(f"  [summary] Warning: training_metrics.json not found in {run_dir}")
            summary['training'] = {}

    # ── Probe results ─────────────────────────────────────────────────────────
    finetune_root = os.path.join(run_dir, 'final_finetune_results' if finetune_dir is None else finetune_dir)
    probe_seg     = seg_label  # matches the sub-folder written by run_post_training_probes

    raw_probe = _walk_probe_results(finetune_root, probe_seg)
    if raw_probe:
        summary['probes'] = _flatten_probe_summary(raw_probe)
    else:
        # Fallback: try 'all_classes' sub-folder used when seg_type is None
        raw_probe = _walk_probe_results(finetune_root, 'all_classes')
        summary['probes'] = _flatten_probe_summary(raw_probe) if raw_probe else {}
        if not summary['probes']:
            print(f"  [summary] Warning: no probe results found under {finetune_root}")

    # ── Label efficiency ──────────────────────────────────────────────────────
    le = _walk_label_efficiency(finetune_root, probe_seg)
    if not le:
        le = _walk_label_efficiency(finetune_root, 'all_classes')
    summary['label_efficiency'] = le

    # ── Pearson correlations ──────────────────────────────────────────────────
    pearson = _walk_pearson(finetune_root, probe_seg)
    if not pearson:
        pearson = _walk_pearson(finetune_root, 'all_classes')
    summary['pearson'] = pearson

    # ── Permutation test ──────────────────────────────────────────────────────
    perm = _walk_permutation_test(finetune_root, probe_seg)
    if not perm:
        perm = _walk_permutation_test(finetune_root, 'all_classes')
    summary['permutation_test'] = perm

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [summary] Saved run summary → {out_path}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compile training + probe results into a single summary JSON.",
    )
    parser.add_argument('--run_dir',      required=True,
                        help="Run directory (args.save from main.py)")
    parser.add_argument('--original_dir',      required=False, default=None,
                        help="Original directory (args.save from main.py)")
    parser.add_argument('--output_dir',   required=True,
                        help="Where to write the summary JSON")
    parser.add_argument('--filename',   required=False, default=None,
                        help="Filename to be saved")
    parser.add_argument('--model',        required=True,
                        help="Model name (node / vae / monode / hbnode)")
    parser.add_argument('--dataset',      required=True,
                        help="Dataset name (medalcare-xl / uk-biobank)")
    parser.add_argument('--segment_type', default=None,
                        choices=['atrial', 'ventricular', 'whole', 'combined'],
                        help="Segment type (leave blank for no segmentation)")
    args = parser.parse_args()

    path = save_run_summary(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        model=args.model,
        dataset=args.dataset,
        segment_type=args.segment_type,
        original_dir=args.original_dir,
        filename=args.filename,
    )
    print(f"Done: {path}")


if __name__ == '__main__':
    main()
