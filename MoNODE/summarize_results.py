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
                        k: metrics[k] for k in ('r2', 'mse') if k in metrics
                    }

    return summary


# ─── main entry point ─────────────────────────────────────────────────────────

def save_run_summary(
    run_dir: str,
    output_dir: str,
    model: str,
    dataset: str,
    segment_type: str | None,
    original_dir: str | None,
) -> str:
    """Compile and save a summary JSON for one run.

    Parameters
    ----------
    run_dir      : directory where main.py saved its outputs (args.save)
    output_dir   : directory to write the summary JSON
    model        : model name (node / vae / monode / hbnode)
    dataset      : dataset name (medalcare-xl / uk-biobank)
    segment_type : atrial | ventricular | whole | None

    Returns the path to the saved JSON.
    """
    seg_label   = segment_type or 'all'
    dataset_slug = dataset.lower().replace('-', '_').replace(' ', '_')
    filename    = f"{seg_label}_{model}_{dataset_slug}.json"

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
    finetune_root = os.path.join(run_dir, 'final_finetune_results')
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
    parser.add_argument('--model',        required=True,
                        help="Model name (node / vae / monode / hbnode)")
    parser.add_argument('--dataset',      required=True,
                        help="Dataset name (medalcare-xl / uk-biobank)")
    parser.add_argument('--segment_type', default=None,
                        choices=['atrial', 'ventricular', 'whole'],
                        help="Segment type (leave blank for no segmentation)")
    args = parser.parse_args()

    path = save_run_summary(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        model=args.model,
        dataset=args.dataset,
        segment_type=args.segment_type,
        original_dir=args.original_dir,
    )
    print(f"Done: {path}")


if __name__ == '__main__':
    main()
