"""
Bayesian hyperparameter search for MoNODE on MedalCare-XL (25 % subset).

Tunable parameters
------------------
  node : de_L, de_H, dec_L, dec_H
  vae  : rnn_hidden (encoder + decoder independently via rnn_hidden_dec)

Optimisation
------------
  Optuna TPE (Bayesian) with MedianPruner early stopping.
  Multi-GPU: one worker process per GPU sharing an SQLite Optuna study.

Usage
-----
  python hyperparameter_search.py --model node --segment_type atrial
  python hyperparameter_search.py --model vae  --segment_type ventricular --n_gpus 4 --n_trials 60
"""

import argparse
import copy
import json
import logging
import os
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Dataset constants ─────────────────────────────────────────────────────────

ATRIAL_CLASSES      = {'avblock', 'fam', 'iab', 'lae'}
VENTRICULAR_CLASSES = {'lbbb', 'rbbb'}

def _is_mi(cls: str) -> bool:
    """True for MI subclass labels (start with LAD_, LCX_, or RCA_)."""
    return cls.startswith(('LAD_', 'LCX_', 'RCA_'))

def _class_group(cls: str) -> str:
    return 'mi' if _is_mi(cls) else cls

def _segment_classes(segment_type: str) -> set:
    """Class groups relevant for this segment type."""
    if segment_type == 'atrial':
        return ATRIAL_CLASSES | {'sinus'}
    if segment_type == 'ventricular':
        return VENTRICULAR_CLASSES | {'mi', 'sinus'}
    # whole
    return ATRIAL_CLASSES | VENTRICULAR_CLASSES | {'mi', 'sinus'}


# ── Subset generation ─────────────────────────────────────────────────────────

def _stratified_sample(paths, classes, run_ids, segment_type, fraction, rng):
    """Return (paths, classes, run_ids) sampled at *fraction* with equal class counts.

    MI subclasses are pooled into one 'mi' group for balancing, then
    individual subclasses are sampled proportionally within that group.
    """
    allowed = _segment_classes(segment_type)

    # Filter to segment-relevant entries
    entries = [
        (p, c, r) for p, c, r in zip(paths, classes, run_ids)
        if _class_group(c) in allowed
    ]
    if not entries:
        return [], [], []

    # Group by class group
    groups: dict = defaultdict(list)
    for p, c, r in entries:
        groups[_class_group(c)].append((p, c, r))

    # Equal-ish allocation per group summing to fraction * total
    total_target = max(len(groups), int(len(entries) * fraction))
    n_groups     = len(groups)
    base_n       = total_target // n_groups
    remainder    = total_target % n_groups

    selected = []
    for i, grp in enumerate(sorted(groups)):
        items = groups[grp]
        n     = min(base_n + (1 if i < remainder else 0), len(items))
        chosen = [items[j] for j in rng.choice(len(items), size=n, replace=False)]
        selected.extend(chosen)

    sel_paths   = [e[0] for e in selected]
    sel_classes = [e[1] for e in selected]
    sel_run_ids = [e[2] for e in selected]
    return sel_paths, sel_classes, sel_run_ids


def generate_subset_data_params(
    original_path: str,
    output_path: str,
    segment_type: str,
    fraction: float = 0.25,
    seed: int = 42,
) -> dict:
    """Generate a stratified *fraction* subset of an existing data_params JSON.

    Reports class distribution per split and saves to *output_path*.
    Returns a per-split distribution report dict.
    """
    rng = np.random.default_rng(seed)

    with open(original_path) as f:
        orig = json.load(f)

    subset      = {}
    dist_report = {}

    for split in ('train', 'valid', 'test'):
        if split not in orig:
            continue
        s        = orig[split]
        paths    = s['file_paths']
        classes  = s.get('class',  ['unknown'] * len(paths))
        run_ids  = s.get('run_id', [''] * len(paths))

        sp, sc, sr = _stratified_sample(paths, classes, run_ids,
                                         segment_type, fraction, rng)
        subset[split] = {
            'file_paths':       sp,
            'class':            sc,
            'run_id':           sr,
            'exclude_leads_in': s.get('exclude_leads_in', []),
        }

        dist = Counter(_class_group(c) for c in sc)
        dist_report[split] = dict(dist)
        n_orig = len([p for p, c in zip(paths, classes)
                      if _class_group(c) in _segment_classes(segment_type)])
        print(f"  [{split}] {len(sp)}/{n_orig} samples "
              f"({len(sp)/max(n_orig,1)*100:.0f}%) | "
              + " | ".join(f"{k}: {v}" for k, v in sorted(dist.items())))

    with open(output_path, 'w') as f:
        json.dump(subset, f, indent=2)
    print(f"  Saved → {output_path}")
    return dist_report


def _print_dist(subset_path: str) -> None:
    with open(subset_path) as f:
        sp = json.load(f)
    for split in ('train', 'valid', 'test'):
        if split not in sp:
            continue
        classes = sp[split].get('class', [])
        dist    = Counter(_class_group(c) for c in classes)
        print(f"  [{split}] {len(sp[split]['file_paths'])} samples | "
              + " | ".join(f"{k}: {v}" for k, v in sorted(dist.items())))


# ── Path helpers ──────────────────────────────────────────────────────────────

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _original_params_path(segment_type, dataset='MedalCare-XL', sample_type='median') -> str:
    return os.path.join(_script_dir(), 'data', 'ecg',
                        f'{dataset}_{segment_type}_{sample_type}_data_params.json')

def _subset_params_path(segment_type, fraction=0.25,
                        dataset='MedalCare-XL', sample_type='median') -> str:
    pct = int(fraction * 100)
    return os.path.join(_script_dir(), 'data', 'ecg',
                        f'{dataset}_{segment_type}_{sample_type}_data_params_{pct}pct.json')


# ── Dataset builder (bypasses load_data to use subset JSON directly) ──────────

def _build_loaders(subset_path: str, args, dtype, ecg_cfg: dict):
    """Build train/valid/test DataLoaders from the subset data_params JSON."""
    sys.path.insert(0, _script_dir())
    from data.data_utils import ECGDataset, pad_collate
    import torch.utils.data as torchdata

    with open(subset_path) as f:
        params = json.load(f)

    loaders = {}
    for split in ('train', 'valid', 'test'):
        if split not in params or not params[split]['file_paths']:
            loaders[split] = None
            continue
        s  = params[split]
        ds = ECGDataset(
            s['file_paths'],
            s.get('class'),
            s.get('run_id'),
            dtype,
            'MedalCare-XL',
            s.get('exclude_leads_in', []),
            shared_cache={},
        )
        bs     = min(args.batch_size, len(s['file_paths']))
        loader = torchdata.DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == 'train'),
            num_workers=0,
            drop_last=(split == 'train'),
            collate_fn=pad_collate,
        )
        loaders[split] = loader

    # task params dict expected by train_model
    task_params = {
        'train':              {'T': ecg_cfg.get('T', 200), 'N': len(params['train']['file_paths'])},
        'dataset':            'MedalCare-XL',
        'beat_type':          args.segment_type,
        'sample_type':        ecg_cfg.get('sample_type', 'median'),
        'exclude_leads_in':   ecg_cfg.get('exclude_leads_in', []),
        'exclude_leads_out':  ecg_cfg.get('exclude_leads_out', []),
        'f':                  ecg_cfg.get('f', 500),
        'use_cache':          True,
    }
    return loaders['train'], loaders['valid'], loaders['test'], task_params


# ── Training objective ────────────────────────────────────────────────────────

def _build_and_train(trial, args, gpu_id: int, subset_path: str, ecg_cfg: dict) -> float:
    """Train one trial; report intermediates to Optuna; return best val MSE.

    Post-training probes (run_post_training_probes) are intentionally skipped
    during hyperparameter search — only reconstruction val MSE is optimised.
    """
    import wandb
    from model.build_model import build_model
    from model.model_misc import train_model
    from model.misc.torch_utils import seed_everything
    from model.misc.plot_utils import Plotter

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float64
    seed_everything(args.seed + trial.number)   # different seed per trial

    os.makedirs(args.save, exist_ok=True)
    os.makedirs(os.path.join(args.save, 'plots'), exist_ok=True)

    logger = logging.getLogger(f'trial_{trial.number}')
    logger.setLevel(logging.WARNING)

    trainset, validset, testset, task_params = _build_loaders(
        subset_path, args, dtype, ecg_cfg)

    run = wandb.init(mode='disabled', name=f'hp_trial_{trial.number}')

    config = {
        'inp_dim': 12 - len(task_params['exclude_leads_in']),
        'w_dt':    args.sobolev_weight,
        'l_w':     args.l_w,
        'out_dim': 12 - len(task_params['exclude_leads_out']),
    }
    model = build_model(args, device, dtype, **config)
    model.to(device).to(dtype)

    plotter = Plotter(os.path.join(args.save, 'plots'), args.task)

    best_val = [float('inf')]

    def epoch_callback(ep: int, val_mse: float) -> bool:
        trial.report(val_mse, ep)
        best_val[0] = min(best_val[0], val_mse)
        return trial.should_prune()

    try:
        train_model(args, model, plotter, trainset, validset, testset,
                    logger, task_params, run, epoch_callback=epoch_callback)
        # NOTE: run_post_training_probes is deliberately NOT called here.
        # Post-training analysis runs only after full training with best HPs.
    except optuna.exceptions.TrialPruned:
        pass
    finally:
        run.finish()

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return best_val[0]


def objective(trial, base_args, subset_path: str, gpu_id: int, ecg_cfg: dict) -> float:
    """Optuna objective: sample HPs, build args, run training."""
    args = copy.deepcopy(base_args)

    if args.model == 'node':
        args.de_L  = trial.suggest_int('de_L',  2, 4)
        args.de_H  = trial.suggest_int('de_H',  100, 300, step=50)
        args.dec_L = trial.suggest_int('dec_L', 2, 4)
        args.dec_H = trial.suggest_int('dec_H', 100, 300, step=50)
    elif args.model == 'vae':
        args.rnn_hidden_dec = trial.suggest_int('rnn_hidden_dec', 32, 64, step=32)
    
    if args.segment_type == 'whole':
        args.rnn_hidden = 128 
        args.ode_latent_dim = 64
    else:
        args.rnn_hidden = 64 
        args.ode_latent_dim = 32

    args.save = os.path.join(
        base_args.hp_output_dir,
        f'{args.model}_{args.segment_type}',
        f'trial_{trial.number:04d}',
    )

    try:
        return _build_and_train(trial, args, gpu_id, subset_path, ecg_cfg)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  [trial {trial.number}] ERROR on GPU {gpu_id}: {e}")
        raise optuna.exceptions.TrialPruned()


# ── Worker (one per GPU) ──────────────────────────────────────────────────────

def _worker(rank: int, base_args, study_name: str, storage_url: str,
            subset_path: str, n_trials: int, ecg_cfg: dict) -> None:
    """Worker process: run on one GPU, share the Optuna study via SQLite."""
    n_gpus = torch.cuda.device_count()
    gpu_id = rank % n_gpus if n_gpus > 0 else 0

    # Restrict this process to one GPU
    if n_gpus > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        gpu_local = 0   # after restriction, device 0 is the target GPU
    else:
        gpu_local = 0

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(
        lambda trial: objective(trial, base_args, subset_path, gpu_local, ecg_cfg),
        n_trials=n_trials,
        gc_after_trial=True,
    )


# ── Results reporting ─────────────────────────────────────────────────────────

def _report_results(study_name: str, storage_url: str, args) -> None:
    """Load the finished study and print / save a comprehensive results report."""
    study    = optuna.load_study(study_name=study_name, storage=storage_url)
    finished = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED]
    failed   = [t for t in study.trials
                if t.state == optuna.trial.TrialState.FAIL]

    print(f"\n{'='*60}")
    print(f"Hyperparameter search complete — {study_name}")
    print(f"  Total trials : {len(study.trials)}")
    print(f"  Completed    : {len(finished)}")
    print(f"  Pruned       : {len(pruned)}")
    print(f"  Failed       : {len(failed)}")

    if not finished:
        print("\nNo trials completed — nothing to report.")
        return

    # ── Top-10 trials table ───────────────────────────────────────────────────
    top_n  = min(10, len(finished))
    ranked = sorted(finished, key=lambda t: t.value)[:top_n]
    param_keys = list(study.best_trial.params.keys())

    col_w  = max(10, *(len(k) + 2 for k in param_keys))
    header = f"{'Rank':>4}  {'Trial':>6}  {'Val MSE':>10}  " + \
             "  ".join(f"{k:>{col_w}}" for k in param_keys)
    sep    = "-" * len(header)

    print(f"\nTop-{top_n} trials (by validation MSE):")
    print(sep)
    print(header)
    print(sep)
    for rank, t in enumerate(ranked, 1):
        vals = "  ".join(f"{t.params.get(k, '—'):>{col_w}}" for k in param_keys)
        marker = " ◀ best" if rank == 1 else ""
        print(f"{rank:>4}  {t.number:>6}  {t.value:>10.6f}  {vals}{marker}")
    print(sep)

    # ── Parameter importance ──────────────────────────────────────────────────
    if len(finished) >= 5:
        try:
            importance = optuna.importance.get_param_importances(study)
            print("\nParameter importance (FAnova):")
            for param, imp in importance.items():
                bar = "█" * int(imp * 30)
                print(f"  {param:<20} {imp:.3f}  {bar}")
        except Exception:
            pass   # importance needs enough trials; silently skip if it fails

    # ── Val MSE statistics across all completed trials ────────────────────────
    all_vals = [t.value for t in finished]
    print(f"\nVal MSE across all {len(finished)} completed trials:")
    print(f"  best   : {min(all_vals):.6f}")
    print(f"  median : {np.median(all_vals):.6f}")
    print(f"  mean   : {np.mean(all_vals):.6f}")
    print(f"  worst  : {max(all_vals):.6f}")

    # ── Save comprehensive JSON report ────────────────────────────────────────
    best = study.best_trial
    report = {
        'model':        args.model,
        'segment_type': args.segment_type,
        'study_name':   study_name,
        'summary': {
            'n_total':    len(study.trials),
            'n_complete': len(finished),
            'n_pruned':   len(pruned),
            'n_failed':   len(failed),
        },
        'val_mse_stats': {
            'best':   float(min(all_vals)),
            'median': float(np.median(all_vals)),
            'mean':   float(np.mean(all_vals)),
            'worst':  float(max(all_vals)),
        },
        'best': {
            'trial':   best.number,
            'val_mse': best.value,
            'params':  best.params,
        },
        'top_trials': [
            {'trial': t.number, 'val_mse': t.value, 'params': t.params}
            for t in ranked
        ],
        'all_trials': [
            {'trial': t.number, 'val_mse': t.value, 'params': t.params}
            for t in sorted(finished, key=lambda t: t.value)
        ],
    }

    report_path = os.path.join(args.hp_output_dir,
                               f'results_{args.model}_{args.segment_type}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved → {report_path}")
    print(f"\nBest hyperparameters for {args.model} / {args.segment_type}:")
    for k, v in best.params.items():
        print(f"  --{k} {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser('MoNODE Hyperparameter Search')

    # Required
    p.add_argument('--model',        required=True, choices=['node', 'vae'])
    p.add_argument('--segment_type', required=True,
                   choices=['atrial', 'ventricular', 'whole'])

    # Search control
    p.add_argument('--n_trials',   type=int,   default=50,
                   help='Number of trials THIS job should run. For SLURM job arrays, '
                        'set this per job (e.g. 10) and submit multiple jobs — they '
                        'share one Optuna study via --hp_output_dir.')
    p.add_argument('--n_gpus',     type=int,   default=1,
                   help='GPUs to use within this job (spawns one worker per GPU). '
                        'For Snellius: set to the number of GPUs in this SLURM job '
                        '(1–4 for gpu_a100/gpu_h100). Submit multiple SLURM jobs '
                        'to parallelise across jobs.')
    p.add_argument('--hp_epochs',  type=int,   default=100,
                   help='Training epochs per trial')
    p.add_argument('--fraction',   type=float, default=0.25,
                   help='Fraction of dataset to use (default 0.25)')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--report_only', action='store_true',
                   help='Skip optimisation — load the existing study and print results. '
                        'Run this after all parallel jobs have finished.')

    # I/O
    p.add_argument('--dataset_root',  type=str, default='/projects/prjs1890/',
                   help='Root directory of the MedalCare-XL dataset')
    p.add_argument('--hp_output_dir', type=str, default='results/hp_search',
                   help='Directory for trial checkpoints and the Optuna DB')

    # Fixed training hyper-params (same defaults as main.py)
    p.add_argument('--batch_size',      type=int,   default=25)
    p.add_argument('--lr',              type=float, default=0.002)
    p.add_argument('--ode_latent_dim',  type=int,   default=16)
    p.add_argument('--modulator_dim',   type=int,   default=0)
    p.add_argument('--content_dim',     type=int,   default=0)
    p.add_argument('--order',           type=int,   default=1)
    p.add_argument('--solver',          type=str,   default='euler')
    p.add_argument('--T_in',            type=int,   default=10)
    p.add_argument('--T_inv',           type=int,   default=100)
    p.add_argument('--Nincr',           type=int,   default=10)
    p.add_argument('--sobolev_weight',  type=float, default=0.0)
    p.add_argument('--l_w',             type=float, default=0.0)
    p.add_argument('--beta',            type=float, default=1.0)
    p.add_argument('--use_adjoint',     type=str,   default='no_adjoint')
    p.add_argument('--enc_H',           type=int,   default=50)
    p.add_argument('--dec_act',         type=str,   default='relu')
    p.add_argument('--cnn_filt_enc',    type=int,   default=16)
    p.add_argument('--cnn_filt_de',     type=int,   default=16)
    p.add_argument('--cnn_filt_inv',    type=int,   default=16)
    p.add_argument('--early_stopping_patience', type=int, default=0)

    # Default HP values (overridden by Optuna per trial)
    p.add_argument('--de_L',         type=int, default=2)
    p.add_argument('--de_H',         type=int, default=100)
    p.add_argument('--dec_L',        type=int, default=2)
    p.add_argument('--dec_H',        type=int, default=100)
    p.add_argument('--rnn_hidden',     type=int, default=10)
    p.add_argument('--rnn_hidden_dec', type=int, default=None)

    # Unused by HP search but required by model/data internals
    p.add_argument('--num_workers',    type=int,   default=0)
    p.add_argument('--shuffle',        type=eval,  default=True)
    p.add_argument('--noise',          type=float, default=None)
    p.add_argument('--Nobj',           type=int,   default=1)
    p.add_argument('--dt',             type=float, default=0.1)
    p.add_argument('--sonode_v',       type=str,   default='MLP')
    p.add_argument('--plot_every',     type=int,   default=99999)
    p.add_argument('--plotL',          type=int,   default=1)
    p.add_argument('--forecast_tr',    type=int,   default=2)
    p.add_argument('--forecast_vl',    type=int,   default=2)
    p.add_argument('--exp_id',         type=int,   default=0)
    p.add_argument('--data_root',      type=str,   default='data/')
    p.add_argument('--continue_training', type=eval, default=False)
    p.add_argument('--aladin_metadata_dir', type=str, default=None)

    args = p.parse_args()
    args.task    = 'ecg'
    args.dataset = 'MedalCare-XL'
    args.Nepoch  = args.hp_epochs
    # hp_output_dir is an absolute path relative to cwd; make it absolute
    args.hp_output_dir = os.path.abspath(args.hp_output_dir)
    args.save    = args.hp_output_dir  # placeholder; overridden per trial
    return args


def main():
    args = _parse_args()

    # ── Shared study identifiers (needed for both search and report) ──────────
    study_name  = f'hp_{args.model}_{args.segment_type}'
    storage_url = f'sqlite:///{os.path.join(args.hp_output_dir, study_name + ".db")}'

    # ── Report-only mode: print results from an existing study and exit ───────
    if args.report_only:
        if not os.path.exists(storage_url.replace('sqlite:///', '')):
            raise FileNotFoundError(
                f"No study DB found at {storage_url}. "
                "Run without --report_only first to create a study.")
        _report_results(study_name, storage_url, args)
        return

    # ── Load ECG config ───────────────────────────────────────────────────────
    import yaml
    cfg_path = os.path.join(_script_dir(), 'data', 'config.yml')
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    ecg_cfg = full_cfg['ecg']
    ecg_cfg['T'] = ecg_cfg['train']['T']

    # ── Resolve data_params path ──────────────────────────────────────────────
    orig_path = _original_params_path(args.segment_type)

    if args.fraction >= 1.0:
        # Use the full dataset — no sampling needed
        subset_path = orig_path
        if not os.path.exists(orig_path):
            print(f"Original data_params not found at:\n  {orig_path}")
            print("Generating from dataset directory…")
            sys.path.insert(0, _script_dir())
            from data.data_utils import get_data_params
            get_data_params(args.dataset_root, 'MedalCare-XL', 'median',
                            args.segment_type, 'ecg')
            if not os.path.exists(orig_path):
                raise FileNotFoundError(
                    f"Could not generate {orig_path}. "
                    "Check --dataset_root points to the MedalCare-XL root.")
            print(f"Generated → {orig_path}")
        print(f"Using full dataset (fraction=1): {orig_path}")
        _print_dist(orig_path)
    else:
        subset_path = _subset_params_path(args.segment_type, fraction=args.fraction)

        if not os.path.exists(subset_path):
            if not os.path.exists(orig_path):
                print(f"Original data_params not found at:\n  {orig_path}")
                print("Generating from dataset directory…")
                sys.path.insert(0, _script_dir())
                from data.data_utils import get_data_params
                get_data_params(args.dataset_root, 'MedalCare-XL', 'median',
                                args.segment_type, 'ecg')
                if not os.path.exists(orig_path):
                    raise FileNotFoundError(
                        f"Could not generate {orig_path}. "
                        "Check --dataset_root points to the MedalCare-XL root.")
                print(f"Generated → {orig_path}")

            pct = int(args.fraction * 100)
            print(f"\nGenerating {pct}% subset for segment='{args.segment_type}':")
            generate_subset_data_params(
                orig_path, subset_path, args.segment_type, args.fraction, args.seed)
        else:
            print(f"Subset data_params found: {subset_path}")
            _print_dist(subset_path)

    # ── Create / resume Optuna study ──────────────────────────────────────────
    # NOTE: hp_output_dir must be on a shared filesystem (e.g. /scratch or
    # /projects) when submitting multiple SLURM jobs so all workers share the
    # same SQLite DB.  Pass --hp_output_dir /scratch/<project>/hp_search.
    os.makedirs(args.hp_output_dir, exist_ok=True)

    n_startup = max(10, args.n_trials // 5)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='minimize',
        sampler=TPESampler(seed=args.seed, n_startup_trials=n_startup),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
        load_if_exists=True,
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nStudy '{study_name}' | {completed} completed trials | "
          f"this job: {args.n_trials} trials on {args.n_gpus} GPU(s)")

    # ── Launch workers (one per GPU within this job) ──────────────────────────
    n_gpus         = min(args.n_gpus, torch.cuda.device_count() or 1)
    trials_each    = max(1, args.n_trials // n_gpus)
    # Give the remainder to the first worker
    trials_per_gpu = [trials_each + (args.n_trials % n_gpus if i == 0 else 0)
                      for i in range(n_gpus)]

    if n_gpus > 1:
        mp.set_start_method('spawn', force=True)
        processes = []
        for rank in range(n_gpus):
            p = mp.Process(
                target=_worker,
                args=(rank, args, study_name, storage_url, subset_path,
                      trials_per_gpu[rank], ecg_cfg),
                daemon=False,
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        _worker(0, args, study_name, storage_url, subset_path,
                trials_per_gpu[0], ecg_cfg)

    # Each job prints a snapshot of results when it finishes.
    # Run with --report_only after all jobs complete for the final summary.
    _report_results(study_name, storage_url, args)


if __name__ == '__main__':
    main()
