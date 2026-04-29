#!/usr/bin/env python3
"""
analyse_results.py — Cross-model comparison of MoNODE run summaries.

Reads summary JSONs produced by summarize_results.py and generates:
  • Per-segment/dataset sub-directories, each with comparison plots
  • A root-level summary.md with bold-best tables

Input file naming (from summarize_results.py):
  {results_dir}/{segment_type}_{model}_{dataset_slug}.json

Output layout:
  {output_dir}/
    {segment}/
      {dataset}/
        mse_comparison.png
        classification_overview.png
        classification_perclass.png    (MedalCare-XL only)
        classification_other.png       (non-'class' clf params, if any)
        linear_probe_comparison.png
    summary.md

Usage:
  python analyse_results.py \\
      --results_dir results/summaries \\
      --output_dir  results/analysis
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

# ─── style ────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    'node':   '#2196F3',
    'vae':    '#F44336',
    'hbnode': '#4CAF50',
    'sonode': '#FF9800',
}
MODEL_LABELS = {
    'node':   'NODE',
    'vae':    'VAE',
    'hbnode': 'HBNODE',
    'sonode': 'SONODE',
}
SEGMENT_LABELS = {
    'atrial':       'Atrial',
    'ventricular':  'Ventricular',
    'whole':        'Whole',
    'all':          'All',
}

# Latent keys to exclude from all plots/tables
_SKIP_LKEYS = {'z0_sample', 'z0_sample_m'}

# Per-model preferred latent key: monode uses z0_m; everything else uses z0
_MODEL_PREFERRED_LKEY: dict[str, str] = {'monode': 'z0_m'}
_DEFAULT_LKEY = 'z0'

# MedalCare-XL regression params to exclude
_REG_SKIP_MEDALCARE = {'APD.v_d', 'APD.z_d'}

# UK Biobank params to exclude from linear probe regression plots
_REG_SKIP_UKBB = {'laef', 'lasv', 'lvef', 'lvgls', 'age'}

# ─── UK Biobank anatomy groupings ─────────────────────────────────────────────
_UKBB_ATRIAL_PARAMS      = {'laef', 'lasv', 'p_ax', 'pq_int', 'p_dur'}
_UKBB_VENTRICULAR_PARAMS = {'lv_mass', 'lvedv', 'lvef', 'lvgls',
                             'qrs_dur', 'qtc_int', 'r_ax', 'rvedv', 't_ax'}
_UKBB_BOTH_PARAMS        = {'age', 'rr_interval', 'pp_int'}
_UKBB_PARAM_RENAME       = {'heart_rate': 'rr_interval'}
_UKBB_ANATOMY_COLORS     = {
    'atrial':      '#BBDEFB',   # light blue
    'ventricular': '#FFCCBC',   # light orange
    'both':        '#E1BEE7',   # light purple
}

# Metrics shown in classification overview (balanced_accuracy excluded)
CLF_OVERVIEW_METRICS = [
    ('accuracy',   'Accuracy'),
    ('f1_macro',   'F1 (macro)'),
    ('auroc_macro','AUROC (macro)'),
]
# Preferred method names (first match wins)
CLF_METHOD_PREF  = ['linear', 'logistic', 'logistic_l2', 'logistic_l1', 'knn']
REG_METHOD_PREF  = ['ridge', 'ridgecv', 'linear', 'lasso', 'knn']

FIGSIZE_WIDE = (14, 5)
DPI          = 150
TITLE_FS     = 16
LABEL_FS     = 13
TICK_FS      = 11
LEGEND_FS    = 10

plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})


# ─── UK Biobank helpers ───────────────────────────────────────────────────────

def _is_ukbb(dataset: str) -> bool:
    d = dataset.lower()
    return 'ukb' in d or 'biobank' in d


def _anatomy_of(param: str) -> str | None:
    if param in _UKBB_ATRIAL_PARAMS:
        return 'atrial'
    if param in _UKBB_VENTRICULAR_PARAMS:
        return 'ventricular'
    if param in _UKBB_BOTH_PARAMS:
        return 'both'
    return None


def _apply_param_renames(params: list[str]) -> list[str]:
    return [_UKBB_PARAM_RENAME.get(p, p) for p in params]


_ANATOMY_ORDER = ['atrial', 'ventricular', 'both']


def _seg_allowed_params(seg: str) -> set[str] | None:
    """Return the set of UKBB param display-names allowed for *seg*.

    atrial      → atrial-specific + shared ('both') params
    ventricular → ventricular-specific + shared params
    anything else → None (no filter, show all)
    """
    if seg == 'atrial':
        return _UKBB_ATRIAL_PARAMS | _UKBB_BOTH_PARAMS
    if seg == 'ventricular':
        return _UKBB_VENTRICULAR_PARAMS | _UKBB_BOTH_PARAMS
    return None


def _sort_params_by_anatomy(params: list[str]) -> list[str]:
    """Sort params: atrial group first, then ventricular, then both, then unknown."""
    def _key(p: str):
        anat = _anatomy_of(p)
        order = _ANATOMY_ORDER.index(anat) if anat in _ANATOMY_ORDER else len(_ANATOMY_ORDER)
        return (order, p)
    return sorted(params, key=_key)


def _add_anatomy_shading(ax, params: list[str], seg: str = '') -> None:
    """Shade bar-chart background by anatomy.

    The segment's own anatomy group is highlighted (alpha=0.40); others are dimmed
    (alpha=0.10). When seg is neither 'atrial' nor 'ventricular' all groups use
    alpha=0.25. Two separate legends are attached: one for models, one for anatomy.
    """
    # Determine emphasis
    emphasis = seg if seg in ('atrial', 'ventricular') else None

    anat_patches: dict[str, Patch] = {}
    for i, p in enumerate(params):
        anat = _anatomy_of(p)
        if anat is None:
            continue
        color = _UKBB_ANATOMY_COLORS[anat]
        if emphasis is None:
            alpha = 0.25
        elif anat == emphasis:
            alpha = 0.40
        else:
            alpha = 0.10
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=alpha, zorder=0)
        if anat not in anat_patches:
            label = anat.capitalize() + (' ★' if anat == emphasis else '')
            anat_patches[anat] = Patch(facecolor=color, alpha=0.6, label=label)

    # Legend 1: models (bars already registered via _bar_group)
    model_h, model_l = ax.get_legend_handles_labels()
    leg1 = ax.legend(model_h, model_l,
                     fontsize=LEGEND_FS, framealpha=0.8,
                     loc='upper right', title='Model')

    # Legend 2: anatomy (separate artist, must re-add leg1 after)
    if anat_patches:
        ax.legend(list(anat_patches.values()), list(anat_patches.keys()),
                  fontsize=LEGEND_FS - 1, framealpha=0.8,
                  loc='upper left', title='Anatomy')
        ax.add_artist(leg1)  # ax.legend() replaced leg1; restore it


# ─── data loading + filtering ─────────────────────────────────────────────────

def _filter_probes(probes: dict, dataset: str) -> dict:
    """Apply display filters to the probes section of a summary JSON.

    Rules:
      - z0_sample / z0_sample_m latent keys excluded everywhere.
      - balanced_accuracy excluded from all classification metrics.
      - MSE excluded from all regression metrics.
      - MedalCare-XL classification: only the 'class' (pathology) param.
      - MedalCare-XL regression: APD.v_d and APD.z_d excluded.
      - Non-MedalCare classification: only 'accuracy' metric kept.
    """
    is_medalcare = 'medalcare' in dataset.lower()
    filtered: dict = {}

    for lkey, tasks in probes.items():
        if lkey in _SKIP_LKEYS:
            continue
        filtered[lkey] = {}

        # Classification
        clf = tasks.get('classification', {})
        if clf:
            filtered[lkey]['classification'] = {}
            for param, method_dict in clf.items():
                if is_medalcare and param != 'class':
                    continue
                filtered[lkey]['classification'][param] = {}
                for method, res in method_dict.items():
                    entry: dict = {}
                    if is_medalcare:
                        for k in ('accuracy', 'f1', 'f1_binary', 'f1_macro',
                                  'auroc', 'auroc_macro', 'recall_macro'):
                            if k in res:
                                entry[k] = res[k]
                        if 'per_class_accuracy' in res:
                            entry['per_class_accuracy'] = res['per_class_accuracy']
                    else:
                        for k in ('accuracy', 'recall_macro'):
                            if k in res:
                                entry[k] = res[k]
                    if entry:
                        filtered[lkey]['classification'][param][method] = entry

        # Regression
        reg = tasks.get('regression', {})
        if reg:
            filtered[lkey]['regression'] = {}
            for param, method_dict in reg.items():
                if is_medalcare and param in _REG_SKIP_MEDALCARE:
                    continue
                if _is_ukbb(dataset) and param in _REG_SKIP_UKBB:
                    continue
                filtered[lkey]['regression'][param] = {}
                for method, res in method_dict.items():
                    entry = {k: res[k] for k in ('r2', 'r2_ci', 'mae', 'mae_ci') if k in res}
                    if entry:
                        filtered[lkey]['regression'][param][method] = entry

    return filtered


def _keep_preferred_lkey(section: dict, model: str) -> dict:
    """Keep only the preferred latent key for *model*, renamed to _DEFAULT_LKEY.

    monode → selects z0_m, stored under 'z0'
    all others → selects z0, stored under 'z0'

    Renaming to the same key means all models share one row in every plot,
    so monode(z0_m) is compared directly with node/vae/hbnode(z0).
    Falls back to the original section if the preferred key is absent.
    """
    preferred = _MODEL_PREFERRED_LKEY.get(model, _DEFAULT_LKEY)
    if preferred in section:
        return {_DEFAULT_LKEY: section[preferred]}
    return section


def load_summaries(results_dir: str) -> dict:
    """Return {(segment, model, dataset): summary_dict} with filtered probes."""
    out = {}
    for fpath in sorted(Path(results_dir).glob('*.json')):
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [warn] Could not read {fpath}: {e}")
            continue
        seg     = data.get('segment_type') or 'all'
        model   = data.get('model', 'unknown')
        dataset = data.get('dataset', 'unknown')
        if 'probes' in data:
            data['probes'] = _filter_probes(data['probes'], dataset)
            data['probes'] = _keep_preferred_lkey(data['probes'], model)
        is_ukbb_data = _is_ukbb(dataset)
        if 'label_efficiency' in data:
            le = {lk: v for lk, v in data['label_efficiency'].items()
                  if lk not in _SKIP_LKEYS}
            if is_ukbb_data:
                le = {
                    lk: {
                        frac_str: {
                            task: {p: stats for p, stats in task_data.items()
                                   if p not in _REG_SKIP_UKBB}
                            for task, task_data in frac_data.items()
                        }
                        for frac_str, frac_data in lk_data.items()
                    }
                    for lk, lk_data in le.items()
                }
            data['label_efficiency'] = _keep_preferred_lkey(le, model)
        if 'pearson' in data:
            pearson = {lk: v for lk, v in data['pearson'].items()
                       if lk not in _SKIP_LKEYS}
            if is_ukbb_data:
                pearson = {
                    lk: {p: info for p, info in param_dict.items()
                         if p not in _REG_SKIP_UKBB}
                    for lk, param_dict in pearson.items()
                }
            data['pearson'] = _keep_preferred_lkey(pearson, model)
        if 'permutation_test' in data:
            perm = {lk: v for lk, v in data['permutation_test'].items()
                    if lk not in _SKIP_LKEYS}
            data['permutation_test'] = _keep_preferred_lkey(perm, model)
        out[(seg, model, dataset)] = data
    return out


def group_summaries(summaries: dict) -> dict:
    """Return {segment: {dataset: {model: summary}}}."""
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for (seg, model, dataset), summary in summaries.items():
        grouped[seg][dataset][model] = summary
    return grouped


# ─── small helpers ────────────────────────────────────────────────────────────

def _color(model: str) -> str:
    return MODEL_COLORS.get(model, '#9E9E9E')


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model.upper())


def _prefer_method(available: list[str], pref: list[str]) -> str:
    for m in pref:
        if m in available:
            return m
    return available[0] if available else ''


def _bar_group(ax, groups: list[str], models: list[str],
               values: dict[str, list[float | None]],
               ylabel: str = '', title: str = '',
               ylim: tuple | None = None, legend: bool = True,
               errors: dict[str, list[tuple[float, float] | None]] | None = None) -> None:
    """Draw a grouped bar chart.

    groups : x-axis tick labels
    models : one bar per model (one colour each)
    values : {model: [value_per_group]}  — None → missing bar
    errors : {model: [(lower_err, upper_err) | None]}  — asymmetric CI half-widths
    """
    n_groups = len(groups)
    n_models = len(models)
    bar_w    = 0.7 / max(n_models, 1)
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    for i, model in enumerate(models):
        vals  = values.get(model, [None] * n_groups)
        errs  = errors.get(model, [None] * n_groups) if errors else [None] * n_groups
        xs    = np.arange(n_groups) + offsets[i]
        ys    = [v if v is not None else 0.0 for v in vals]

        # Build asymmetric yerr arrays (2 × n_groups) where available
        has_err = any(e is not None for e in errs)
        if has_err:
            lo_arr = [e[0] if e is not None else 0.0 for e in errs]
            hi_arr = [e[1] if e is not None else 0.0 for e in errs]
            yerr = np.array([lo_arr, hi_arr])
        else:
            yerr = None

        bars = ax.bar(xs, ys, width=bar_w * 0.9,
                      color=_color(model), label=_label(model), alpha=0.85,
                      yerr=yerr, capsize=3,
                      error_kw={'elinewidth': 1.0, 'ecolor': 'black', 'alpha': 0.7})
        # annotate
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=7, rotation=45)

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(groups, fontsize=TICK_FS, rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=TICK_FS)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    if title:
        ax.set_title(title, fontsize=LABEL_FS, fontweight='bold')
    if ylim:
        ax.set_ylim(*ylim)
    if legend and n_models > 1:
        ax.legend(fontsize=LEGEND_FS, framealpha=0.8)


# ─── MSE comparison ───────────────────────────────────────────────────────────

def plot_mse(model_data: dict[str, dict], out_path: str, seg: str, dataset: str) -> None:
    """One figure: val MSE (left) + test MSE per horizon (right)."""
    models = sorted(model_data.keys())

    # Gather val MSE
    val_mses: dict[str, float | None] = {}
    for m in models:
        tr = model_data[m].get('training', {})
        val_mses[m] = tr.get('best_val_mse')

    # Gather test MSE — horizons may differ; take union
    horizon_sets = []
    for m in models:
        tr = model_data[m].get('training', {})
        hs = [h for h in tr.get('test_mse', {}).keys()]
        horizon_sets.append(set(hs))
    all_horizons = sorted(set().union(*horizon_sets), key=lambda x: float(x))

    test_mses: dict[str, list[float | None]] = {m: [] for m in models}
    for h in all_horizons:
        for m in models:
            tr = model_data[m].get('training', {})
            entry = tr.get('test_mse', {}).get(h)
            test_mses[m].append(entry['mean'] if entry else None)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    fig.suptitle(
        f'MSE Comparison — {SEGMENT_LABELS.get(seg, seg)} | {dataset}',
        fontsize=TITLE_FS, fontweight='bold', y=1.01
    )

    # left: val MSE
    _bar_group(
        axes[0],
        groups=['Validation MSE'],
        models=models,
        values={m: [val_mses[m]] for m in models},
        ylabel='MSE',
        title='Validation MSE',
    )

    # right: test MSE per horizon
    if all_horizons:
        _bar_group(
            axes[1],
            groups=[f'H={h}' for h in all_horizons],
            models=models,
            values=test_mses,
            ylabel='MSE',
            title='Test MSE per Horizon',
        )
    else:
        axes[1].text(0.5, 0.5, 'No test MSE data', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=LABEL_FS)
        axes[1].set_title('Test MSE per Horizon', fontsize=LABEL_FS, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─── classification helpers ───────────────────────────────────────────────────

def _collect_clf_data(
    model_data: dict[str, dict],
    param: str,
    method_pref: list[str] = CLF_METHOD_PREF,
) -> tuple[dict, dict, list[str], str | None]:
    """Extract per-latent-key classification results for a given param.

    Returns:
        overview  : {lkey: {metric_key: {model: value}}}
        per_class : {lkey: {class_name: {model: value}}}
        all_lkeys : sorted list of latent keys across all models
        chosen_method: name of the method actually used
    """
    overview:   dict = defaultdict(lambda: defaultdict(dict))
    per_class:  dict = defaultdict(lambda: defaultdict(dict))
    lkey_set:   set  = set()
    chosen_meth: str | None = None

    for model, summary in model_data.items():
        probes = summary.get('probes', {})
        for lkey, ldata in probes.items():
            clf = ldata.get('classification', {})
            if param not in clf:
                continue
            method_dict = clf[param]
            meth = _prefer_method(list(method_dict.keys()), method_pref)
            if not meth:
                continue
            if chosen_meth is None:
                chosen_meth = meth
            res = method_dict[meth]

            lkey_set.add(lkey)
            for mk, ml in CLF_OVERVIEW_METRICS:
                # For 'accuracy', prefer recall_macro (balanced accuracy over classes)
                if mk == 'accuracy':
                    v = res.get('recall_macro') if res.get('recall_macro') is not None else res.get('accuracy')
                else:
                    v = res.get(mk)
                if v is not None:
                    overview[lkey][mk][model] = float(v)

            for cls_name, acc in res.get('per_class_accuracy', {}).items():
                per_class[lkey][str(cls_name)][model] = float(acc)

    return dict(overview), dict(per_class), sorted(lkey_set), chosen_meth


def _collect_clf_params(model_data: dict[str, dict]) -> set[str]:
    params: set = set()
    for summary in model_data.values():
        for ldata in summary.get('probes', {}).values():
            params.update(ldata.get('classification', {}).keys())
    return params


# ─── Classification overview plot ─────────────────────────────────────────────

def plot_clf_overview(
    model_data: dict[str, dict],
    out_path: str,
    seg: str,
    dataset: str,
    param: str = 'class',
    title_suffix: str = '',
) -> None:
    """Grid: rows=latent_keys, cols=4 overview metrics. Bars=models."""
    overview, _, lkeys, method = _collect_clf_data(model_data, param)
    models = sorted(model_data.keys())

    if not lkeys:
        print(f"    [skip] No classification data for param='{param}'")
        return

    n_rows = len(lkeys)
    n_cols = len(CLF_OVERVIEW_METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.5 + 1),
                             squeeze=False)
    fig.suptitle(
        f'Classification — {SEGMENT_LABELS.get(seg, seg)} | {dataset}'
        f' | param: {param}{title_suffix}'
        + (f'\n(method: {method})' if method else ''),
        fontsize=TITLE_FS, fontweight='bold', y=1.01
    )

    for r, lkey in enumerate(lkeys):
        for c, (mk, ml) in enumerate(CLF_OVERVIEW_METRICS):
            ax = axes[r][c]
            metric_vals = overview.get(lkey, {}).get(mk, {})
            values_dict = {m: [metric_vals.get(m)] for m in models}
            _bar_group(
                ax,
                groups=[''],
                models=models,
                values=values_dict,
                ylabel=ml if c == 0 else '',
                title=ml if r == 0 else '',
                ylim=(0, 1.15),
                legend=(r == 0 and c == n_cols - 1),
            )
            if c == 0:
                ax.set_ylabel(f'{lkey}\n{ml}', fontsize=LABEL_FS)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─── Per-class accuracy plot ──────────────────────────────────────────────────

def plot_clf_perclass(
    model_data: dict[str, dict],
    out_path: str,
    seg: str,
    dataset: str,
    param: str = 'class',
) -> None:
    """Per-class accuracy: rows=latent_keys, one grouped bar per class."""
    _, per_class, lkeys, method = _collect_clf_data(model_data, param)
    models = sorted(model_data.keys())

    if not lkeys or not any(per_class.get(lk) for lk in lkeys):
        print(f"    [skip] No per-class data for param='{param}'")
        return

    n_rows = len(lkeys)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(max(12, len(next(iter(per_class.values()), {})) * 1.8), n_rows * 4),
                             squeeze=False)
    fig.suptitle(
        f'Per-Class Accuracy — {SEGMENT_LABELS.get(seg, seg)} | {dataset} | param: {param}'
        + (f'\n(method: {method})' if method else ''),
        fontsize=TITLE_FS, fontweight='bold', y=1.01
    )

    for r, lkey in enumerate(lkeys):
        ax = axes[r][0]
        classes = sorted(per_class.get(lkey, {}).keys())
        if not classes:
            ax.set_visible(False)
            continue
        values_dict = {
            m: [per_class[lkey].get(cls, {}).get(m) for cls in classes]
            for m in models
        }
        _bar_group(
            ax,
            groups=classes,
            models=models,
            values=values_dict,
            ylabel='Accuracy',
            title=f'Latent: {lkey}',
            ylim=(0, 1.2),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─── Other classification params ──────────────────────────────────────────────

def plot_clf_other(
    model_data: dict[str, dict],
    out_path: str,
    seg: str,
    dataset: str,
    other_params: list[str],
) -> None:
    """One row per (other_param × latent_key), 4 cols of overview metrics."""
    models = sorted(model_data.keys())

    rows: list[tuple[str, str]] = []  # (param, lkey)
    overviews: dict[tuple[str, str], dict] = {}

    for param in other_params:
        overview, _, lkeys, _ = _collect_clf_data(model_data, param)
        for lkey in lkeys:
            rows.append((param, lkey))
            overviews[(param, lkey)] = overview.get(lkey, {})

    if not rows:
        return

    n_cols = len(CLF_OVERVIEW_METRICS)
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(n_cols * 3.5, len(rows) * 3.5 + 1),
                             squeeze=False)
    fig.suptitle(
        f'Other Classification Params — {SEGMENT_LABELS.get(seg, seg)} | {dataset}',
        fontsize=TITLE_FS, fontweight='bold', y=1.01
    )

    for r, (param, lkey) in enumerate(rows):
        for c, (mk, ml) in enumerate(CLF_OVERVIEW_METRICS):
            ax = axes[r][c]
            metric_vals = overviews.get((param, lkey), {}).get(mk, {})
            values_dict = {m: [metric_vals.get(m)] for m in models}
            _bar_group(
                ax,
                groups=[''],
                models=models,
                values=values_dict,
                ylabel='',
                title=ml if r == 0 else '',
                ylim=(0, 1.15),
                legend=(r == 0 and c == n_cols - 1),
            )
            if c == 0:
                ax.set_ylabel(f'{param} / {lkey}\n{ml}', fontsize=LABEL_FS - 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─── Linear probe (regression) plot ───────────────────────────────────────────

def plot_linear_probe(
    model_data: dict[str, dict],
    out_path: str,
    seg: str,
    dataset: str,
) -> None:
    """Regression probes: rows=latent_keys, cols=R² and MAE, x=param."""
    models = sorted(model_data.keys())

    # Collect: {lkey: {param: {model: {r2, mae}}}}
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    lkey_set: set = set()
    param_set: set = set()

    for model, summary in model_data.items():
        for lkey, ldata in summary.get('probes', {}).items():
            reg = ldata.get('regression', {})
            for param, method_dict in reg.items():
                meth = _prefer_method(list(method_dict.keys()), REG_METHOD_PREF)
                if not meth:
                    continue
                res = method_dict[meth]
                if res.get('r2') is not None:
                    data[lkey][param][model]['r2'] = float(res['r2'])
                if res.get('mae') is not None:
                    data[lkey][param][model]['mae'] = float(res['mae'])
                if res.get('r2_ci') is not None:
                    data[lkey][param][model]['r2_ci'] = res['r2_ci']
                if res.get('mae_ci') is not None:
                    data[lkey][param][model]['mae_ci'] = res['mae_ci']
                lkey_set.add(lkey)
                param_set.add(param)

    if not lkey_set:
        print(f"    [skip] No regression probe data")
        return

    if _is_ukbb(dataset):
        data = {lk: {_UKBB_PARAM_RENAME.get(p, p): v for p, v in pd.items()}
                for lk, pd in data.items()}
        param_set = {_UKBB_PARAM_RENAME.get(p, p) for p in param_set}

    lkeys  = sorted(lkey_set)
    params = (_sort_params_by_anatomy(list(param_set))
              if _is_ukbb(dataset) else sorted(param_set))
    n_rows = len(lkeys)
    has_mae = any(
        data[lk].get(p, {}).get(m, {}).get('mae') is not None
        for lk in lkeys for p in params for m in models
    )
    n_cols = 2 if has_mae else 1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(10, len(params) * 2.5) * n_cols, n_rows * 3.5 + 1),
                             squeeze=False)
    fig.suptitle(
        f'Linear Probe (Regression) — {SEGMENT_LABELS.get(seg, seg)} | {dataset}',
        fontsize=TITLE_FS, fontweight='bold', y=1.01
    )

    def _ci_to_err(val, ci):
        """Convert [lo_bound, hi_bound] CI to (lo_half, hi_half) error bar."""
        if val is None or ci is None:
            return None
        return (max(0.0, float(val) - float(ci[0])),
                max(0.0, float(ci[1]) - float(val)))

    for r, lkey in enumerate(lkeys):
        r2_vals = {m: [data[lkey].get(p, {}).get(m, {}).get('r2') for p in params] for m in models}
        r2_errs = {
            m: [_ci_to_err(data[lkey].get(p, {}).get(m, {}).get('r2'),
                           data[lkey].get(p, {}).get(m, {}).get('r2_ci'))
                for p in params]
            for m in models
        }
        _bar_group(axes[r][0], groups=params, models=models,
                   values=r2_vals, ylabel=f'{lkey}\nR²',
                   title='R²' if r == 0 else '',
                   errors=r2_errs)
        if _is_ukbb(dataset):
            _add_anatomy_shading(axes[r][0], params, seg=seg)

        if has_mae:
            mae_vals = {m: [data[lkey].get(p, {}).get(m, {}).get('mae') for p in params] for m in models}
            mae_errs = {
                m: [_ci_to_err(data[lkey].get(p, {}).get(m, {}).get('mae'),
                               data[lkey].get(p, {}).get(m, {}).get('mae_ci'))
                    for p in params]
                for m in models
            }
            _bar_group(axes[r][1], groups=params, models=models,
                       values=mae_vals, ylabel=f'{lkey}\nMAE',
                       title='MAE' if r == 0 else '',
                       errors=mae_errs)
            if _is_ukbb(dataset):
                _add_anatomy_shading(axes[r][1], params, seg=seg)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─── Linear probe diff plot ───────────────────────────────────────────────────

def plot_linear_probe_diff(
    model_data: dict[str, dict],
    out_path: str,
    seg: str,
    dataset: str,
    ref_model: str = 'vae',
) -> None:
    """Diverging bar chart: R²(model) − R²(ref_model). Positive = model better."""
    models = sorted(model_data.keys())
    if ref_model not in models:
        print(f"    [skip] ref_model='{ref_model}' not in data; skipping diff plot")
        return
    compare_models = [m for m in models if m != ref_model]
    if not compare_models:
        return

    data: dict = defaultdict(lambda: defaultdict(dict))
    lkey_set: set = set()
    param_set: set = set()

    for model, summary in model_data.items():
        for lkey, ldata in summary.get('probes', {}).items():
            for param, method_dict in ldata.get('regression', {}).items():
                meth = _prefer_method(list(method_dict.keys()), REG_METHOD_PREF)
                if not meth:
                    continue
                r2 = method_dict[meth].get('r2')
                if r2 is not None:
                    data[lkey][param][model] = float(r2)
                    lkey_set.add(lkey)
                    param_set.add(param)

    if not lkey_set:
        return

    if _is_ukbb(dataset):
        data = {lkey: {_UKBB_PARAM_RENAME.get(p, p): v for p, v in pdata.items()}
                for lkey, pdata in data.items()}
        param_set = {_UKBB_PARAM_RENAME.get(p, p) for p in param_set}

    lkeys  = sorted(lkey_set)
    params = (_sort_params_by_anatomy(list(param_set))
              if _is_ukbb(dataset) else sorted(param_set))
    n_rows = len(lkeys)
    n_cmp  = len(compare_models)
    bar_w  = 0.7 / max(n_cmp, 1)
    offsets = np.linspace(-(n_cmp - 1) / 2, (n_cmp - 1) / 2, n_cmp) * bar_w

    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(max(10, len(params) * 2.0), n_rows * 3.5 + 1),
                             squeeze=False)
    fig.suptitle(
        f'R² Difference vs {_label(ref_model)} — {SEGMENT_LABELS.get(seg, seg)} | {dataset}',
        fontsize=TITLE_FS, fontweight='bold', y=1.01
    )

    for r, lkey in enumerate(lkeys):
        ax = axes[r][0]
        for i, model in enumerate(compare_models):
            diffs = []
            for p in params:
                r2_m   = data[lkey].get(p, {}).get(model)
                r2_ref = data[lkey].get(p, {}).get(ref_model)
                diffs.append(r2_m - r2_ref if (r2_m is not None and r2_ref is not None) else None)

            xs   = np.arange(len(params)) + offsets[i]
            ys   = [d if d is not None else 0.0 for d in diffs]
            bars = ax.bar(xs, ys, width=bar_w * 0.9,
                          color=_color(model), label=_label(model), alpha=0.85)
            for bar, d in zip(bars, diffs):
                if d is not None and abs(d) > 0.001:
                    va  = 'bottom' if d >= 0 else 'top'
                    off = 0.003 if d >= 0 else -0.003
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + off,
                            f'{d:+.3f}', ha='center', va=va,
                            fontsize=7, rotation=45)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(np.arange(len(params)))
        ax.set_xticklabels(params, fontsize=TICK_FS, rotation=30, ha='right')
        ax.tick_params(axis='y', labelsize=TICK_FS)
        ax.set_ylabel(f'{lkey}\nΔR²', fontsize=LABEL_FS)
        if r == 0:
            ax.set_title(f'Positive = better than {_label(ref_model)}',
                         fontsize=LABEL_FS, fontweight='bold')
        if n_cmp > 1:
            ax.legend(fontsize=LEGEND_FS, framealpha=0.8)
        if _is_ukbb(dataset):
            _add_anatomy_shading(ax, params, seg=seg)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─── Label efficiency plot ────────────────────────────────────────────────────

def plot_label_efficiency(
    model_data: dict[str, dict],
    out_path: str,
    seg: str,
    dataset: str,
) -> None:
    """Learning curves: training fraction vs mean R² (± std) per param.

    Each model gets its own line with a shaded ±1 std band.
    Data comes from the mean/std aggregated across seeds saved in
    label_efficiency_summary.json.
    """
    def _frac_float(s: str) -> float:
        return float(s.rstrip('%')) / 100

    allowed = _seg_allowed_params(seg) if _is_ukbb(dataset) else None

    # Collect: {lkey: {param: {model: {frac_float: (mean, std)}}}}
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    lkey_set: set = set()
    param_set: set = set()

    for model, summary in model_data.items():
        for lkey, frac_dict in summary.get('label_efficiency', {}).items():
            if lkey in _SKIP_LKEYS:
                continue
            lkey_set.add(lkey)
            for frac_str in _LE_FRAC_ORDER:
                agg = frac_dict.get(frac_str, {}).get('regression', {})
                for param, stats in agg.items():
                    mean = stats.get('r2_mean')
                    std  = stats.get('r2_std', 0.0)
                    if mean is not None:
                        disp = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                        if allowed is not None and disp not in allowed:
                            continue
                        data[lkey][disp][model][_frac_float(frac_str)] = (mean, std)
                        param_set.add(disp)

    if not lkey_set or not param_set:
        print(f"    [skip] No label efficiency data")
        return

    lkeys  = sorted(lkey_set - _SKIP_LKEYS)
    params = (_sort_params_by_anatomy(list(param_set))
              if _is_ukbb(dataset) else sorted(param_set))
    models = sorted(model_data.keys())
    fracs  = [_frac_float(f) for f in _LE_FRAC_ORDER]

    n_rows = len(lkeys)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(max(8, len(params) * 0.8 + 3), n_rows * 4 + 1),
                             squeeze=False)
    fig.suptitle(
        f'Label Efficiency (OLS, mean±std) — {SEGMENT_LABELS.get(seg, seg)} | {dataset}',
        fontsize=TITLE_FS, fontweight='bold', y=1.01,
    )

    for r, lkey in enumerate(lkeys):
        ax = axes[r][0]
        line_idx = 0
        for model in models:
            for param in params:
                pts = data[lkey].get(param, {}).get(model, {})
                if not pts:
                    continue
                xs   = sorted(pts.keys())
                ys   = np.array([pts[x][0] for x in xs])
                errs = np.array([pts[x][1] for x in xs])
                color = f'C{line_idx % 10}'
                ax.plot(xs, ys, marker='o', color=color,
                        label=f'{_label(model)} / {param}',
                        linewidth=1.5, markersize=5)
                ax.fill_between(xs, ys - errs, ys + errs, alpha=0.15, color=color)
                line_idx += 1

        ax.set_xscale('log')
        ax.set_xticks(fracs)
        ax.set_xticklabels([f'{int(f*100)}%' for f in fracs], fontsize=TICK_FS)
        ax.set_xlabel('Training fraction', fontsize=LABEL_FS)
        ax.set_ylabel(f'{lkey}\nR²', fontsize=LABEL_FS)
        ax.set_title('R² vs training fraction (OLS)' if r == 0 else '',
                     fontsize=LABEL_FS, fontweight='bold')
        ax.tick_params(axis='y', labelsize=TICK_FS)
        ax.spines[['top', 'right']].set_visible(False)
        if line_idx <= 12:
            ax.legend(fontsize=max(6, LEGEND_FS - 2), framealpha=0.8,
                      bbox_to_anchor=(1, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─── Pearson max-abs-r comparison plot ────────────────────────────────────────

def plot_pearson_comparison(
    model_data: dict[str, dict],
    out_path: str,
    seg: str,
    dataset: str,
) -> None:
    """Grouped bar chart: largest |Pearson r| per param per model, one row per latent key."""
    # Collect: {lkey: {param: {model: max_abs_r}}}
    data: dict = defaultdict(lambda: defaultdict(dict))
    lkey_set: set = set()
    param_set: set = set()

    for model, summary in model_data.items():
        for lkey, param_dict in summary.get('pearson', {}).items():
            lkey_set.add(lkey)
            for param, info in param_dict.items():
                max_r = info.get('max_abs_r')
                if max_r is not None:
                    disp_param = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                    data[lkey][disp_param][model] = float(max_r)
                    param_set.add(disp_param)

    if not lkey_set or not param_set:
        print(f"    [skip] No Pearson correlation data")
        return

    lkeys  = sorted(lkey_set)
    params = (_sort_params_by_anatomy(list(param_set))
              if _is_ukbb(dataset) else sorted(param_set))
    models = sorted(model_data.keys())
    n_rows = len(lkeys)

    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(max(10, len(params) * 2.0), n_rows * 3.5 + 1),
                             squeeze=False)
    fig.suptitle(
        f'Pearson Correlation (max |r| per param) — {SEGMENT_LABELS.get(seg, seg)} | {dataset}',
        fontsize=TITLE_FS, fontweight='bold', y=1.01,
    )

    for r, lkey in enumerate(lkeys):
        values = {m: [data[lkey].get(p, {}).get(m) for p in params] for m in models}
        _bar_group(axes[r][0], groups=params, models=models,
                   values=values, ylabel=f'{lkey}\nmax |r|',
                   title='max |Pearson r|' if r == 0 else '',
                   ylim=(0, 1.05))
        if _is_ukbb(dataset):
            _add_anatomy_shading(axes[r][0], params, seg=seg)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ─── UK Biobank label-efficiency focused plots ────────────────────────────────

_LE_FRAC_ORDER  = ['1%', '10%', '50%', '100%']
_LE_MODELS      = ('node', 'vae', 'monode')   # monode uses z0_m (renamed to z0)


def _collect_le_data(model_data: dict[str, dict],
                     dataset: str,
                     seg: str = '') -> dict:
    """Extract label efficiency R² mean/std from summaries.

    Returns:
      {lkey: {param_disp: {model: {frac_str: (mean, std)}}}}
    Only models in _LE_MODELS that are present in model_data are included.
    For UKBB, params are filtered to those appropriate for *seg*.
    """
    out: dict = {}
    models_present = [m for m in _LE_MODELS if m in model_data]
    allowed = _seg_allowed_params(seg) if _is_ukbb(dataset) else None

    for model in models_present:
        for lkey, frac_dict in model_data[model].get('label_efficiency', {}).items():
            if lkey in _SKIP_LKEYS:
                continue
            out.setdefault(lkey, {})
            for frac_str in _LE_FRAC_ORDER:
                agg = frac_dict.get(frac_str, {}).get('regression', {})
                for param, stats in agg.items():
                    mean = stats.get('r2_mean')
                    std  = stats.get('r2_std', 0.0)
                    if mean is None:
                        continue
                    disp = (_UKBB_PARAM_RENAME.get(param, param)
                            if _is_ukbb(dataset) else param)
                    if allowed is not None and disp not in allowed:
                        continue
                    out[lkey].setdefault(disp, {}).setdefault(model, {})[frac_str] = (
                        float(mean), float(std))
    return out


def plot_le_bars_by_fraction(
    model_data: dict[str, dict],
    out_dir: str,
    seg: str,
    dataset: str,
) -> None:
    """Bar chart: x = phenotype parameters, y = R²  mean ± std.

    One figure per latent key, 2×2 grid of subplots (one per label-efficiency
    fraction).  Bars are grouped by model (node / vae).  Only produced for
    UK Biobank × (atrial | ventricular).
    """
    le_data = _collect_le_data(model_data, dataset, seg=seg)
    if not le_data:
        print(f"    [skip] No label efficiency data for le_bars_by_fraction")
        return

    models_present = [m for m in _LE_MODELS if m in model_data]
    bar_w = 0.35
    offsets = np.linspace(-(len(models_present) - 1) / 2,
                          (len(models_present) - 1) / 2,
                          len(models_present)) * bar_w

    for lkey, param_dict in le_data.items():
        params = (_sort_params_by_anatomy(list(param_dict.keys()))
                  if _is_ukbb(dataset) else sorted(param_dict.keys()))
        if not params:
            continue

        n_frac = len(_LE_FRAC_ORDER)
        ncols  = 2
        nrows  = (n_frac + 1) // 2
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(max(10, len(params) * 1.4) * ncols / 2,
                                          nrows * 4 + 1),
                                 squeeze=False)
        fig.suptitle(
            f'Label Efficiency — R² by Fraction\n'
            f'{SEGMENT_LABELS.get(seg, seg)} | {dataset} | latent: {lkey}',
            fontsize=TITLE_FS, fontweight='bold', y=1.02,
        )

        xs = np.arange(len(params))
        for fi, frac_str in enumerate(_LE_FRAC_ORDER):
            ax = axes[fi // ncols][fi % ncols]
            
            for mi, model in enumerate(models_present):
                means = []
                stds  = []
                for p in params:
                    entry = param_dict.get(p, {}).get(model, {}).get(frac_str)
                    if entry is not None:
                        means.append(entry[0])
                        stds.append(entry[1])
                    else:
                        means.append(0.0)
                        stds.append(0.0)

                bars = ax.bar(xs + offsets[mi], means, bar_w * 0.9,
                              yerr=stds, capsize=3,
                              color=_color(model), alpha=0.85,
                              label=_label(model),
                              error_kw={'elinewidth': 1, 'alpha': 0.7})
                for bar, m_val in zip(bars, means):
                    if m_val > 0.01:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.005,
                                f'{m_val:.2f}', ha='center', va='bottom',
                                fontsize=6, rotation=45)

            ax.set_xticks(xs)
            ax.set_xticklabels(params, rotation=40, ha='right', fontsize=TICK_FS - 1)
            ax.set_ylabel('R²', fontsize=LABEL_FS)
            ax.set_title(f'Training fraction: {frac_str}',
                         fontsize=LABEL_FS, fontweight='bold')
            ax.set_ylim(bottom=-1, top=1)
            ax.tick_params(axis='y', labelsize=TICK_FS)
            ax.spines[['top', 'right']].set_visible(False)
            if _is_ukbb(dataset):
                _add_anatomy_shading(ax, params, seg=seg)
            else:
                ax.legend(fontsize=LEGEND_FS, framealpha=0.8)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'le_bars_by_fraction_{lkey}.png')
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fname}")


def plot_le_curves_by_param(
    model_data: dict[str, dict],
    out_dir: str,
    seg: str,
    dataset: str,
) -> None:
    """Learning curves: one subplot per phenotype parameter.

    x = training fraction (log scale), y = R² mean ± std shading.
    Lines: node (blue) vs vae (red).  One figure per latent key.
    Only produced for UK Biobank × (atrial | ventricular).
    """
    le_data = _collect_le_data(model_data, dataset, seg=seg)
    if not le_data:
        print(f"    [skip] No label efficiency data for le_curves_by_param")
        return

    models_present = [m for m in _LE_MODELS if m in model_data]
    frac_vals = [float(f.rstrip('%')) / 100 for f in _LE_FRAC_ORDER]

    for lkey, param_dict in le_data.items():
        params = (_sort_params_by_anatomy(list(param_dict.keys()))
                  if _is_ukbb(dataset) else sorted(param_dict.keys()))
        if not params:
            continue

        ncols = min(4, len(params))
        nrows = (len(params) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 3.5 + 0.8),
                                 squeeze=False)
        fig.suptitle(
            f'Label Efficiency — R² per Parameter\n'
            f'{SEGMENT_LABELS.get(seg, seg)} | {dataset} | latent: {lkey}',
            fontsize=TITLE_FS, fontweight='bold', y=1.02,
        )

        for pi, param in enumerate(params):
            ax = axes[pi // ncols][pi % ncols]
            
            # Initialize tracking variables for the data bounds in this specific subplot
            param_min_y = float('inf')
            param_max_y = float('-inf')
            
            for model in models_present:
                model_data_p = param_dict.get(param, {}).get(model, {})
                xs, ys, es = [], [], []
                for frac_str, fv in zip(_LE_FRAC_ORDER, frac_vals):
                    entry = model_data_p.get(frac_str)
                    if entry is not None:
                        xs.append(fv)
                        ys.append(entry[0])
                        es.append(entry[1])
                if not xs:
                    continue
                xs_arr = np.array(xs)
                ys_arr = np.array(ys)
                es_arr = np.array(es)
                
                # Update the global min/max for this subplot (including error shaded regions)
                param_min_y = min(param_min_y, np.min(ys_arr - es_arr))
                param_max_y = max(param_max_y, np.max(ys_arr + es_arr))
                
                ax.plot(xs_arr, ys_arr, marker='o', color=_color(model),
                        label=_label(model), linewidth=1.8, markersize=5)
                ax.fill_between(xs_arr, ys_arr - es_arr, ys_arr + es_arr,
                                alpha=0.15, color=_color(model))

            ax.set_xscale('log')
            ax.set_xticks(frac_vals)
            ax.set_xticklabels([f'{int(f*100)}%' for f in frac_vals], fontsize=TICK_FS - 1)
            ax.set_xlabel('Training fraction', fontsize=LABEL_FS - 1)
            ax.set_ylabel('R²', fontsize=LABEL_FS - 1)
            
            # Dynamically set ylim based on the data limits or default to [-1, 1]
            if param_min_y != float('inf') and param_max_y != float('-inf'):
                ax.set_ylim(bottom=max(-1.0, param_min_y), top=min(1.0, param_max_y))
            else:
                # Fallback just in case a subplot has completely empty data
                ax.set_ylim(bottom=-1.0, top=1.0)

            # anatomy-aware title colour
            anat = _anatomy_of(param) if _is_ukbb(dataset) else None
            title_color = (_UKBB_ANATOMY_COLORS.get(anat, 'black')
                           if anat else 'black')
            ax.set_title(param, fontsize=LABEL_FS - 1, fontweight='bold',
                         color='black',
                         bbox=dict(facecolor=title_color, alpha=0.35,
                                   edgecolor='none', pad=2))
            ax.tick_params(axis='both', labelsize=TICK_FS - 2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=LEGEND_FS - 2, framealpha=0.7)

        # Hide unused subplots
        for pi in range(len(params), nrows * ncols):
            axes[pi // ncols][pi % ncols].set_visible(False)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'le_curves_by_param_{lkey}.png')
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fname}")


# ─── R² vs Pearson r² scatter plot ───────────────────────────────────────────

def plot_r2_vs_pearson(
    model_data: dict[str, dict],
    out_dir: str,
    seg: str,
    dataset: str,
) -> None:
    """Scatter: x = max|Pearson r|², y = linear-probe R² (full dataset).

    One figure per latent key.  Each point is a (param, model) pair, coloured
    by model.  A y=x reference line shows where probe and Pearson² agree.
    Params are filtered to those appropriate for *seg* (UKBB only).
    """
    allowed = _seg_allowed_params(seg) if _is_ukbb(dataset) else None
    models  = sorted(model_data.keys())

    # Collect probe R²: {lkey: {param: {model: r2}}}
    r2_data: dict = {}
    for model, summary in model_data.items():
        for lkey, ldata in summary.get('probes', {}).items():
            if lkey in _SKIP_LKEYS:
                continue
            for param, method_dict in ldata.get('regression', {}).items():
                meth = _prefer_method(list(method_dict.keys()), REG_METHOD_PREF)
                if not meth:
                    continue
                r2 = method_dict[meth].get('r2')
                if r2 is None:
                    continue
                disp = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                if allowed is not None and disp not in allowed:
                    continue
                r2_data.setdefault(lkey, {}).setdefault(disp, {})[model] = float(r2)

    # Collect Pearson max|r|: {lkey: {param: {model: max_abs_r}}}
    pearson_data: dict = {}
    for model, summary in model_data.items():
        for lkey, param_dict in summary.get('pearson', {}).items():
            if lkey in _SKIP_LKEYS:
                continue
            for param, info in param_dict.items():
                max_r = info.get('max_abs_r')
                if max_r is None:
                    continue
                disp = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                if allowed is not None and disp not in allowed:
                    continue
                pearson_data.setdefault(lkey, {}).setdefault(disp, {})[model] = float(max_r)

    all_lkeys = sorted((set(r2_data) | set(pearson_data)) - _SKIP_LKEYS)
    if not all_lkeys:
        print(f"    [skip] No data for R² vs Pearson² plot")
        return

    for lkey in all_lkeys:
        param_set = (set(r2_data.get(lkey, {})) | set(pearson_data.get(lkey, {})))
        if not param_set:
            continue
        params = (_sort_params_by_anatomy(list(param_set))
                  if _is_ukbb(dataset) else sorted(param_set))

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.suptitle(
            f'Linear-probe R² vs Pearson r²\n'
            f'{SEGMENT_LABELS.get(seg, seg)} | {dataset} | latent: {lkey}',
            fontsize=TITLE_FS, fontweight='bold',
        )

        all_vals: list[float] = []
        for model in models:
            xs, ys, labels = [], [], []
            for p in params:
                r2  = r2_data.get(lkey, {}).get(p, {}).get(model)
                r_sq = pearson_data.get(lkey, {}).get(p, {}).get(model)
                if r2 is None or r_sq is None:
                    continue
                xs.append(r_sq ** 2)
                ys.append(r2)
                labels.append(p)
                all_vals.extend([r_sq ** 2, r2])

            if not xs:
                continue
            ax.scatter(xs, ys, color=_color(model), label=_label(model),
                       s=60, zorder=3, alpha=0.85)
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (x, y), textcoords='offset points',
                            xytext=(4, 3), fontsize=7, color=_color(model))

        if all_vals:
            lo = min(0.0, min(all_vals))
            hi = max(all_vals) * 1.05
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.4, label='y = x')

        ax.set_xlabel('max |Pearson r|²', fontsize=LABEL_FS)
        ax.set_ylabel('Linear-probe R²', fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=LEGEND_FS, framealpha=0.8)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'r2_vs_pearson_{lkey}.png')
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fname}")


# ─── Permutation test comparison plot ────────────────────────────────────────

def plot_permutation_test(
    model_data: dict[str, dict],
    out_dir: str,
    seg: str,
    dataset: str,
) -> None:
    """Grouped bar chart: observed probe R² vs null (permuted) R² ± std.

    For each param in _PERMTEST_PARAMS, two bars per model per latent key:
    one for the observed OLS R² (from probes) and one for the null R² (mean ± std
    from the permutation test).  One figure per latent key.
    """
    _PERMTEST_DISPLAY = {'lv_mass', 'lvedv', 'rvedv'}

    models = sorted(model_data.keys())

    # Collect observed R² from probes: {lkey: {param: {model: r2}}}
    obs: dict = {}
    for model, summary in model_data.items():
        for lkey, ldata in summary.get('probes', {}).items():
            for param, method_dict in ldata.get('regression', {}).items():
                disp = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                if disp not in _PERMTEST_DISPLAY:
                    continue
                meth = _prefer_method(list(method_dict.keys()), REG_METHOD_PREF)
                if not meth:
                    continue
                r2 = method_dict[meth].get('r2')
                if r2 is not None:
                    obs.setdefault(lkey, {}).setdefault(disp, {})[model] = float(r2)

    # Collect null R² from permutation test: {lkey: {param: {model: (mean, std)}}}
    null: dict = {}
    for model, summary in model_data.items():
        for lkey, param_dict in summary.get('permutation_test', {}).items():
            for param, stats in param_dict.items():
                disp = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                if disp not in _PERMTEST_DISPLAY:
                    continue
                mean = stats.get('r2_mean')
                std  = stats.get('r2_std', 0.0)
                if mean is not None:
                    null.setdefault(lkey, {}).setdefault(disp, {})[model] = (
                        float(mean), float(std))

    all_lkeys = sorted((set(obs) | set(null)) - _SKIP_LKEYS)
    if not all_lkeys:
        print(f"    [skip] No permutation test data")
        return

    for lkey in all_lkeys:
        params = sorted(
            (set(obs.get(lkey, {})) | set(null.get(lkey, {}))) & _PERMTEST_DISPLAY
        )
        if not params:
            continue

        n_models = len(models)
        # Each param gets 2 bars per model (observed + null), grouped by param
        group_w  = 0.8
        bar_w    = group_w / (n_models * 2 + 0.5)
        xs = np.arange(len(params))

        fig, ax = plt.subplots(figsize=(max(6, len(params) * n_models * 1.2 + 2), 4))
        fig.suptitle(
            f'OLS R²: Observed vs Permutation Null\n'
            f'{SEGMENT_LABELS.get(seg, seg)} | {dataset} | latent: {lkey}',
            fontsize=TITLE_FS, fontweight='bold',
        )

        for mi, model in enumerate(models):
            obs_means  = [obs.get(lkey, {}).get(p, {}).get(model) for p in params]
            null_entries = [null.get(lkey, {}).get(p, {}).get(model) for p in params]
            null_means = [e[0] if e else None for e in null_entries]
            null_stds  = [e[1] if e else 0.0  for e in null_entries]

            base_off = (mi * 2 - n_models + 0.5) * bar_w
            obs_off  = base_off - bar_w / 2
            nul_off  = base_off + bar_w / 2

            obs_ys  = [v if v is not None else 0.0 for v in obs_means]
            null_ys = [v if v is not None else 0.0 for v in null_means]

            ax.bar(xs + obs_off, obs_ys, bar_w * 0.9,
                   color=_color(model), alpha=0.85,
                   label=f'{_label(model)} observed')
            ax.bar(xs + nul_off, null_ys, bar_w * 0.9,
                   yerr=null_stds, capsize=3,
                   color=_color(model), alpha=0.35, hatch='//',
                   label=f'{_label(model)} null',
                   error_kw={'elinewidth': 1})

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(params, fontsize=TICK_FS)
        ax.set_ylabel('R²', fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=LEGEND_FS - 1, framealpha=0.8,
                  bbox_to_anchor=(1, 1), loc='upper left')

        plt.tight_layout()
        fname = os.path.join(out_dir, f'permutation_test_{lkey}.png')
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fname}")


# ─── Per-segment/dataset orchestration ────────────────────────────────────────

def analyse_group(
    seg: str,
    dataset: str,
    model_data: dict[str, dict],
    out_root: str,
) -> None:
    is_medalcare = 'medalcare' in dataset.lower()
    seg_dir = os.path.join(out_root, seg, dataset.lower().replace(' ', '_'))
    os.makedirs(seg_dir, exist_ok=True)

    print(f"\n  [{SEGMENT_LABELS.get(seg, seg)} / {dataset}]")

    # 1 — MSE
    plot_mse(
        model_data,
        os.path.join(seg_dir, 'mse_comparison.png'),
        seg, dataset,
    )

    # Determine classification params
    all_clf_params = _collect_clf_params(model_data)
    class_param     = 'class' if 'class' in all_clf_params else None
    other_clf_params = sorted(all_clf_params - {'class'})

    # 2 — Classification
    if is_medalcare and class_param:
        # (a) overview metrics for pathology param
        plot_clf_overview(
            model_data,
            os.path.join(seg_dir, 'classification_overview.png'),
            seg, dataset, param='class',
            title_suffix=' (Pathology)',
        )
        # (b) per-class accuracy
        plot_clf_perclass(
            model_data,
            os.path.join(seg_dir, 'classification_perclass.png'),
            seg, dataset, param='class',
        )
        # (c) other params
        if other_clf_params:
            plot_clf_other(
                model_data,
                os.path.join(seg_dir, 'classification_other.png'),
                seg, dataset, other_clf_params,
            )
    else:
        # non-MedalCare or no 'class' param: single plot with all params
        if class_param:
            plot_clf_overview(
                model_data,
                os.path.join(seg_dir, 'classification_overview.png'),
                seg, dataset, param=class_param,
            )
        if other_clf_params:
            plot_clf_other(
                model_data,
                os.path.join(seg_dir, 'classification_other.png'),
                seg, dataset, other_clf_params,
            )

    # 3 — Linear probe
    plot_linear_probe(
        model_data,
        os.path.join(seg_dir, 'linear_probe_comparison.png'),
        seg, dataset,
    )

    # 4 — R² diff plot (vs VAE baseline; skipped if VAE not present)
    plot_linear_probe_diff(
        model_data,
        os.path.join(seg_dir, 'linear_probe_diff.png'),
        seg, dataset,
        ref_model='vae',
    )

    # 5 — Label efficiency learning curves
    plot_label_efficiency(
        model_data,
        os.path.join(seg_dir, 'label_efficiency.png'),
        seg, dataset,
    )

    # 6 — Pearson max |r| comparison
    plot_pearson_comparison(
        model_data,
        os.path.join(seg_dir, 'pearson_comparison.png'),
        seg, dataset,
    )

    # 7 — UK Biobank label efficiency detailed plots (node vs vae only)
    if _is_ukbb(dataset) and seg in ('atrial', 'ventricular', 'whole'):
        plot_le_bars_by_fraction(model_data, seg_dir, seg, dataset)
        plot_le_curves_by_param(model_data, seg_dir, seg, dataset)

    # 8 — R² vs Pearson r² scatter (UKBB, seg-filtered params)
    if _is_ukbb(dataset):
        plot_r2_vs_pearson(model_data, seg_dir, seg, dataset)

    # 9 — Permutation test: observed R² vs null R²
    if _is_ukbb(dataset):
        plot_permutation_test(model_data, seg_dir, seg, dataset)


# ─── Markdown summary ─────────────────────────────────────────────────────────

def _bold(val: float, best: float, higher_is_better: bool = True,
          fmt: str = '.4f') -> str:
    is_best = (val == best) if higher_is_better else (val == best)
    s = f'{val:{fmt}}'
    return f'**{s}**' if is_best else s


def _md_mse_table(model_data: dict[str, dict], models: list[str]) -> str:
    rows = ['| Metric | ' + ' | '.join(_label(m) for m in models) + ' |',
            '|' + '--------|' * (len(models) + 1)]

    # Val MSE
    vals = {m: model_data[m].get('training', {}).get('best_val_mse') for m in models}
    existing = [v for v in vals.values() if v is not None]
    best = min(existing) if existing else None
    cells = []
    for m in models:
        v = vals[m]
        cells.append(f'**{v:.4f}**' if (v is not None and v == best) else (f'{v:.4f}' if v is not None else '—'))
    rows.append('| Val MSE | ' + ' | '.join(cells) + ' |')

    # Test MSE avg
    avg_tests = {}
    for m in models:
        te = model_data[m].get('training', {}).get('test_mse', {})
        means = [e['mean'] for e in te.values() if 'mean' in e]
        avg_tests[m] = np.mean(means) if means else None
    existing = [v for v in avg_tests.values() if v is not None]
    best = min(existing) if existing else None
    cells = []
    for m in models:
        v = avg_tests[m]
        cells.append(f'**{v:.4f}**' if (v is not None and v == best) else (f'{v:.4f}' if v is not None else '—'))
    rows.append('| Test MSE (avg) | ' + ' | '.join(cells) + ' |')

    return '\n'.join(rows)


def _md_clf_table(model_data: dict[str, dict], models: list[str], param: str) -> str:
    overview, per_class, lkeys, method = _collect_clf_data(model_data, param)
    if not lkeys:
        return '_No data._'

    header_metrics = CLF_OVERVIEW_METRICS

    # One row per (lkey, model), metric as columns
    cols = ['Latent', 'Model'] + [ml for _, ml in header_metrics]
    rows = ['| ' + ' | '.join(cols) + ' |',
            '|' + '---|' * len(cols)]

    for lkey in lkeys:
        # find best per metric (higher is better for all)
        bests = {}
        for mk, _ in header_metrics:
            vals = [overview.get(lkey, {}).get(mk, {}).get(m) for m in models]
            vals = [v for v in vals if v is not None]
            bests[mk] = max(vals) if vals else None

        for m in models:
            cells = [lkey, _label(m)]
            for mk, _ in header_metrics:
                v = overview.get(lkey, {}).get(mk, {}).get(m)
                if v is None:
                    cells.append('—')
                elif bests[mk] is not None and v == bests[mk]:
                    cells.append(f'**{v:.4f}**')
                else:
                    cells.append(f'{v:.4f}')
            rows.append('| ' + ' | '.join(cells) + ' |')

    note = f'\n_Method: {method}_' if method else ''
    return '\n'.join(rows) + note


def _md_reg_table(model_data: dict[str, dict], models: list[str],
                  dataset: str = '') -> str:
    data: dict = defaultdict(lambda: {'r2': {}, 'mae': {}})
    lkey_set: set = set()
    param_set: set = set()

    for model, summary in model_data.items():
        for lkey, ldata in summary.get('probes', {}).items():
            for param, method_dict in ldata.get('regression', {}).items():
                meth = _prefer_method(list(method_dict.keys()), REG_METHOD_PREF)
                if not meth:
                    continue
                res = method_dict[meth]
                param_disp = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                if res.get('r2') is not None:
                    data[(lkey, param_disp)]['r2'][model] = float(res['r2'])
                if res.get('mae') is not None:
                    data[(lkey, param_disp)]['mae'][model] = float(res['mae'])
                lkey_set.add(lkey)
                param_set.add(param_disp)

    if not lkey_set:
        return '_No regression probe data._'

    if _is_ukbb(dataset):
        return _md_reg_table_ukbb(data, models, lkey_set, param_set)

    cols = (['Latent', 'Param']
            + [f'{_label(m)} R²' for m in models]
            + [f'{_label(m)} MAE' for m in models])
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]

    for lkey in sorted(lkey_set):
        for param in sorted(param_set):
            key = (lkey, param)
            if key not in data:
                continue
            r2_vals  = [data[key]['r2'].get(m)  for m in models]
            mae_vals = [data[key]['mae'].get(m) for m in models]
            best_r2  = max((v for v in r2_vals  if v is not None), default=None)
            best_mae = min((v for v in mae_vals if v is not None), default=None)
            cells = [lkey, param]
            for m in models:
                v = data[key]['r2'].get(m)
                cells.append(f'**{v:.4f}**' if (v is not None and v == best_r2)
                              else (f'{v:.4f}' if v is not None else '—'))
            for m in models:
                v = data[key]['mae'].get(m)
                cells.append(f'**{v:.4f}**' if (v is not None and v == best_mae)
                              else (f'{v:.4f}' if v is not None else '—'))
            rows.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(rows)


def _md_reg_table_ukbb(data: dict, models: list[str],
                       lkey_set: set, param_set: set) -> str:
    """UKBB regression table: separate atrial/ventricular sections + R² diff + MAE columns."""
    ref_m = 'vae' if 'vae' in models else models[0]
    diff_models = [m for m in models if m != ref_m]

    diff_cols = [f'Δ vs {_label(ref_m)} ({_label(m)})' for m in diff_models]
    r2_cols   = [f'{_label(m)} R²'  for m in models]
    mae_cols  = [f'{_label(m)} MAE' for m in models]
    cols      = ['Latent', 'Param'] + r2_cols + diff_cols + mae_cols

    sections: list[str] = []
    anatomy_groups = [
        ('Atrial',      _UKBB_ATRIAL_PARAMS),
        ('Ventricular', _UKBB_VENTRICULAR_PARAMS),
        ('Both',        _UKBB_BOTH_PARAMS),
    ]
    for anat_label, anat_set in anatomy_groups:
        anat_params = sorted(p for p in param_set if p in anat_set)
        if not anat_params:
            continue
        rows = [f'##### {anat_label}', '',
                '| ' + ' | '.join(cols) + ' |',
                '|' + '---|' * len(cols)]
        for lkey in sorted(lkey_set):
            for param in anat_params:
                key = (lkey, param)
                if key not in data:
                    continue
                r2_vals  = [data[key]['r2'].get(m)  for m in models]
                mae_vals = [data[key]['mae'].get(m) for m in models]
                best_r2  = max((v for v in r2_vals  if v is not None), default=None)
                best_mae = min((v for v in mae_vals if v is not None), default=None)
                cells = [lkey, param]
                for m in models:
                    v = data[key]['r2'].get(m)
                    cells.append(f'**{v:.4f}**' if (v is not None and v == best_r2)
                                  else (f'{v:.4f}' if v is not None else '—'))
                for dm in diff_models:
                    v_dm  = data[key]['r2'].get(dm)
                    v_ref = data[key]['r2'].get(ref_m)
                    cells.append(f'{v_dm - v_ref:+.4f}'
                                 if (v_dm is not None and v_ref is not None) else '—')
                for m in models:
                    v = data[key]['mae'].get(m)
                    cells.append(f'**{v:.4f}**' if (v is not None and v == best_mae)
                                  else (f'{v:.4f}' if v is not None else '—'))
                rows.append('| ' + ' | '.join(cells) + ' |')
        sections.append('\n'.join(rows))

    return '\n\n'.join(sections) if sections else '_No regression probe data._'


def _md_label_efficiency_table(model_data: dict[str, dict], models: list[str],
                                dataset: str = '') -> str:
    """Table of OLS R² mean±std at each training fraction.

    Rows: (latent_key, param, model).  Columns: one per fraction showing
    'mean ± std'.  Best mean per (lkey, param, fraction) is bolded.
    """
    # Collect: {(lkey, param): {model: {frac: (mean, std)}}}
    data: dict = defaultdict(lambda: defaultdict(dict))
    lkey_set: set = set()
    param_set: set = set()

    for model, summary in model_data.items():
        for lkey, frac_dict in summary.get('label_efficiency', {}).items():
            if lkey in _SKIP_LKEYS:
                continue
            lkey_set.add(lkey)
            for frac_str in _LE_FRAC_ORDER:
                agg = frac_dict.get(frac_str, {}).get('regression', {})
                for param, stats in agg.items():
                    mean = stats.get('r2_mean')
                    std  = stats.get('r2_std', 0.0)
                    if mean is not None:
                        disp = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                        data[(lkey, disp)][model][frac_str] = (float(mean), float(std))
                        param_set.add(disp)

    if not lkey_set:
        return '_No label efficiency data._'

    frac_cols = _LE_FRAC_ORDER
    cols = ['Latent', 'Param', 'Model'] + frac_cols
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]

    for lkey in sorted(lkey_set):
        for param in sorted(param_set):
            key = (lkey, param)
            if key not in data:
                continue
            # best mean per fraction across models (higher R² is better)
            bests = {}
            for frac in frac_cols:
                means = [data[key][m][frac][0] for m in models
                         if data[key].get(m) and frac in data[key][m]]
                bests[frac] = max(means) if means else None

            for model in models:
                model_fracs = data[key].get(model, {})
                if not model_fracs:
                    continue
                cells = [lkey, param, _label(model)]
                for frac in frac_cols:
                    entry = model_fracs.get(frac)
                    if entry is None:
                        cells.append('—')
                    else:
                        mean, std = entry
                        s = f'{mean:.3f}±{std:.3f}'
                        cells.append(f'**{s}**' if (bests[frac] is not None
                                                     and mean == bests[frac]) else s)
                rows.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(rows)


def _md_pearson_table(model_data: dict[str, dict], models: list[str],
                      dataset: str = '') -> str:
    """Table of max |Pearson r| per (lkey, param) for each model."""
    # Collect: {(lkey, param): {model: {max_abs_r, dim, r}}}
    data: dict = defaultdict(dict)
    lkey_set: set = set()
    param_set: set = set()

    for model, summary in model_data.items():
        for lkey, param_dict in summary.get('pearson', {}).items():
            lkey_set.add(lkey)
            for param, info in param_dict.items():
                disp = _UKBB_PARAM_RENAME.get(param, param) if _is_ukbb(dataset) else param
                data[(lkey, disp)][model] = info
                param_set.add(disp)

    if not lkey_set:
        return '_No Pearson correlation data._'

    cols = ['Latent', 'Param'] + [f'{_label(m)} max|r| (dim)' for m in models]
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]

    for lkey in sorted(lkey_set):
        for param in sorted(param_set):
            key = (lkey, param)
            if key not in data:
                continue
            r_vals = [data[key].get(m, {}).get('max_abs_r') for m in models]
            best   = max((v for v in r_vals if v is not None), default=None)
            cells  = [lkey, param]
            for model in models:
                info = data[key].get(model)
                if info is None:
                    cells.append('—')
                else:
                    v   = info.get('max_abs_r')
                    dim = info.get('dim', '?')
                    s   = f'{v:.4f} (z{dim})'
                    cells.append(f'**{s}**' if (v is not None and v == best) else s)
            rows.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(rows)


def write_summary_md(grouped: dict, out_path: str) -> None:
    lines: list[str] = [
        '# MoNODE Results Summary',
        '',
        '> **Bold** values indicate the best result in each metric column.',
        '',
    ]

    seg_order = ['atrial', 'ventricular', 'whole', 'all']
    segs = sorted(grouped.keys(), key=lambda s: seg_order.index(s) if s in seg_order else 99)

    for seg in segs:
        lines += [f'---', f'', f'## Segment: {SEGMENT_LABELS.get(seg, seg.capitalize())}', '']
        datasets = sorted(grouped[seg].keys())

        for dataset in datasets:
            model_data = grouped[seg][dataset]
            models = sorted(model_data.keys())
            lines += [f'### Dataset: {dataset}', '']

            # ── MSE ──
            lines += ['#### MSE', '']
            lines.append(_md_mse_table(model_data, models))
            lines += ['']

            # ── Classification ──
            all_clf_params = _collect_clf_params(model_data)
            is_medalcare   = 'medalcare' in dataset.lower()

            if all_clf_params:
                lines += ['#### Classification', '']
                if is_medalcare and 'class' in all_clf_params:
                    # MedalCare: 'class' first, then any other params
                    lines += ['##### Pathology (class)', '']
                    lines.append(_md_clf_table(model_data, models, 'class'))
                    lines += ['']
                    for param in sorted(all_clf_params - {'class'}):
                        lines += [f'##### {param}', '']
                        lines.append(_md_clf_table(model_data, models, param))
                        lines += ['']
                else:
                    # Other datasets: emit all params
                    for param in sorted(all_clf_params):
                        lines += [f'##### {param}', '']
                        lines.append(_md_clf_table(model_data, models, param))
                        lines += ['']

            # ── Linear probe ──
            lines += ['#### Linear Probe (Regression)', '']
            lines.append(_md_reg_table(model_data, models, dataset=dataset))
            lines += ['', '']

            # ── Label efficiency ──
            lines += ['#### Label Efficiency (best R² at 100% training)', '']
            lines.append(_md_label_efficiency_table(model_data, models, dataset=dataset))
            lines += ['', '']

            # ── Pearson correlations ──
            lines += ['#### Pearson Correlation (max |r| per param)', '']
            lines.append(_md_pearson_table(model_data, models, dataset=dataset))
            lines += ['', '']

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Saved summary: {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Cross-model comparison of MoNODE run summaries.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing summary JSONs from summarize_results.py')
    parser.add_argument('--output_dir',  required=True,
                        help='Directory to write plots and summary.md')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading summaries from: {args.results_dir}")
    summaries = load_summaries(args.results_dir)
    if not summaries:
        print("  No summary JSONs found — exiting.")
        return

    print(f"Found {len(summaries)} run(s):")
    for (seg, model, dataset) in sorted(summaries.keys()):
        print(f"  • segment={seg}, model={model}, dataset={dataset}")

    grouped = group_summaries(summaries)

    # Architecture plots go under {output_dir}/architecture/
    arch_dir = os.path.join(args.output_dir, 'architecture')
    os.makedirs(arch_dir, exist_ok=True)

    # Generate per-segment/dataset plots
    for seg, datasets in grouped.items():
        for dataset, model_data in datasets.items():
            analyse_group(seg, dataset, model_data, arch_dir)

    # Generate summary markdown
    write_summary_md(grouped, os.path.join(arch_dir, 'summary.md'))

    print("\nDone.")


if __name__ == '__main__':
    main()
