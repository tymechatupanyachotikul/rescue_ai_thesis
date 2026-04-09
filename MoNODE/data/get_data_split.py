from collections import defaultdict
import os
import math
import hashlib
import random
import pandas as pd
import argparse


MI_SUBCLASSES = [
    'LAD_0.3', 'LAD_1.0',
    'LCX_0.3_ant', 'LCX_0.3_post',
    'LCX_1.0_ant', 'LCX_1.0_post',
    'RCA_0.3', 'RCA_1.0',
]

CLASSES_DICT = {
    'ventricular': MI_SUBCLASSES + ['lbbb', 'rbbb'],
    'atrial':      ['avblock', 'fam', 'iab', 'lae'],
    'normal':      ['sinus'],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _collect_medalcare_paths(root_dir: str, split: str) -> defaultdict:
    """Walk the MedalCare-XL directory tree and return all raw CSV paths.

    Returns
    -------
    data_paths : defaultdict  class_label -> run_id -> [file_path, ...]
    """
    data_paths: defaultdict = defaultdict(lambda: defaultdict(list))

    for cls_dir in os.listdir(root_dir):
        if cls_dir == 'mi':
            for subclass_dir in os.listdir(os.path.join(root_dir, cls_dir)):
                if subclass_dir not in MI_SUBCLASSES:
                    continue
                cur_dir = os.path.join(root_dir, cls_dir, subclass_dir, split)
                if not os.path.isdir(cur_dir):
                    continue
                for run_dir in os.listdir(cur_dir):
                    run_path = os.path.join(cur_dir, run_dir)
                    for data_file in os.listdir(run_path):
                        data_path = os.path.join(run_path, data_file)
                        if data_path.endswith('csv') and 'raw' in data_path:
                            data_paths[subclass_dir][run_dir].append(data_path)
        else:
            cur_dir = os.path.join(root_dir, cls_dir, split)
            if not os.path.isdir(cur_dir):
                continue
            for run_dir in os.listdir(cur_dir):
                run_path = os.path.join(cur_dir, run_dir)
                for data_file in os.listdir(run_path):
                    data_path = os.path.join(run_path, data_file)
                    if data_path.endswith('csv') and 'raw' in data_path:
                        data_paths[cls_dir][run_dir].append(data_path)

    return data_paths


def _add_row(out_dataset: dict, path: str, label: str):
    out_dataset['data_path'].append(path)
    out_dataset['label'].append(label)
    out_dataset['hash'].append(hashlib.sha256(path.encode()).hexdigest())


# ---------------------------------------------------------------------------
# All-samples mode (no resampling)
# ---------------------------------------------------------------------------

def _gen_medalcare_all(args):
    """Include every sample from every class without any resampling."""
    split    = args.split
    root_dir = args.root_dir
    out_dir  = args.out_dir
    dataset  = args.dataset

    data_paths = _collect_medalcare_paths(root_dir, split)

    out_dataset = {'data_path': [], 'label': [], 'hash': []}
    total_cls   = defaultdict(int)

    for cls, runs in data_paths.items():
        for paths in runs.values():
            for path in paths:
                _add_row(out_dataset, path, cls)
                total_cls[cls] += 1

    print(f'All-samples split={split}:')
    for cls, n in sorted(total_cls.items()):
        print(f'  {cls}: {n}')
    print(f'  TOTAL: {sum(total_cls.values())}')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{dataset.lower()}_{split}_all.csv')
    pd.DataFrame(out_dataset).to_csv(out_path, index=False)
    print(f'Saved → {out_path}')


# ---------------------------------------------------------------------------
# Anatomy-resampled mode (original behaviour)
# ---------------------------------------------------------------------------

def _gen_medalcare_resampled(args):
    """Collect anatomy-specific pathology samples and resample the sinus class."""
    split    = args.split
    anatomy  = args.anatomy
    root_dir = args.root_dir
    out_dir  = args.out_dir
    dataset  = args.dataset

    data_paths = _collect_medalcare_paths(root_dir, split)

    # Count samples per class
    total_cls: defaultdict = defaultdict(int)
    for cls, runs in data_paths.items():
        for paths in runs.values():
            total_cls[cls] += len(paths)
    # MI subclasses roll up into 'mi' for the sinus-budget calculation
    total_cls['mi'] = sum(total_cls[sc] for sc in MI_SUBCLASSES)
    print(total_cls)

    n_sinus = (
        total_cls['lbbb'] + total_cls['rbbb'] + total_cls['mi']
        if anatomy == 'ventricular'
        else total_cls['av_block'] + total_cls['fam'] +
             total_cls['iab'] + total_cls['lae']
    )
    print(f'Processing split={split} for anatomy={anatomy}')
    print(f'Found {n_sinus} pathology samples')

    n_sinus = int(math.ceil(n_sinus * 1.25))
    print(f'Using {n_sinus} sinus samples (budget)')

    n_sinus_class = (
        len(CLASSES_DICT['ventricular']) + 1
        if anatomy == 'atrial'
        else len(CLASSES_DICT['atrial']) + 1
    )
    n_per_class = math.ceil(n_sinus / n_sinus_class)

    out_dataset = {'data_path': [], 'label': [], 'hash': []}
    n_s_total   = 0
    n_a_total   = 0

    for cls, run in data_paths.items():
        n_cls = 0

        if cls in CLASSES_DICT[anatomy]:
            for paths in run.values():
                for path in paths:
                    _add_row(out_dataset, path, cls)
                    n_cls   += 1
                    n_a_total += 1
            print(f'  {cls}: {n_cls} pathology samples')

        else:
            n_runs    = len(run)
            n_per_run = math.ceil(n_per_class / n_runs)
            for paths in run.values():
                random.shuffle(paths)
                for path in paths[:n_per_run]:
                    _add_row(out_dataset, path, cls)
                    n_cls     += 1
                    n_s_total += 1
            print(f'  {cls}: {n_cls} sinus-substitute samples')

    # Top-up from the sinus class if still under budget
    if n_s_total < n_sinus and 'sinus' in data_paths:
        seen = set(out_dataset['data_path'])
        for paths in data_paths['sinus'].values():
            for path in paths:
                if path not in seen:
                    _add_row(out_dataset, path, 'sinus')
                    seen.add(path)
                    n_s_total += 1
                    if n_s_total >= n_sinus:
                        break
            if n_s_total >= n_sinus:
                break

    print(f'Total: {n_s_total} sinus, {n_a_total} {anatomy} pathology')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{dataset.lower()}_{split}_{anatomy}.csv')
    pd.DataFrame(out_dataset).to_csv(out_path, index=False)
    print(f'Saved → {out_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def gen_ecg_data(args):
    if args.dataset.lower() == 'medalcare_xl':
        if args.all_samples:
            _gen_medalcare_all(args)
        else:
            if not args.anatomy:
                raise ValueError("--anatomy is required when --all_samples is not set")
            _gen_medalcare_resampled(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate ECG dataset CSV for MedalCare-XL"
    )
    parser.add_argument("--dataset",     type=str, default="medalcare_xl")
    parser.add_argument("--split",       type=str, required=True,
                        choices=["train", "valid", "test"])
    parser.add_argument("--anatomy",     type=str, default=None,
                        choices=["ventricular", "atrial"],
                        help="Pathology type. Required unless --all_samples is set.")
    parser.add_argument("--all_samples", action="store_true",
                        help="Include all samples from every class without resampling. "
                             "Outputs {dataset}_{split}_all.csv. Ignores --anatomy.")
    parser.add_argument("--root_dir",    type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--out_dir",     type=str, required=True,
                        help="Output directory for the generated CSV")
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    gen_ecg_data(args)


if __name__ == "__main__":
    main()
