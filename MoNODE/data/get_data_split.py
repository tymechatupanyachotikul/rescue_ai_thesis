from collections import defaultdict
import os
import math
import hashlib
import random
import pandas as pd
import argparse


def gen_ecg_data(args):

    split = args.split
    anatomy = args.anatomy
    dataset = args.dataset
    root_dir = args.root_dir
    out_dir = args.out_dir

    data_paths = defaultdict(lambda: defaultdict(list))

    if dataset.lower() == 'medalcare_xl':

        subclasses = [
            'LAD_0.3', 'LAD_1.0',
            'LCX_0.3_ant', 'LCX_0.3_post',
            'LCX_1.0_ant', 'LCX_1.0_post',
            'RCA_0.3', 'RCA_1.0'
        ]

        classes_dict = {
            'ventricular': subclasses + ['lbbb', 'rbbb'],
            'atrial': ['avblock', 'fam', 'iab', 'lae'],
            'normal': ['sinus']
        }

        total_cls = defaultdict(int)

        for cls_dir in os.listdir(root_dir):
            if cls_dir == 'mi':

                for subclass_dir in os.listdir(os.path.join(root_dir, cls_dir)):
                    if subclass_dir in subclasses:

                        cur_dir = os.path.join(root_dir, cls_dir, subclass_dir, split)

                        for run_dir in os.listdir(cur_dir):
                            run_path = os.path.join(cur_dir, run_dir)

                            for data_file in os.listdir(run_path):
                                data_path = os.path.join(run_path, data_file)

                                if data_path.endswith('csv') and 'raw' in data_path:
                                    total_cls['mi'] += 1
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
                            total_cls[cls_dir] += 1
                            data_paths[cls_dir][run_dir].append(data_path)

        out_dataset = {
            'data_path': [],
            'label': [],
            'hash': []
        }

        n_sinus = (
            total_cls['lbbb'] + total_cls['rbbb'] + total_cls['mi']
            if anatomy == 'ventricular'
            else total_cls['av_block'] + total_cls['fam'] +
                 total_cls['iab'] + total_cls['lae']
        )
        print(f'Processing split {split} for {anatomy}')
        print(f'Found {n_sinus} samples for {anatomy} pathology')

        n_sinus = int(math.ceil(n_sinus * 1.25))
        print(f'Using {n_sinus} samples for sinus class')

        n_sinus_class = (
            len(classes_dict['ventricular']) + 1
            if anatomy == 'atrial'
            else len(classes_dict['atrial']) + 1
        )

        n_per_class = math.ceil(n_sinus / n_sinus_class)

        n_s_total = 0
        n_a_total = 0

        for cls, run in data_paths.items():

            n_cls = 0

            if cls in classes_dict[anatomy]:

                for paths in run.values():
                    for path in paths:

                        out_dataset['data_path'].append(path)
                        out_dataset['label'].append(cls)

                        filename = os.path.basename(path)
                        out_dataset['hash'].append(
                            hashlib.sha256(filename.encode()).hexdigest()
                        )

                        n_cls += 1
                        n_a_total += 1

                print(f'Found {n_cls} samples for class {cls}')

            else:

                n_runs = len(run)
                n_per_run = math.ceil(n_per_class / n_runs)

                for paths in run.values():

                    random.shuffle(paths)

                    for i in range(min(n_per_run, len(paths))):

                        out_dataset['data_path'].append(paths[i])
                        out_dataset['label'].append(cls)

                        filename = os.path.basename(paths[i])
                        out_dataset['hash'].append(
                            hashlib.sha256(filename.encode()).hexdigest()
                        )

                        n_cls += 1
                        n_s_total += 1

                print(f'Found {n_cls} samples for class sinus (substitute for {cls})')

        if n_s_total < n_sinus:
            for paths in data_paths['sinus'].values():
                for path in paths:
                    if path not in out_dataset['data_path']:
                        out_dataset['data_path'].append(path)
                        out_dataset['label'].append('sinus')

                        filename = os.path.basename(path)
                        out_dataset['hash'].append(
                            hashlib.sha256(filename.encode()).hexdigest()
                        )

                        n_s_total += 1

                        if n_s_total >= n_sinus:
                            break
        print(
            f'Total {n_s_total} samples for sinus class and '
            f'{n_a_total} samples for {anatomy} classes'
        )

        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(out_dataset)

        df.to_csv(
            os.path.join(out_dir, f"{dataset.lower()}_{split}_{anatomy}.csv"),
            index=False
        )

def build_parser():

    parser = argparse.ArgumentParser(
        description="Generate ECG dataset CSV for MedalCare-XL"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="medalcare_xl",
        help="Dataset name"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        required=True,
        help="Dataset split"
    )

    parser.add_argument(
        "--anatomy",
        type=str,
        choices=["ventricular", "atrial"],
        required=True,
        help="Pathology type"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of dataset"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for generated CSV"
    )

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    gen_ecg_data(args)


if __name__ == "__main__":
    main()