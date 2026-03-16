import json
import random
import time
import pandas as pd 
import os
import glob
import wfdb
from tqdm import tqdm
import ast
import argparse
import numpy as np 

from aladin import ALADIN
from aladin.core import Record

import matplotlib.pyplot as plt
from pprint import pprint 

MEDALCARE_XL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
def load_case(dir, case, metadata):

    file = os.path.join(dir,case)
    rec = wfdb.rdrecord(file)
    ecg = {name: rec.p_signal[:, i] for i, name in enumerate(rec.sig_name)}

    record = Record(ecg, rec.fs, "DEMO", case)
    record.groundtruth = metadata.label
    record.hash = metadata.hash

    return record, rec

def analyse_single_case(record):

    aladin = ALADIN(modelpaths=["ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": True, "afibdetector": False, "reflection": False, "total": False})
    st = time.time()
    aladin.segmenter.batch(record)
    print("Segmenter", time.time()-st)
    st = time.time()
    aladin.reflection.batch(record)
    print("Reflection", time.time()-st)

def segment(record, original_record, case, segment_type, out_dir):

    #get segments
    if segment_type == 'ventricular':
        qrst_idx = []
        for beat in record.qrs:
            start = int(beat.onset)
            if beat.t is not None:
                end = int(beat.t.offset)
            else:
                continue

            qrst_idx.append((start, end))

        segments =random.sample(qrst_idx, 2)
    elif segment_type == 'atrial':
        segments = random.sample(record.p, 2)
        segments = [(int(seg.onset), int(seg.offset)) for seg in segments]

    # normalise
    mu = np.mean(original_record.p_signal, axis=0, keepdims=True)
    sigma = np.std(original_record.p_signal, axis=0, keepdims=True)

    norm_ecg = (original_record.p_signal - mu) / (sigma + 1e-8)

    save_dir = os.path.join(out_dir, 'sampled')
    os.makedirs(save_dir, exist_ok=True)
    for idx, segment in enumerate(segments):
        ecg_segment = norm_ecg[segment[0]: segment[1], :]

        delta_t = segment[1] - segment[0]
        np.save(os.path.join(save_dir, f'T{delta_t}_{record.hash}_{record.groundtruth}_{idx}.npy'), ecg_segment.astype(np.float32))

    fig, ax = plt.subplots(1, 1, figsize=(len(norm_ecg)/(record.fs), 4), dpi=200)
    ax.plot(np.arange(0, len(norm_ecg)), norm_ecg, color='black', label="ECG Signal")

    ymax = np.max(norm_ecg)
    ymin = np.min(norm_ecg)

    for (onset, offset) in segments:
        qrscolor = '#f1c40f' 
        ax.axvspan(onset, offset, ymax = ymax, ymin = ymin, color=qrscolor, alpha=0.5)

    ax.set_ylabel("Amplitude (mV)")
    ax.set_xlabel("Time (s)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{case}_segments.png')
    plt.close()

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Load and analyze ECG cases")
    argparser.add_argument("--input_path", type=str, help="Path to the input directory", required=True)
    argparser.add_argument("--batch_size", type=int, help="Number of cases to process in each batch", default=5)
    argparser.add_argument("--out_dir", type=str, help="Path to the output directory", required=True)
    args = argparser.parse_args()

    batch_size = args.batch_size
    out_dir = args.out_dir
    segment_type = os.path.splitext(args.input_path)[0].split('_')[-1]
    split = os.path.splitext(args.input_path)[0].split('_')[-2]
    print(f'Processing {split} split for {segment_type} segments')
    out_dir = os.path.join(out_dir, split, segment_type)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.input_path, nrows=5)

    cur_batch = []
    for cur_idx, row in enumerate(df.itertuples(index=False)):
        ecg_path = str(row.data_path)
        print(f'Processing {ecg_path}')

        case = ecg_path.split("/")[-1].split(".")[0] 
        directory_path = ecg_path.split("/")[:-1]
        directory_path = "/".join(directory_path)

        filepath = os.path.join(directory_path, case)
        if not os.path.exists(filepath + '.dat') or not os.path.exists(filepath + '.hea'):
            print(f'Converting {ecg_path} to wfdb format')
            ecg = pd.read_csv(ecg_path, header=None).to_numpy().T
            print(f'Shape of loaded ECG : {ecg.shape}')
            wfdb.wrsamp(
                record_name=case, 
                write_dir=directory_path,
                fs=500, 
                units=['mV'] * ecg.shape[1], 
                sig_name=MEDALCARE_XL_LEADS, 
                p_signal=ecg, 
                fmt=['32'] * ecg.shape[1]
            )    

        record, original_record = load_case(directory_path, case, row)
        cur_batch.append((record, original_record, case))
        if len(cur_batch) == batch_size or cur_idx == len(df)-1:
            records = [_rec for _rec, _, _ in cur_batch]
            analyse_single_case(records)
            for idx, _rec in enumerate(records):
                segment(_rec, cur_batch[idx][1], cur_batch[idx][2], segment_type, out_dir)

            cur_batch = []