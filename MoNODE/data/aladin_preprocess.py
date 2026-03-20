import json
import random
import time
import pandas as pd 
import os
import glob
import gc
import wfdb
from tqdm import tqdm
import ast
import argparse
import numpy as np 
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch 

from aladin import ALADIN
from aladin.core import Record

import matplotlib.pyplot as plt
from pprint import pprint 



MEDALCARE_XL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
UK_BB_LEADS    = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEADS_DICT = {
    'MedalCare-XL': MEDALCARE_XL_LEADS,
    'ukbb': UK_BB_LEADS
}

def load_and_convert_case(row, dataset):
    """Worker function to handle file checking, conversion, and loading."""
    ecg_path = str(row.data_path)
    directory_path, filename = os.path.split(ecg_path)
    case = os.path.splitext(filename)[0]
    
    filepath = os.path.join(directory_path, case)

    if not os.path.exists(filepath + '.dat') or not os.path.exists(filepath + '.hea'):
        if filename.endswith('.csv'):
            ecg = pd.read_csv(ecg_path, header=None, dtype=np.float32).to_numpy()
        elif filename.endswith('.npy'):
            ecg = np.load(ecg_path)

        if ecg.shape[0] < ecg.shape[1]:
            ecg = ecg.T

        wfdb.wrsamp(
            record_name=case, 
            write_dir=directory_path,
            fs=500, 
            units=['mV'] * ecg.shape[1], 
            sig_name=LEADS_DICT[dataset][:ecg.shape[1]], 
            p_signal=ecg, 
            fmt=['16'] * ecg.shape[1]
        )    

    rec = wfdb.rdrecord(filepath)
    ecg_dict = {name: rec.p_signal[:, i] for i, name in enumerate(rec.sig_name)}

    record = Record(ecg_dict, rec.fs, "DEMO", case)
    if hasattr(row, 'label'):
        record.groundtruth = row.label 
    if hasattr(row, 'hash'):
        record.hash = row.hash        
    record.original_file_path = row.data_path

    return record, rec

def get_ecg_segments_idx(record, segment_type, beat_type):

    segments = []
    
    if segment_type == 'ventricular':
        if beat_type == 'sampled':
            qrst_idx = []
            for beat in record.qrs:
                start = int(beat.onset)
                if beat.t is not None:
                    end = int(beat.t.offset)
                else:
                    continue
                qrst_idx.append((start, end))
                
            k = min(2, len(qrst_idx))
            segments = random.sample(qrst_idx, k) if k > 0 else []
        elif beat_type == 'median':
            onset = record.median_beat.delineations.qrs.onset
            offset = record.median_beat.delineations.t.offset

            segments = [(int(onset), int(offset))] if onset is not None and offset is not None else []
        
    elif segment_type == 'atrial':
        if beat_type == 'sampled':
            k = min(2, len(record.p))
            sampled_p = random.sample(record.p, k) if k > 0 else []
            segments = [(int(seg.onset), int(seg.offset)) for seg in sampled_p]
        elif beat_type == 'median':
            onset = record.median_beat.delineations.p.onset
            offset = record.median_beat.delineations.p.offset

            segments = [(int(onset), int(offset))] if onset is not None and offset is not None else []

    return segments 

def save_ecg_segment(segments, norm_ecg, original_record, record, segment_type, out_dir, beat_type, dataset, plot=False):

    save_dir = os.path.join(out_dir, segment_type, beat_type)
    os.makedirs(save_dir, exist_ok=True)

    for idx, (start, end) in enumerate(segments):
        ecg_segment = norm_ecg[start:end, :]
        delta_t = end - start
        
        if dataset == 'MedalCare-XL':
            run_id = record.original_file_path.split('/')[-2].split('_')[1]
            session_id = record.original_file_path.split('/')[-1].split('_')[0]
            label = record.groundtruth.replace('.', '')

            if beat_type == 'sampled':
                base_name = f'T{delta_t}_{run_id}_{session_id}_{label}_{idx}'
            else:
                base_name = f'T{delta_t}_{run_id}_{session_id}_{label}'
        elif dataset in ['ukbb', 'mimic-iv']:
            run_id = os.path.splitext(record.original_file_path)[0]
            if beat_type == 'sampled':
                base_name = f'T{delta_t}_{run_id}_{idx}'
            else:
                base_name = f'T{delta_t}_{run_id}'

        ecg_segment = torch.from_numpy(ecg_segment.astype(np.float32))
        save_path = os.path.join(save_dir, f'{base_name}.pth')
        torch.save(ecg_segment, save_path)

        if plot:
            if beat_type == 'median':
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(original_record.p_signal[:, 0], label='Original ECG')
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude (mV)')
                ax.legend()
                fig.savefig(f'{base_name}_full_ecg_{beat_type}.png')
                plt.close(fig) 

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(norm_ecg[:, 0], label='Normalised median beat')
                ax.axvspan(start, end, color='#e74c3c', alpha=0.5, label=f"Segmented for {segment_type}")
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')
                ax.legend()
                fig.savefig(f'{base_name}_norm_segment_{beat_type}.png')
                plt.close(fig) 

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(record.median_beat.ecg[0, :], label='Median beat')
                ax.axvspan(start, end, color='#e74c3c', alpha=0.5, label=f"Segmented for {segment_type}")
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude (mV)')
                ax.legend()
                fig.savefig(f'{base_name}_segment_{beat_type}.png')
                plt.close(fig)

def process_and_save_segments(record, original_record, segment_type, out_dir, beat_type, dataset, plot=False):
    """Worker function to handle normalization and saving to disk."""
    segments_dict = {}
    segment_type = ['atrial', 'ventricular'] if segment_type == 'both' else [segment_type]

    for seg_type in segment_type:
        segment_idx = get_ecg_segments_idx(record, seg_type, beat_type)
        if segment_idx:
            segments_dict[seg_type] = segment_idx

    if len(segments_dict.keys()) == 0:
        return

    original_ecg = original_record.p_signal if beat_type == 'sampled' else record.median_beat.ecg.T
    print(f'ECG for {beat_type} {segment_type} has shape {original_ecg.shape}')

    mu = np.mean(original_ecg, axis=0, keepdims=True)
    sigma = np.std(original_ecg, axis=0, keepdims=True)
    norm_ecg = (original_ecg - mu) / (sigma + 1e-8)

    for seg_type, seg_idx in segments_dict.items():
        save_ecg_segment(seg_idx, norm_ecg, original_record, record, seg_type, out_dir, beat_type, dataset, plot=plot)
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Load and analyze ECG cases")
    argparser.add_argument("--input_path", type=str, help="Path to the input directory", required=True)
    argparser.add_argument("--batch_size", type=int, help="Number of cases to process in each batch", default=32) # Increased batch size
    argparser.add_argument("--out_dir", type=str, help="Path to the output directory", required=True)
    argparser.add_argument("--workers", type=int, help="Number of parallel workers", default=os.cpu_count())
    argparser.add_argument("--beat_type", type=str, choices=['sampled', 'median'], help="Types of beat to segment", required=True)
    argparser.add_argument("--demo", action='store_true', help="Run a quick demo with a subset of data")
    argparser.add_argument("--plot_only", action='store_true', help="Only generate plots without saving segments")
    args = argparser.parse_args()

    dataset = 'medalcare-xl' if 'medalcare-xl' in args.input_path else os.path.basename(args.input_path).split('_')[0]

    if dataset == 'medalcare-xl':
        segment_type = os.path.splitext(args.input_path)[0].split('_')[-1]
    elif dataset in ['ukbb', 'mimic-iv']:
        segment_type = 'both'

    beat_type = args.beat_type
    if dataset == 'medalcare-xl':
        split = os.path.splitext(args.input_path)[0].split('_')[-2]
    elif dataset in ['ukbb', 'mimic-iv']:
        split = os.path.splitext(args.input_path)[0].split('_')[-1]

    demo = args.demo
    plot_only = args.plot_only
    print(f'Processing {split} split for {segment_type} segments')
    
    out_dir = os.path.join(args.out_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading ALADIN model into memory...")
    aladin = ALADIN(
        modelpaths=["ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
        debug={"segmenter": False, "afibdetector": False, "reflection": False, "total": False}
    )

    df = pd.read_csv(args.input_path) if not demo else pd.read_csv(args.input_path, nrows=3)

    chunks = [df.iloc[i:i + args.batch_size] for i in range(0, len(df), args.batch_size)]
    
    print(f"Starting processing with {args.workers} workers...")
    
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Processing Batches")):
        
        loaded_data = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_row = {executor.submit(load_and_convert_case, row, dataset): row for row in chunk.itertuples(index=False)}
            for future in as_completed(future_to_row):
                try:
                    loaded_data.append(future.result())
                except Exception as e:
                    print(f"Error loading record: {e}")
        
        if not loaded_data:
            continue
            
        records = [data[0] for data in loaded_data]
        original_records = [data[1] for data in loaded_data]

        st = time.time()
        if beat_type == 'sampled':
            aladin.segmenter.batch(records)
            aladin.reflection.batch(records)
        elif beat_type == 'median':
            aladin.extract_median_beat_batch(records)
        
        if plot_only:
            plot_dir = os.path.join(out_dir, segment_type)
            os.makedirs(plot_dir, exist_ok=True)

            for record in records:
                if dataset == 'medalcare-xl':
                    run_id = record.original_file_path.split('/')[-2].split('_')[1]
                    session_id = record.original_file_path.split('/')[-1].split('_')[0]
                    label = record.groundtruth.replace('.', '')
                    aladin.plot(record, name=os.path.join(plot_dir, f'{run_id}_{session_id}_{label}_ecg.png'))
                else:
                    run_id = os.path.splitext(record.original_file_path)[0]
                    aladin.plot(record, name=os.path.join(plot_dir, f'{run_id}_ecg.png'))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = []
                for rec, orig in zip(records, original_records):
                    futures.append(executor.submit(process_and_save_segments, rec, orig, segment_type, out_dir, beat_type, dataset, plot=demo))
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error saving segment: {e}")
        
        del loaded_data 
        del records 
        del original_records
        gc.collect()

    print("Processing complete!")