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
from concurrent.futures import ThreadPoolExecutor, as_completed

from aladin import ALADIN
from aladin.core import Record

import matplotlib.pyplot as plt
from pprint import pprint 

MEDALCARE_XL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def load_and_convert_case(row):
    """Worker function to handle file checking, conversion, and loading."""
    ecg_path = str(row.data_path)
    directory_path, filename = os.path.split(ecg_path)
    case = os.path.splitext(filename)[0]
    
    filepath = os.path.join(directory_path, case)

    # Convert if missing
    if not os.path.exists(filepath + '.dat') or not os.path.exists(filepath + '.hea'):
        ecg = pd.read_csv(ecg_path, header=None).to_numpy()
        
        # Ensure shape is (samples, leads)
        if ecg.shape[0] < ecg.shape[1]:
            ecg = ecg.T
            
        wfdb.wrsamp(
            record_name=case, 
            write_dir=directory_path,
            fs=500, 
            units=['mV'] * ecg.shape[1], 
            sig_name=MEDALCARE_XL_LEADS[:ecg.shape[1]], 
            p_signal=ecg, 
            fmt=['32'] * ecg.shape[1]
        )    

    # Load record
    rec = wfdb.rdrecord(filepath)
    ecg_dict = {name: rec.p_signal[:, i] for i, name in enumerate(rec.sig_name)}

    record = Record(ecg_dict, rec.fs, "DEMO", case)
    record.groundtruth = row.label 
    record.hash = row.hash        
    return record, rec

def process_and_save_segments(record, original_record, segment_type, out_dir, beat_type, plot=False):
    """Worker function to handle normalization and saving to disk."""
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
                
            # Safe sample in case there are fewer than 2 segments
            k = min(2, len(qrst_idx))
            segments = random.sample(qrst_idx, k) if k > 0 else []
        elif beat_type == 'median':
            start = int(record.median_beat.delineations.qrs.onset)
            end = int(record.median_beat.delineations.t.offset)

            segments = [(start, end)]
            if start is None or end is None:
                return # Skip if median beat is not properly delineated
        
    elif segment_type == 'atrial':
        if beat_type == 'sampled':
            k = min(2, len(record.p))
            sampled_p = random.sample(record.p, k) if k > 0 else []
            segments = [(int(seg.onset), int(seg.offset)) for seg in sampled_p]
        elif beat_type == 'median':
            start = int(record.median_beat.delineations.p.onset)
            end = int(record.median_beat.delineations.p.offset)

            segments = [(start, end)]
            if start is None or end is None:
                return # Skip if median beat is not properly delineated

    if not segments:
        return # Skip if no valid segments found

    original_ecg = original_record.p_signal if beat_type == 'sampled' else record.median_beat.ecg.T
    print(f'ECG for {beat_type} {segment_type} has shape {original_ecg.shape}')

    mu = np.mean(original_ecg, axis=0, keepdims=True)
    sigma = np.std(original_ecg, axis=0, keepdims=True)
    norm_ecg = (original_ecg - mu) / (sigma + 1e-8)

    save_dir = os.path.join(out_dir, beat_type)
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, (start, end) in enumerate(segments):
        ecg_segment = norm_ecg[start:end, :]
        delta_t = end - start
        
        if beat_type == 'sampled':
            base_name = f'T{delta_t}_{record.hash}_{record.groundtruth}_{idx}'
        else:
            base_name = f'T{delta_t}_{record.hash}_{record.groundtruth}'

        np.save(os.path.join(save_dir, f'{base_name}.npy'), ecg_segment.astype(np.float32))

        if plot:
            if beat_type == 'median':
                plt.figure(figsize=(10, 4))
                plt.plot(original_record.p_signal[:, 0], label='Original ECG')
                plt.xlabel('Time')
                plt.ylabel('Amplitude (mV)')
                plt.legend()
                plt.savefig(f'{base_name}_full_ecg_{beat_type}.png')
                plt.close() 

                plt.figure(figsize=(10, 4))
                plt.plot(ecg_segment[:, 0], label='Normalised median beat')
                plt.axvspan(start, end, color='#e74c3c', alpha=0.5, label=f"Segmented for {segment_type}")
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.savefig(f'{base_name}_norm_segment_{beat_type}.png')
                plt.close() 

                plt.figure(figsize=(10, 4))
                plt.plot(record.median_beat.ecg[0, :], label='Median beat')
                plt.axvspan(start, end, color='#e74c3c', alpha=0.5, label=f"Segmented for {segment_type}")
                plt.xlabel('Time')
                plt.ylabel('Amplitude (mV)')
                plt.legend()
                plt.savefig(f'{base_name}_segment_{beat_type}.png')
                plt.close()
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Load and analyze ECG cases")
    argparser.add_argument("--input_path", type=str, help="Path to the input directory", required=True)
    argparser.add_argument("--batch_size", type=int, help="Number of cases to process in each batch", default=32) # Increased batch size
    argparser.add_argument("--out_dir", type=str, help="Path to the output directory", required=True)
    argparser.add_argument("--workers", type=int, help="Number of parallel workers", default=os.cpu_count())
    argparser.add_argument("--beat_type", type=str, choices=['sampled', 'median'], help="Types of beat to segment", required=True)
    argparser.add_argument("--demo", action='store_true', help="Run a quick demo with a subset of data")

    args = argparser.parse_args()

    segment_type = os.path.splitext(args.input_path)[0].split('_')[-1]
    beat_type = args.beat_type
    split = os.path.splitext(args.input_path)[0].split('_')[-2]
    demo = args.demo
    print(f'Processing {split} split for {segment_type} segments')
    
    out_dir = os.path.join(args.out_dir, split, segment_type)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading ALADIN model into memory...")
    aladin = ALADIN(
        modelpaths=["ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
        debug={"segmenter": True, "afibdetector": False, "reflection": False, "total": False}
    )

    df = pd.read_csv(args.input_path) if not demo else pd.read_csv(args.input_path, nrows=3)
    
    chunks = [df.iloc[i:i + args.batch_size] for i in range(0, len(df), args.batch_size)]
    
    print(f"Starting processing with {args.workers} workers...")
    
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Processing Batches")):
        
        loaded_data = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_row = {executor.submit(load_and_convert_case, row): row for row in chunk.itertuples(index=False)}
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
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for rec, orig in zip(records, original_records):
                futures.append(executor.submit(process_and_save_segments, rec, orig, segment_type, out_dir, beat_type, plot=demo))
            
            for future in as_completed(futures):
                try:
                    future.result() # Catch any save errors
                except Exception as e:
                    print(f"Error saving segment: {e}")

    print("Processing complete!")