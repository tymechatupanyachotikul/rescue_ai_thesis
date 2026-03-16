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

def process_and_save_segments(record, original_record, segment_type, out_dir):
    """Worker function to handle normalization and saving to disk."""
    segments = []
    
    if segment_type == 'ventricular':
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
        
    elif segment_type == 'atrial':
        k = min(2, len(record.p))
        sampled_p = random.sample(record.p, k) if k > 0 else []
        segments = [(int(seg.onset), int(seg.offset)) for seg in sampled_p]

    if not segments:
        return # Skip if no valid segments found

    mu = np.mean(original_record.p_signal, axis=0, keepdims=True)
    sigma = np.std(original_record.p_signal, axis=0, keepdims=True)
    norm_ecg = (original_record.p_signal - mu) / (sigma + 1e-8)

    save_dir = os.path.join(out_dir, 'sampled')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the full original record ONCE per patient, not per segment
    for idx, (start, end) in enumerate(segments):
        ecg_segment = norm_ecg[start:end, :]
        delta_t = end - start
        
        base_name = f'T{delta_t}_{record.hash}_{record.groundtruth}_{idx}'
        np.save(os.path.join(save_dir, f'{base_name}.npy'), ecg_segment.astype(np.float32))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Load and analyze ECG cases")
    argparser.add_argument("--input_path", type=str, help="Path to the input directory", required=True)
    argparser.add_argument("--batch_size", type=int, help="Number of cases to process in each batch", default=32) # Increased batch size
    argparser.add_argument("--out_dir", type=str, help="Path to the output directory", required=True)
    argparser.add_argument("--workers", type=int, help="Number of parallel workers", default=os.cpu_count())
    args = argparser.parse_args()

    segment_type = os.path.splitext(args.input_path)[0].split('_')[-1]
    split = os.path.splitext(args.input_path)[0].split('_')[-2]
    print(f'Processing {split} split for {segment_type} segments')
    
    out_dir = os.path.join(args.out_dir, split, segment_type)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading ALADIN model into memory...")
    aladin = ALADIN(
        modelpaths=["ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
        debug={"segmenter": True, "afibdetector": False, "reflection": False, "total": False}
    )

    df = pd.read_csv(args.input_path)
    
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
        aladin.segmenter.batch(records)
        aladin.reflection.batch(records)
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for rec, orig in zip(records, original_records):
                futures.append(executor.submit(process_and_save_segments, rec, orig, segment_type, out_dir))
            
            for future in as_completed(futures):
                try:
                    future.result() # Catch any save errors
                except Exception as e:
                    print(f"Error saving segment: {e}")

    print("Processing complete!")