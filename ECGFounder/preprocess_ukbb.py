import numpy as np
import argparse
from pathlib import Path
import sys
import xmltodict
from scipy.io import savemat
from scipy.interpolate import interp1d
from collections import OrderedDict
import traceback
from util import filter_bandpass

# Standard 12-lead order
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
TARGET_SAMPLE_RATE = 500  # Hz
TARGET_SEGMENT_SECONDS = 10  # seconds
TARGET_SEGMENT_SIZE = TARGET_SAMPLE_RATE * TARGET_SEGMENT_SECONDS

def xml_to_dict(file: str):
    try:
        with open(file, "r", encoding="ISO-8859-1") as f:
            xml_string = f.read()
        return xmltodict.parse(xml_string)
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

def parse_waveform_data(waveform_str: str) -> np.ndarray:
    values = waveform_str.strip().split(',')
    values = [int(v.strip()) for v in values if v.strip()]
    return np.array(values, dtype=np.float32)

def augment_leads(leads):
    if "I" not in leads or "II" not in leads: return leads
    if "III" not in leads: leads["III"] = leads["II"] - leads["I"]
    if "aVR" not in leads: leads["aVR"] = -(leads["I"] + leads["II"]) / 2
    if "aVL" not in leads: leads["aVL"] = (leads["I"] - leads["III"]) / 2
    if "aVF" not in leads: leads["aVF"] = (leads["II"] + leads["III"]) / 2
    return leads

def resample(feats, curr_rate, target_rate):
    if curr_rate == target_rate: return feats
    target_size = int(feats.shape[-1] * (target_rate / curr_rate))
    x = np.linspace(0, target_size - 1, feats.shape[-1])
    features = interp1d(x, feats, kind='linear')(np.arange(target_size))
    return features

def standardize_global(feats: np.ndarray):
    """Global Instance Normalization:
    1. Center each lead (remove DC offset)
    2. Divide ALL leads by the SAME global standard deviation (preserve relative amplitude)
    """
    # 1. Per-lead centering
    mean = feats.mean(axis=1, keepdims=True)
    feats = feats - mean
    
    # 2. Global scaling
    std_global = feats.std() 
    std_global = 1.0 if std_global == 0 else std_global
    feats = feats / (std_global + 1e-8)
    
    # Return formatted to match conventions
    return feats, mean.flatten(), np.full(12, std_global)

def process_file(xml_path, out_dir):
    eid = xml_path.stem.split('_')[0]
    out_path = out_dir / f"{eid}.npy"
    
    if out_path.exists():
        print(f"Skipping {eid}, already done.")
        return

    print(f"Processing {eid} from {xml_path.name}...")
    
    xml_dict = xml_to_dict(str(xml_path))
    if not xml_dict: return

    try:
        root = xml_dict['CardiologyXML']
        strip_data = root.get('StripData', {})
        sample_rate = int(strip_data.get('SampleRate', {}).get('#text', 500))
        resolution = float(strip_data.get('Resolution', {}).get('#text', 5))
        
        waveform_data = strip_data.get('WaveformData', [])
        if not isinstance(waveform_data, list): waveform_data = [waveform_data]
        
        leads = OrderedDict()
        for wf in waveform_data:
            lead_name = wf.get('@lead', '')
            if lead_name in LEAD_ORDER:
                waveform = parse_waveform_data(wf.get('#text', ''))
                waveform = waveform * resolution / 1000.0  # mV
                leads[lead_name] = waveform
        
        if len(leads) < 2: return

        leads = augment_leads(leads)
        feats = np.stack([leads.get(l, np.zeros_like(list(leads.values())[0])) for l in LEAD_ORDER])
        feats = np.nan_to_num(feats, nan=0)
        feats = filter_bandpass(feats, TARGET_SAMPLE_RATE) 
        
        if feats.shape[1] < TARGET_SEGMENT_SIZE:
             pad = TARGET_SEGMENT_SIZE - feats.shape[1]
             feats = np.pad(feats, ((0,0), (0, pad)), mode='constant')
        feats = feats[:, :TARGET_SEGMENT_SIZE]
        
        # Apply GLOBAL Norm
        feats = (feats - np.mean(feats)) / (np.std(feats) +1e-8)
        
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, feats.astype(np.float32))
        print(f"  Saved {out_path}")
        
    except Exception as e:
        print(f"Error processing {eid}: {e}")
        # traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg_dir", type=Path, required=True, help="Input directory")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--max_samples", type=int, default=10, help="Number of files to process")
    args = parser.parse_args()

    files = sorted(list(args.ecg_dir.glob("*.xml")))
    print(f"Found {len(files)} XML files in {args.ecg_dir}")
    
    count = 0
    for f in files:
        if count >= args.max_samples: break
        process_file(f, args.out_dir)
        count += 1

if __name__ == "__main__":
    main()