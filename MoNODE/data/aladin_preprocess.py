import json
import pandas as pd 
import os
import glob
import wfdb
from tqdm import tqdm
import ast
import argparse

from aladin import ALADIN
from aladin.core import Record

import matplotlib.pyplot as plt


MEDALCARE_XL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
def load_case(dir, case, metadata):

    file = os.path.join(dir,case)
    rec = wfdb.rdrecord(file)
    ecg = {name: rec.p_signal[:, i] for i, name in enumerate(rec.sig_name)}

    record = Record(ecg, rec.fs, "DEMO", case)
    record.groundtruth = metadata.label
    record.hash = metadata.hash

    return record

def analyse_single_case(record):

    aladin = ALADIN(modelpaths=["ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": True, "afibdetector": False, "reflection": False, "total": True})
    aladin.analyse(record)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Load and analyze ECG cases")
    argparser.add_argument("--input_path", type=str, help="Path to the input directory", required=True)
    args = argparser.parse_args()


    # 1 : Convert file to .dat and .hea format
    df = pd.read_csv(args.input_path, nrows=1)
    for row in df.itertuples(index=False):
        ecg_path = str(row.data_path)
        print(f'Processing {ecg_path}')

        case = ecg_path.split("/")[-1].split(".")[0] 
        directory_path = ecg_path.split("/")[:-1]
        directory_path = "/".join(directory_path)

        filepath = os.path.join(directory_path, case)
        print(f'Checking if {filepath}.dat and {filepath}.hea exist')
        if not os.path.exists(filepath + '.dat') or not os.path.exists(filepath + '.hea'):
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

        record = load_case(directory_path, case, row)
        analyse_single_case(record)