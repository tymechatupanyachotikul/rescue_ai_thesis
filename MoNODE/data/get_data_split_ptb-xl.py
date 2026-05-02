import os
import pandas as pd
import numpy as np
import ast

# load data
df = pd.read_csv('/home/tchatupanyacho/project/ptb_xl/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv', index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(ast.literal_eval)

# load scp statements
scp = pd.read_csv('/home/tchatupanyacho/project/ptb_xl/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv', index_col=0)

# helper to aggregate labels
def aggregate_labels(y_dic, agg_df):
    out = np.zeros(len(agg_df))
    for key, val in y_dic.items():
        if key in agg_df.index:
            out[agg_df.index.get_loc(key)] = 1
    return out

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def get_superclass(y_dic, agg_df):
    out = np.zeros(len(SUPERCLASSES))
    for key in y_dic:
        if key in agg_df.index:
            diag_class = agg_df.loc[key].diagnostic_class
            if diag_class in SUPERCLASSES:
                out[SUPERCLASSES.index(diag_class)] = 1
    return out

# --- labels ---
agg_super  = scp[scp.diagnostic == 1].copy()
agg_sub    = scp[scp.diagnostic == 1].copy()
agg_form   = scp[scp.form == 1].copy()
agg_rhythm = scp[scp.rhythm == 1].copy()

df['superclass'] = df.scp_codes.apply(lambda x: get_superclass(x, agg_super))
df['subclass']   = df.scp_codes.apply(lambda x: aggregate_labels(x, agg_sub))
df['form']       = df.scp_codes.apply(lambda x: aggregate_labels(x, agg_form))
df['rhythm']     = df.scp_codes.apply(lambda x: aggregate_labels(x, agg_rhythm))

# --- splits ---
base_path = "/home/tchatupanyacho/project/ptb_xl/physionet.org/files/ptb-xl/1.0.3/"
cols = ['filename_hr', 'superclass', 'subclass', 'form', 'rhythm', 'strat_fold']

def process_split(df_, base_path):
    df_ = df_.copy()
    df_['data_path'] = df_['filename_hr'].apply(
        lambda x: os.path.join(base_path, str(x))
    )
    df_ = df_.drop(columns=['filename_hr', 'strat_fold'])
    return df_

train_df = process_split(df[df.strat_fold <= 8][cols], base_path)
val_df   = process_split(df[df.strat_fold == 9][cols],  base_path)
test_df  = process_split(df[df.strat_fold == 10][cols], base_path)

# sanity check
total = len(train_df) + len(val_df) + len(test_df)
print(f"original : {len(df)}")
print(f"train    : {len(train_df)}")
print(f"val      : {len(val_df)}")
print(f"test     : {len(test_df)}")
print(f"total    : {total}")
assert total == len(df), f"Row count mismatch! {total} != {len(df)}"

# --- save ---
out_dir = "/home/tchatupanyacho/project/ptb_xl/data_split"
os.makedirs(out_dir, exist_ok=True)

train_df.to_csv(os.path.join(out_dir, "ptb-xl_train.csv"), index=False)
val_df.to_csv(os.path.join(out_dir,   "ptb-xl_val.csv"),   index=False)
test_df.to_csv(os.path.join(out_dir,  "ptb-xl_test.csv"),  index=False)

print("Saved successfully.")