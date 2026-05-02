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

# --- superclass ---
agg_super = scp[scp.diagnostic == 1].copy()
# map to superclass

def get_superclass(y_dic, agg_df):
    super_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    out = np.zeros(len(super_classes))
    for key in y_dic:
        if key in agg_df.index:
            if agg_df.loc[key].diagnostic_class in super_classes:
                out[super_classes.index(agg_df.loc[key].diagnostic_class)] = 1
    return out

df['superclass'] = df.scp_codes.apply(
    lambda x: get_superclass(x, agg_super)
)

# --- subclass ---
agg_sub = scp[scp.diagnostic == 1].copy()
df['subclass'] = df.scp_codes.apply(
    lambda x: aggregate_labels(x, agg_sub)
)

# --- form ---
agg_form = scp[scp.form == 1].copy()
df['form'] = df.scp_codes.apply(
    lambda x: aggregate_labels(x, agg_form)
)

# --- rhythm ---
agg_rhythm = scp[scp.rhythm == 1].copy()
df['rhythm'] = df.scp_codes.apply(
    lambda x: aggregate_labels(x, agg_rhythm)
)

cols = ['filename_hr', 'superclass', 'subclass', 'form', 'rhythm', 'strat_fold']

train_df = df[df.strat_fold <= 8][cols].copy()
val_df   = df[df.strat_fold == 9][cols].copy()
test_df  = df[df.strat_fold == 10][cols].copy()



base_path = "/home/tchatupanyacho/project/ptb_xl/physionet.org/files/ptb-xl/1.0.3/"

for df_ in [train_df, val_df, test_df]:

    df_["filename_hr"] = df_["filename_hr"].apply(

        lambda x: os.path.join(base_path, str(x))

    )


    df_.rename(columns={"filename_hr": "data_path"}, inplace=True)

test_df.to_csv("/home/tchatupanyacho/project/ptb_xl/data_split/ptb-xl_test.csv", index=False)
val_df.to_csv("/home/tchatupanyacho/project/ptb_xl/data_split/ptb-xl_val.csv", index=False)
train_df.to_csv("/home/tchatupanyacho/project/ptb_xl/data_split/ptb-xl_train.csv", index=False)