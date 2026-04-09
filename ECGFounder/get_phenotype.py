rom _future_ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PHENO_FILE_DEFAULT = (
    "/gpfs/work2/0/aus20644/data/ukbiobank/phenotypes/ukb678882.tab.gz"
)
EIDS_FILE_DEFAULT = '/projects/prjs1890/uk_biobank/eid.txt'

OUTPUT_DEFAULT = '/projects/prjs1890/uk_biobank/phenotype_targets_2.pt'

# Ordered phenotype definitions``
PHENOTYPES = [
    # (name,    field_id,  task_type)
    ("lvef",    "24103",   "regression"),
    ("lvedv",   "24100",   "regression"),
    ("lv_mass", "24105",   "regression"),
    ("rvedv",   "24106",   "regression"),
    ("lvgls",   "24181",   "regression"),
    # ("afib",    "131350",  "binary"),
    # ("mi",      "131298",  "binary"),
    ("sex",     "31",   "regression"),
    ("age",     "34",   "regression"),
    ("p_dur",     "12338",   "regression"),
    ("pq_int",     "22330",   "regression"),
    ("pp_int",     "22334",   "regression"),
    ("qrs_dur",     "12340",   "regression"),
    ("qtc_int",     "22332",   "regression"),
    ("heart_rate",     "12336",   "regression"),
    ("p_ax",     "22335",   "regression"),
    ("r_ax",     "22336",   "regression"),
    ("t_ax",     "22337",   "regression"),
    ("lasv",     "24112",   "regression"),
    ("laef",     "24113",   "regression"),
]

COLUMNS   = [p[0] for p in PHENOTYPES]
FIELDS    = [p[1] for p in PHENOTYPES]
TYPES     = [p[2] for p in PHENOTYPES]
BINARY_COLS = [p[0] for p in PHENOTYPES if p[2] == "binary"]

# Coding-819 sentinel dates – ALL treated as positive (ICD code IS recorded)
SENTINEL_DATES = {
    "1900-01-01", "1901-01-01", "1902-02-02",
    "1903-03-03", "1909-09-09", "2037-07-07",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_columns(header: list[str], field_id: str) -> list[str]:
    """Return all header columns that belong to a given UKB field ID.

    Matches patterns: `⁠ f.<field>.x.x ⁠⁠ , ``<field>-x.x ⁠⁠ , exact ``<field> ⁠`.
    """
    out = []
    for col in header:
        if (
            col == field_id
            or col.startswith(f"{field_id}-")
            or col.startswith(f"f.{field_id}.")
        ):
            out.append(col)
    return out


def peek_header(pheno_file: str) -> list[str]:
    """Read just the first line of the (optionally gzipped) file."""
    if pheno_file.endswith(".gz"):
        import gzip
        with gzip.open(pheno_file, "rt") as fh:
            return fh.readline().strip().split("\t")
    with open(pheno_file) as fh:
        return fh.readline().strip().split("\t")


def detect_eid_col(header: list[str]) -> str:
    """Return the name of the participant-ID column."""
    for candidate in ("eid", "f.eid", "Participant_ID", header[0]):
        if candidate in header:
            return candidate
    raise ValueError("Cannot identify EID column in header")


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build(
    pheno_file: str,
    eids_file: str,
    output_path: str,
    chunksize: int = 50_000,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load requested EIDs ------------------------------------------------
    print(f"Loading EIDs from {eids_file}...")
    with open(eids_file) as fh:
        eids = [line.strip() for line in fh if line.strip()]
    n = len(eids)
    print(f"  {n:,} EIDs loaded.")
    eids_set = set(eids)

    # 2. Inspect header -----------------------------------------------------
    print(f"\nPeeking at header of {pheno_file}...")
    header = peek_header(pheno_file)
    eid_col = detect_eid_col(header)
    print(f"  EID column: '{eid_col}'  |  total header cols: {len(header):,}")

    field_to_cols: dict[str, list[str]] = {}
    cols_to_load: set[str] = {eid_col}

    for _name, field_id, _tt in PHENOTYPES:
        matched = find_columns(header, field_id)
        field_to_cols[field_id] = matched
        cols_to_load.update(matched)
        status = f"{len(matched)} col(s): {matched}" if matched else "NOT FOUND"
        print(f"  field {field_id:>7} → {status}")

    # 3. Read file in chunks, keep only requested EIDs ----------------------
    print(f"\nReading {pheno_file} in chunks of {chunksize:,}...")
    read_kw: dict = {
        "sep": "\t",
        "index_col": False,
        "usecols": lambda c: c in cols_to_load,
        "dtype": {eid_col: str},
        "chunksize": chunksize,
    }
    if pheno_file.endswith(".gz"):
        read_kw["compression"] = "gzip"

    chunks = []
    for chunk in tqdm(pd.read_csv(pheno_file, **read_kw), desc="  chunks"):
        chunk[eid_col] = chunk[eid_col].astype(str).str.strip()
        sub = chunk[chunk[eid_col].isin(eids_set)]
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        raise RuntimeError("No matching EIDs found in the phenotype file.")

    df = pd.concat(chunks, ignore_index=True)
    # De-duplicate (keep first occurrence per EID)
    df = df.drop_duplicates(subset=[eid_col], keep="first")
    df = df.set_index(eid_col)
    print(f"  Matched {len(df):,} participants in file.")

    # 4. Build float array (N × 7) ------------------------------------------
    targets = np.full((n, len(PHENOTYPES)), np.nan, dtype=np.float32)

    for col_idx, (name, field_id, task_type) in enumerate(
        tqdm(PHENOTYPES, desc="Processing phenotypes")
    ):
        matched_cols = field_to_cols.get(field_id, [])
        if not matched_cols:
            print(f"  [WARN] '{name}' (field {field_id}): no columns found – all NaN")
            continue

        for row_idx, eid in enumerate(eids):
            if eid not in df.index:
                # EID not in phenotype file → NaN (already set)
                continue

            row = df.loc[eid]

            if task_type == "regression":
                # Use only the first column (first imaging visit);
                # subsequent columns are repeat visits and are ignored.
                col = matched_cols[0]
                v = row[col] if col in df.columns else np.nan
                if pd.notna(v):
                    try:
                        targets[row_idx, col_idx] = float(v)
                    except (ValueError, TypeError):
                        pass

            elif task_type == "binary":
                # Any non-null value (including Coding-819 sentinel dates)
                # indicates a recorded ICD event → positive (1.0).
                # Only one column exists for these fields (131350, 131298).
                col = matched_cols[0]
                v = row[col] if col in df.columns else np.nan
                targets[row_idx, col_idx] = 1.0 if pd.notna(v) else 0.0

        n_valid = int(np.sum(~np.isnan(targets[:, col_idx])))
        if task_type == "binary":
            n_pos = int(np.nansum(targets[:, col_idx]))
            print(
                f"  {name:10s}: {n_valid:,} labelled  |  "
                f"{n_pos:,} positive  {n_valid - n_pos:,} negative"
            )
        else:
            vals = targets[:, col_idx][~np.isnan(targets[:, col_idx])]
            print(
                f"  {name:10s}: {n_valid:,} valid  |  "
                f"mean {vals.mean():.2f}  std {vals.std():.2f}  "
                f"[{vals.min():.2f}, {vals.max():.2f}]"
            )

    # 5. Save ---------------------------------------------------------------
    payload = {
        "eids":        eids,
        "targets":     torch.from_numpy(targets),   # (N, 7) float32, NaN for missing
        "columns":     COLUMNS,                      # list[str]
        "fields":      FIELDS,                       # list[str]
        "binary_cols": BINARY_COLS,                  # list[str]
    }

    print(f"\nSaving → {output}")
    torch.save(payload, output)

    # 6. Summary ------------------------------------------------------------
    print("\n=== Summary ===")
    for i, (name, field_id, task_type) in enumerate(PHENOTYPES):
        col = targets[:, i]
        n_valid = int(np.sum(~np.isnan(col)))
        coverage = 100.0 * n_valid / n
        if task_type == "binary":
            n_pos = int(np.nansum(col))
            prevalence = 100.0 * n_pos / n_valid if n_valid > 0 else float("nan")
            print(
                f"  {name:10s} (field {field_id}): "
                f"{n_valid:>6,}/{n:,} ({coverage:5.1f}%) labelled | "
                f"prevalence {prevalence:.1f}%"
            )
        else:
            vals = col[~np.isnan(col)]
            print(
                f"  {name:10s} (field {field_id}): "
                f"{n_valid:>6,}/{n:,} ({coverage:5.1f}%) valid | "
                f"mean {vals.mean():.3f}  std {vals.std():.3f}"
                if len(vals) > 0 else
                f"  {name:10s} (field {field_id}): 0 valid"
            )

    tensor_shape = payload["targets"].shape
    print(f"\nSaved tensor shape: {tensor_shape}  dtype: {payload['targets'].dtype}")
    print(f"Output: {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build phenotype_targets.pt aligned to cache/eids.txt"
    )
    parser.add_argument(
        "--pheno_file", default=PHENO_FILE_DEFAULT,
        help="Path to the UKB tabular file (.tab.gz or .tsv[.gz])",
    )
    parser.add_argument(
        "--eids_file", default=EIDS_FILE_DEFAULT,
        help="Path to eids.txt (one EID per line, same order desired in output)",
    )
    parser.add_argument(
        "--output", default=OUTPUT_DEFAULT,
        help="Output path for phenotype_targets.pt",
    )
    parser.add_argument(
        "--chunksize", type=int, default=50_000,
        help="Number of rows per chunk when reading the phenotype file",
    )
    args = parser.parse_args()

    build(
        pheno_file=args.pheno_file,
        eids_file=args.eids_file,
        output_path=args.output,
        chunksize=args.chunksize,
    )


if _name_ == "_main_":
    main()