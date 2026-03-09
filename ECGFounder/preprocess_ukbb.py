import numpy as np
import argparse
from pathlib import Path
import xmltodict
from scipy.interpolate import interp1d
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from util import filter_bandpass

LEAD_ORDER    = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_SET      = set(LEAD_ORDER)
TARGET_RATE   = 500
TARGET_SECS   = 10
TARGET_SIZE   = TARGET_RATE * TARGET_SECS


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def augment_leads(leads: dict) -> dict:
    if "I" not in leads or "II" not in leads:
        return leads
    I, II = leads["I"], leads["II"]
    leads.setdefault("III", II - I)
    III = leads["III"]
    leads.setdefault("aVR", -(I + II) / 2)
    leads.setdefault("aVL",  (I - III) / 2)
    leads.setdefault("aVF",  (II + III) / 2)
    return leads


def resample(feats: np.ndarray, curr_rate: int, target_rate: int) -> np.ndarray:
    if curr_rate == target_rate:
        return feats
    target_size = int(feats.shape[-1] * (target_rate / curr_rate))
    x_old = np.linspace(0, target_size - 1, feats.shape[-1])
    return interp1d(x_old, feats, kind='linear', assume_sorted=True)(np.arange(target_size))


# ---------------------------------------------------------------------------
# Per-file worker  (must be top-level for pickling)
# ---------------------------------------------------------------------------

def process_file(task: tuple) -> tuple[str, str]:
    xml_path, out_dir = task
    eid      = xml_path.stem.split('_')[0]
    out_path = out_dir / f"{eid}.npy"

    if out_path.exists():
        return eid, "skipped"

    try:
        # -- Parse XML (binary read avoids Python str decode overhead) ------
        with open(xml_path, "rb") as fh:
            xml_dict = xmltodict.parse(fh.read())

        root       = xml_dict['CardiologyXML']
        strip_data = root.get('StripData', {})
        resolution = float(strip_data.get('Resolution', {}).get('#text', 5))

        waveform_data = strip_data.get('WaveformData', [])
        if not isinstance(waveform_data, list):
            waveform_data = [waveform_data]

        # -- Build lead dict -------------------------------------------------
        leads: dict[str, np.ndarray] = {}
        scale = resolution / 1000.0           # integer counts → mV
        for wf in waveform_data:
            name = wf.get('@lead', '')
            if name not in LEAD_SET:
                continue
            # np.fromstring is significantly faster than split + list-comp
            raw = np.fromstring(wf.get('#text', ''), dtype=np.float32, sep=',')
            leads[name] = raw * scale

        if len(leads) < 2:
            return eid, "error: insufficient leads"

        # -- Augment, stack, clean -------------------------------------------
        leads = augment_leads(leads)
        ref   = next(iter(leads.values()))
        feats = np.stack(
            [leads.get(l, np.zeros(len(ref), dtype=np.float32)) for l in LEAD_ORDER]
        )
        np.nan_to_num(feats, nan=0.0, copy=False)   # in-place

        # -- Filter ----------------------------------------------------------
        feats = filter_bandpass(feats, TARGET_RATE)

        # -- Pad / trim to exact target length --------------------------------
        n = feats.shape[1]
        if n < TARGET_SIZE:
            feats = np.pad(feats, ((0, 0), (0, TARGET_SIZE - n)), mode='constant')
        feats = feats[:, :TARGET_SIZE]

        # -- Global normalisation --------------------------------------------
        std = feats.std()
        feats = (feats - feats.mean()) / (std + 1e-8)

        # -- Save ------------------------------------------------------------
        np.save(out_path, feats.astype(np.float32))
        return eid, "done"

    except Exception as exc:
        return eid, f"error: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg_dir",     type=Path, required=True)
    parser.add_argument("--out_dir",     type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker count (default: all logical CPUs)"
    )
    args = parser.parse_args()

    files = sorted(args.ecg_dir.glob("*.xml"))
    print(f"Found {len(files)} XML files → {args.ecg_dir}")

    # Create output dir once here so workers never race on mkdir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or multiprocessing.cpu_count()
    n_workers = min(n_workers, len(files))   # no idle processes
    tasks     = [(f, args.out_dir) for f in files]

    print(f"Processing with {n_workers} parallel workers...")
    done = skipped = failed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_file, t): t[0] for t in tasks}
        with tqdm(total=len(tasks), desc="Processing ECGs", unit="file") as pbar:
            for future in as_completed(futures):
                eid, status = future.result()
                if   status == "done":    done    += 1
                elif status == "skipped": skipped += 1
                else:
                    failed += 1
                    tqdm.write(f"  ✗ [{eid}] {status}")
                pbar.update(1)

    print(f"\nDone: {done} | Skipped (cached): {skipped} | Failed: {failed}")


if __name__ == "__main__":
    main()
