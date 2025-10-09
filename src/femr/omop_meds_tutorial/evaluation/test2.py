from pathlib import Path 
import math
import pandas as pd
import pickle
import meds_reader
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta


disease_dir = Path("/shared/share_mala/zj2398/mimic/regression_labels_nona/")
task="bilirubin"
meds_path = "/user/zj2398/cache/mimic/meds_v0.6_reader"

disease_path = disease_dir/f"{task}_regression.parquet"
# df = pd.read_parquet(disease_path)

database = meds_reader.SubjectDatabase(meds_path)
# print(database[10000032])
# print(disease_file)
# Ensure types
for parquet_file in disease_dir.glob("*.parquet"):
    print(parquet_file.name)
    df = pd.read_parquet(parquet_file)
    print(f"parquet_file {len(df)}")
    df["patient_id"] = df["patient_id"].astype(int)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # 2) Build unique patient list from the dataframe (so we only fetch what we need)
    all_pids = df["patient_id"].unique().tolist()

    # --- Worker function: open SubjectDatabase inside the process, compute first non-None event time per patient ---
    def first_event_times_for_chunk(pids_chunk, meds_path_str):
        from datetime import datetime
        import meds_reader as _mr

        db = _mr.SubjectDatabase(meds_path_str)

        out = {}
        for pid in pids_chunk:
            try:
                subj = db[pid]
            except KeyError:
                # no subject -> skip
                continue
            first_t = None
            for ev in subj.events:
                # ev.time can be None; keep the earliest non-None
                if ev.time is not None and ev.code != "MEDS_BIRTH":
                    first_t = ev.time
                    break
                    # if (first_t is None) or (ev.time < first_t):
                    #     first_t = ev.time
            if first_t is not None:
                out[pid] = pd.Timestamp(first_t)  # normalize to pandas Timestamp
        return out

    # 3) Parallelize over patient chunks (each worker builds its own SubjectDatabase handle)
    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i+size]

    pid_chunks = list(chunked(all_pids, 1000))
    pid_to_first = {}

    with ProcessPoolExecutor(max_workers=128) as ex:
        futs = [ex.submit(first_event_times_for_chunk, ch, str(meds_path)) for ch in pid_chunks]
        for fut in as_completed(futs):
            pid_to_first.update(fut.result())

    # 4) Map first event times back to the dataframe
    df["first_event_time"] = df["patient_id"].map(pid_to_first)

    # Drop rows for patients with no valid first event time
    df = df.dropna(subset=["first_event_time"])

    # 5) Keep only rows within the first year after the first event (and not earlier)
    one_year = pd.Timedelta(days=0)
    mask = df["time"] >= (df["first_event_time"]+ one_year)
    filtered = df.loc[mask, ["patient_id", "time", "code", "unit"]]  # keep any columns you want
    filtered = filtered.sort_values(["patient_id", "time"]).drop_duplicates("patient_id", keep="first")

    # (Optional) Save
    # filtered.to_parquet("/path/to/filtered.parquet", index=False)
    # store_path = "/shared/share_mala/zj2398/mimic/regression_labels_2_years/"+ parquet_file.name
    # filtered.to_parquet(store_path)

    print(f"Kept {len(filtered):,} of {len(df):,} rows within 1 year of first event.")
