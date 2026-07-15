#!/usr/bin/env python3
"""Rebuild the reAIM-Lab MIMIC ED-discharge dictionary (`ed_out_dict.pkl`) from raw MIMIC-IV.

The benchmark's ED prediction times are **`transfers.outtime` where `hadm_id` is NULL** -- i.e.
**non-admitted ("treat-and-release") ED departures**. The MIMIC->MEDS build emits no discharge event
for these, so the cohort injects them from this pickle.

Output: a dict `{subject_id: [outtime datetime, ...]}`, consumed by `mimic_cohort.py --ed-out-dict`.

    python build_ed_out_dict.py \
        --transfers /path/to/mimiciv/3.1/hosp/transfers.csv.gz \
        --output /tmp/ed_out_dict.pkl
"""
from __future__ import annotations

import argparse
import pickle

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transfers", required=True, help="MIMIC-IV hosp/transfers.csv.gz (raw)")
    ap.add_argument("--output", default="ed_out_dict.pkl", help="output pickle path")
    args = ap.parse_args()

    t = (pl.read_csv(args.transfers, columns=["subject_id", "hadm_id", "outtime"],
                     schema_overrides={"subject_id": pl.Int64, "hadm_id": pl.Int64})
         .filter(pl.col("hadm_id").is_null())                       # non-admitted => ED-only
         .with_columns(pl.col("outtime").str.to_datetime(strict=False, time_unit="us"))
         .drop_nulls("outtime")
         .select("subject_id", "outtime").unique().sort(["subject_id", "outtime"]))

    ed = {int(r["subject_id"]): list(r["outtime"])
          for r in t.group_by("subject_id").agg(pl.col("outtime")).to_dicts()}

    with open(args.output, "wb") as f:
        pickle.dump(ed, f)
    n_times = sum(len(v) for v in ed.values())
    print(f"ed_out_dict: {len(ed):,} subjects | {n_times:,} non-admitted-ED discharge times -> {args.output}")


if __name__ == "__main__":
    main()
