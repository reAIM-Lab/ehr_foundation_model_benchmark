#!/usr/bin/env python3
"""Build the reAIM-Lab MIMIC phenotype cohorts (AMI / MASLD / stroke) on a MIMIC-IV MEDS build (polars).

Extracted from https://github.com/reAIM-Lab/mimic_phenotype_cohort
(`phenotype/mimic/cohort_extract.py`).

Cohort logic (per subject, events sorted by time):
  - observation_start = first event time; an at-risk entry requires >= 2yr of history first.
  - AT-RISK entry  : the first event (>=730d after observation_start) that is a `<task>_at_risk` code
                     OR at which >=2 `<task>_clinical_risk_factor` codes have accumulated (AMI only).
  - CASE (outcome) : a `<task>_case` code that occurs AFTER an emergency visit (hospital/ICU admission
                     or ED registration), de-duplicated to the first per 7-day window.
  - PREDICTION time: a DISCHARGE event (hospital/ED) with time > at_risk_time, strictly before the
                     first case, and <= 730d since the previous event ("recent activity").
  - LABEL          : positive iff the next case is within 365d of the prediction time.
  - Excludes 4 hard-coded subject ids.

NOTE: the EMERGENCY / DISCHARGE code patterns are MEDS-build-specific -- VERIFY them against your
build first. meds_v0.6 typically uses HOSPITAL_ADMISSION / ICU_ADMISSION / ED_REGISTRATION and
HOSPITAL_DISCHARGE / ED_OUT.

Usage:
    python mimic_cohort.py --meds /path/to/MEDS_cohort/data \
        --codes icd_codes.json --task ami --ed-out-dict ed_out_dict.pkl --output ami.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import polars as pl

# --- MEDS-build-specific code patterns (verified against mimiciv meds_v0.6) -----------------------
EMERGENCY = r"^(HOSPITAL_ADMISSION|ICU_ADMISSION|ED_REGISTRATION)"   # inpatient / ICU / ED registration
DISCHARGE = r"^(HOSPITAL_DISCHARGE|ED_OUT)"                          # hospital + ED discharge (NOT ICU)

US = pl.duration(microseconds=1)
OBS_DAYS = 730          # 2-yr observation window before at-risk entry
LOOKAHEAD = 365         # positive if AMI within this many days of the prediction time
CASE_DEDUP_DAYS = 7     # collapse repeated case codes to the first per 7-day window
EXCLUDE = [11704827, 14129581, 17454346, 19824820]

# Per-task cohort parameters. Per reAIM cohort_extract.py, `stroke`/`ami` take the ACUTE-disease path
# (emergency-gated case, binary all-discharges) and `masld` the CHRONIC path (plain case, incident).
# The CRF >=2 counter is AMI-only; stroke has no clinical_risk_factor set -> single at-risk.
#  case_emergency : the case code must follow an emergency visit (True, acute) or is raw (False, chronic)
#  first_case     : incident -- keep only discharges strictly BEFORE the first case (True, chronic),
#                   or binary all-discharges (False, acute)
PARAMS = {
    "ami":    dict(case_emergency=True,  first_case=False),   # acute: binary all-discharges, encounter-gated, CRF-OR arm
    "stroke": dict(case_emergency=True,  first_case=False),   # acute: same as AMI; single at-risk (no CRF set)
    "masld":  dict(case_emergency=False, first_case=True),    # chronic: incident, first-MASLD, plain case
}


def load_ed_out(path: str) -> pl.DataFrame:
    """Load the reference ed_out_dict.pkl -> (subject_id, time) of ED discharge times."""
    import pickle
    obj = pickle.load(open(path, "rb"))
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            sid = int(k[0]) if isinstance(k, tuple) else int(k)
            for t in (v if isinstance(v, (list, tuple, set)) else [v]):
                rows.append((sid, t))
    else:                                            # list of (sid, t) or a pandas DataFrame
        try:
            obj = list(obj.itertuples(index=False))
        except AttributeError:
            obj = list(obj)
        for r in obj:
            rows.append((int(r[0]), r[1]))
    return (pl.DataFrame(rows, schema=["subject_id", "time"], orient="row")
            .with_columns(pl.col("subject_id").cast(pl.Int64), pl.col("time").cast(pl.Datetime("us")))
            .drop_nulls())


def _collapse_episodes(times, days: int = CASE_DEDUP_DAYS) -> list:
    """reAIM cohort_extract.py's episode collapse: a stateful scan keeping a case iff it is >= `days`
    after the LAST KEPT case (NOT the previous raw one -- they differ on <7-day chains)."""
    kept, last = [], None
    for t in times:
        if last is None or (t - last).total_seconds() >= days * 86400:
            kept.append(t)
            last = t
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meds", required=True, help="MIMIC MEDS data dir (shards of *.parquet)")
    ap.add_argument("--codes", default="icd_codes.json", help="icd_codes.json (the code sets)")
    ap.add_argument("--task", default="ami", help="prefix in icd_codes.json (ami / stroke / masld)")
    ap.add_argument("--output", required=True, help="output cohort parquet")
    ap.add_argument("--ed-out-dict", default=None,
                    help="reference ed_out_dict.pkl; if given, ED prediction times come from it "
                         "(instead of the MEDS ED_OUT event) -- to match the reference cohort exactly")
    ap.add_argument("--case-emergency", action=argparse.BooleanOptionalAction, default=None,
                    help="override: case must follow an emergency visit. Default: per-task PARAMS "
                         "(True for AMI, False for MASLD).")
    ap.add_argument("--first-case", action=argparse.BooleanOptionalAction, default=None,
                    help="override: incident cohort (keep only discharges strictly before the first "
                         "case). Default: per-task PARAMS (False for AMI, True for MASLD).")
    ap.add_argument("--no-ed", action="store_true",
                    help="hospital-only: drop ED discharges entirely (matches the hospital-only ACES "
                         "task, whose ed_only_discharge never-matches). Not part of the delivered cohort.")
    args = ap.parse_args()

    # resolve cohort parameters: explicit flag > per-task PARAMS > global default
    p = PARAMS.get(args.task, {})
    case_emergency = args.case_emergency if args.case_emergency is not None else p.get("case_emergency", True)
    first_case = args.first_case if args.first_case is not None else p.get("first_case", False)
    print(f"[{args.task}] case_emergency={case_emergency} first_case={first_case} "
          f"ed={'pkl' if args.ed_out_dict else ('none' if args.no_ed else 'MEDS ED_OUT')}")

    ed_out = load_ed_out(args.ed_out_dict) if args.ed_out_dict else None
    codes = json.load(open(args.codes))
    at_risk = codes[f"{args.task}_at_risk"]
    case = codes[f"{args.task}_case"]
    crf = codes.get(f"{args.task}_clinical_risk_factor", [])   # MASLD has no CRF set -> single at-risk set, no OR arm

    fs = [f for f in glob.glob(os.path.join(args.meds, "**", "*.parquet"), recursive=True) if ".logs" not in f]
    if not fs:
        raise SystemExit(f"[meds] no parquet under {args.meds!r}")
    m = (pl.read_parquet(fs, columns=["subject_id", "time", "code"])
         .with_columns(pl.col("subject_id").cast(pl.Int64), pl.col("time").cast(pl.Datetime("us")))
         .filter(~pl.col("subject_id").is_in(EXCLUDE))
         .drop_nulls("time").sort(["subject_id", "time"]))

    # per-event flags -- PREFIX match (his event_check uses code.startswith), computed on the
    # distinct codes for speed. The sets carry parent ICD stubs (e.g. I25) that match specific MEDS
    # codes (I2510) by prefix; exact membership would miss them.
    pre = lambda cs: "^(" + "|".join(re.escape(c) for c in cs) + ")"
    dc = m.select("code").unique().with_columns(
        is_atrisk=pl.col("code").str.contains(pre(at_risk)).cast(pl.Int64),
        is_case=pl.col("code").str.contains(pre(case)).cast(pl.Int64),
        is_crf=(pl.col("code").str.contains(pre(crf)) if crf else pl.lit(False)).cast(pl.Int64),
        is_emerg=pl.col("code").str.contains(EMERGENCY).cast(pl.Int64),
    )
    m = m.join(dc, on="code", how="left")
    # per-subject running state
    m = m.with_columns(
        # observation_start = first NON-birth timed event (his loop skips MEDS_BIRTH); using the raw
        # min(time) would open the 2-yr window at the birth anchor, decades too early.
        obs_start=pl.when(pl.col("code") != "MEDS_BIRTH").then(pl.col("time")).min().over("subject_id"),
        emerg_cum=pl.col("is_emerg").cum_sum().over("subject_id"),
    )
    # AT-RISK entry: his check is gated on >=2yr of history, so CRF codes are only COUNTED after the
    # 730-day window (and at-risk codes only checked there). At-risk = first post-window event that is
    # an at-risk code OR the 2nd post-window clinical-risk-factor.
    post = (pl.col("time") - pl.col("obs_start")).dt.total_days() >= OBS_DAYS
    m = m.with_columns(crf_cum=(pl.col("is_crf") * post.cast(pl.Int64)).cum_sum().over("subject_id"))
    elig = post & ((pl.col("is_atrisk") == 1) | (pl.col("crf_cum") >= 2))
    atrisk_t = (m.filter(elig).group_by("subject_id").agg(pl.col("time").min().alias("at_risk_time")))

    # CASES: case code (AMI: after an emergency visit; MASLD: any), 7-day de-dup -> first case per subject
    case_flag = (pl.col("is_case") == 1)
    if case_emergency:
        case_flag = case_flag & (pl.col("emerg_cum") >= 1)
    cs = (m.filter(case_flag).select("subject_id", "time").sort(["subject_id", "time"])
          .group_by("subject_id", maintain_order=True).agg(pl.col("time"))
          .with_columns(pl.col("time").map_elements(_collapse_episodes, return_dtype=pl.List(pl.Datetime("us"))))
          .explode("time")
          .select("subject_id", pl.col("time").alias("case_time")))
    last_t = m.group_by("subject_id").agg(pl.col("time").max().alias("last_time"))

    # PREDICTION anchors (BINARY cohort): every discharge after at-risk, <=730d since the previous
    # event. Hospital discharges come from MEDS; ED discharges come from ed_out_dict if given, else
    # the MEDS ED_OUT event. (Binary keeps discharges past the case too -- unlike TTE.)
    hosp = m.filter(pl.col("code").str.contains(r"^HOSPITAL_DISCHARGE")).select("subject_id", "time")
    if args.no_ed:                       # hospital-only (matches the never-matching ed_only_discharge in ACES)
        disch = hosp.unique()
    else:
        if ed_out is not None:
            # his insert_ed_discharge turns a dict ED time into a prediction ONLY when it falls in a GAP
            # (strictly between two existing events). A dict time that coincides with an existing MEDS
            # event time is consumed without injecting -> never a prediction, so drop those.
            ed = ed_out.join(m.select("subject_id", "time").unique(), on=["subject_id", "time"], how="anti")
        else:
            ed = m.filter(pl.col("code").str.contains(r"^ED_OUT")).select("subject_id", "time")
        disch = pl.concat([hosp, ed.select("subject_id", "time")]).unique()
    # prev MEDS event strictly before each discharge (for the recent-activity gate)
    allev = m.select("subject_id", pl.col("time").alias("_p")).unique().sort(["subject_id", "_p"])
    disch = (disch.with_columns(_b=pl.col("time") - US).sort(["subject_id", "_b"])
             .join_asof(allev, left_on="_b", right_on="_p", by="subject_id", strategy="backward")
             .with_columns(prev_time=pl.col("_p")).drop(["_b", "_p"]))
    T = (disch.join(atrisk_t, on="subject_id", how="inner").join(last_t, on="subject_id", how="left"))
    recent = ((pl.col("time") - pl.col("prev_time")).dt.total_days() <= OBS_DAYS).fill_null(True)
    T = (T.filter((pl.col("time") > pl.col("at_risk_time")) & recent)
         .select("subject_id", pl.col("time").alias("t"), "last_time"))

    # LABEL: the next case STRICTLY after t, positive if within 365d. A billing AMI dx is often
    # recorded at the discharge time itself (case_time == t); +1us skips it so we pick the genuine
    # next AMI (his next-target is strict), instead of landing on the same-time case and nulling out.
    T = (T.with_columns(_t1=pl.col("t") + US).sort(["subject_id", "_t1"])
         .join_asof(cs.sort(["subject_id", "case_time"]), left_on="_t1", right_on="case_time",
                    by="subject_id", strategy="forward").drop("_t1"))
    has_case = pl.col("case_time").is_not_null()
    T = T.with_columns(boolean_value=(has_case & (pl.col("case_time") <= pl.col("t") + pl.duration(days=LOOKAHEAD))).cast(pl.Boolean))
    # censoring: drop a no-case discharge that is the subject's last event (tte_days <= 0)
    T = T.filter(has_case | (pl.col("t") < pl.col("last_time")))

    if first_case:        # incident cohort: keep only discharges strictly before the subject's FIRST case
        fc = cs.group_by("subject_id").agg(pl.col("case_time").min().alias("_fc"))
        T = (T.join(fc, on="subject_id", how="left")
             .filter(pl.col("_fc").is_null() | (pl.col("t") < pl.col("_fc"))).drop("_fc"))

    out = T.select("subject_id", pl.col("t").alias("prediction_time"), "boolean_value").unique()
    out.write_parquet(args.output)
    n = out["subject_id"].n_unique()
    print(f"MIMIC {args.task} cohort: {out.height:,} points | {n:,} subjects | "
          f"{out.filter(pl.col('boolean_value')).height:,} positives")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
