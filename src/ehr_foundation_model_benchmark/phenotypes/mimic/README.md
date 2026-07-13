# MIMIC phenotype cohorts

Reproduction of the reAIM-Lab MIMIC binary phenotype cohorts (AMI, MASLD, stroke) on a MIMIC-IV
MEDS build. Extracted from https://github.com/reAIM-Lab/mimic_phenotype_cohort.

## Files
- `icd_codes.json`       — the ICD code sets (`<task>_at_risk`, `<task>_case`, `ami_clinical_risk_factor`).
- `build_ed_out_dict.py` — rebuilds `ed_out_dict.pkl` (non-admitted-ED discharges = `transfers.outtime`
  where `hadm_id` is null) from raw MIMIC-IV.
- `mimic_cohort.py`      — the cohort generator (polars). Per-task parameters are baked into `PARAMS`.

## Requirements
`pip install polars` (a recent 1.x). MEDS build: MIMIC-IV `meds_v0.6` (`MEDS_cohort/data`).

## Usage
```bash
# 1) rebuild the ED-discharge pickle from raw MIMIC-IV transfers
python build_ed_out_dict.py --transfers /path/to/mimiciv/3.1/hosp/transfers.csv.gz --output ed_out_dict.pkl

# 2) build a binary cohort (parameters auto-applied per --task)
python mimic_cohort.py --task ami    --meds /path/to/MEDS_cohort/data --codes icd_codes.json --ed-out-dict ed_out_dict.pkl --output ami.parquet
python mimic_cohort.py --task masld  --meds /path/to/MEDS_cohort/data --codes icd_codes.json --ed-out-dict ed_out_dict.pkl --output masld.parquet
python mimic_cohort.py --task stroke --meds /path/to/MEDS_cohort/data --codes icd_codes.json --ed-out-dict ed_out_dict.pkl --output stroke.parquet
```

> **Note:** the `EMERGENCY` / `DISCHARGE` code patterns in `mimic_cohort.py` are MEDS-build-specific.
> Verify them against your build first (`meds_v0.6` typically uses
> `HOSPITAL_ADMISSION` / `ICU_ADMISSION` / `ED_REGISTRATION` and `HOSPITAL_DISCHARGE` / `ED_OUT`).

## Cohort definition (per subject, events time-sorted)
- **observation_start** = first non-`MEDS_BIRTH` event; at-risk needs ≥2 yr (730 d) of history first.
- **at-risk entry** = first event ≥730 d after obs_start that is a `<task>_at_risk` code, OR (AMI only)
  the 2nd `ami_clinical_risk_factor`.
- **case** = a `<task>_case` code, collapsed to one per 7-day episode (stateful last-kept scan);
  ACUTE (ami/stroke): only after an emergency visit (`HOSPITAL_ADMISSION|ICU_ADMISSION|ED_REGISTRATION`);
  CHRONIC (masld): the raw code.
- **prediction time** = every discharge (`HOSPITAL_DISCHARGE` + ED from the pickle) after at-risk,
  ≤730 d since the previous event; MASLD keeps only discharges strictly before the first case (incident).
- **label** = case within 365 d, strictly after the prediction. Binary = all qualifying discharges.
- Code matching is prefix (`startswith`) — except MASLD's case, which is matched exactly
  (identical here, the 6 codes are leaves). 4 subjects excluded (`EXCLUDE`).

## Per-task parameters (encoded in `mimic_cohort.py:PARAMS`)
| task   | path    | case needs emergency | first-case / incident |
|--------|---------|----------------------|-----------------------|
| ami    | acute   | yes                  | no (all discharges)   |
| stroke | acute   | yes                  | no (all discharges)   |
| masld  | chronic | no                   | yes                   |
