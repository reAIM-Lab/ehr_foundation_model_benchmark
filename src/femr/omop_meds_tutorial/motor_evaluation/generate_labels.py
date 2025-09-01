"""
One of FEMR's main features is utilities for helping write labeling functions.

The following are two simple labelers for inpatient mortality and long admission for MIMIC-IV.
"""

import femr.labelers
import meds_reader
import meds
import datetime
import shutil

from typing import List, Mapping
from pathlib import Path


LABEL_NAMES = [
    "death",
    "long_los",
]
# LABEL_NAMES = ['long_los', '30d']
ADMISSION_EVENTS = ["Visit/IP", "Visit/ERIP", "CMS Place of Service/51", "CMS Place of Service/61"]


class OmopInpatientMortalityLabeler(femr.labelers.Labeler):
    def __init__(self, time_after_admission: datetime.timedelta):
        self.time_after_admission = time_after_admission

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        admission_ranges = set()
        death_times = set()

        for event in subject.events:
            if event.code in ADMISSION_EVENTS and event.end is not None:
                if isinstance(event.end, datetime.datetime):
                    admission_ranges.add((event.time, event.end))
                else:
                    admission_ranges.add((event.time, datetime.datetime.fromisoformat(event.end)))
            if event.code == meds.death_code:
                death_times.add(event.time)

        if len(death_times) not in [0, 1]:
            print(f"Warning: found {len(death_times)} death events in subject: {subject.subject_id}")

        if len(death_times) == 1:
            death_time = list(death_times)[0]
        else:
            death_time = datetime.datetime(9999, 1, 1)  # Very far in the future

        labels = []

        for (admission_start, admission_end) in admission_ranges:
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue

            if prediction_time >= death_time:
                continue

            is_death = death_time < admission_end
            labels.append(
                meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time, boolean_value=is_death))

        return labels


class OmopLongAdmissionLabeler(femr.labelers.Labeler):
    def __init__(self, time_after_admission: datetime.timedelta, admission_length: datetime.timedelta):
        self.time_after_admission = time_after_admission
        self.admission_length = admission_length

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        admission_ranges = set()

        for event in subject.events:
            if event.code in ADMISSION_EVENTS and event.end is not None:
                if isinstance(event.end, datetime.datetime):
                    admission_ranges.add((event.time, event.end))
                else:
                    admission_ranges.add((event.time, datetime.datetime.fromisoformat(event.end)))

        labels = []
        for (admission_start, admission_end) in admission_ranges:
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue

            is_long_admission = (admission_end - admission_start) > self.admission_length

            labels.append(meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time,
                                     boolean_value=is_long_admission))

        return labels


labelers: Mapping[str, femr.labelers.Labeler] = {
    'death': OmopInpatientMortalityLabeler(time_after_admission=datetime.timedelta(hours=48)),
    'long_los': OmopLongAdmissionLabeler(time_after_admission=datetime.timedelta(hours=48),
                                         admission_length=datetime.timedelta(days=7)),
}


def create_omop_meds_tutorial_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for preparing Motor")
    parser.add_argument(
        "--pretraining_data",
        dest="pretraining_data",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--meds_reader",
        dest="meds_reader",
        action="store",
        required=True,
    )
    return parser


def main():
    args = create_omop_meds_tutorial_arg_parser().parse_args()
    
    labels_path = Path(args.pretraining_data) / "labels"
    labels_path.mkdir(exist_ok=False)

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        for label_name in LABEL_NAMES:
            labeler = labelers[label_name]
            labels = labeler.apply(database)
            labels.to_parquet(str(labels_path / (label_name + '.parquet')))


if __name__ == "__main__":
    main()
