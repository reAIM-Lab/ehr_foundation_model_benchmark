from __future__ import annotations

import abc
import collections
import datetime
import functools
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
import sys
import meds
import meds_reader
import numpy as np
import scipy.sparse
import torch
import warnings

import femr.models.config
import femr.models.tokenizer
import femr.ontology
import femr.pat_utils
import femr.stat_utils
import random
import logging

class Task(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_task_config(self) -> femr.models.config.FEMRTaskConfig: ...

    @abc.abstractmethod
    def start_batch(self) -> None: ...

    @abc.abstractmethod
    def start_subject(self, subject: meds_reader.Subject, ontology: Optional[femr.ontology.Ontology]) -> None: ...

    @abc.abstractmethod
    def add_subject_labels(self, subject_label_offsets: List[int]) -> None: ...

    @abc.abstractmethod
    def needs_exact(self) -> bool: ...

    @abc.abstractmethod
    def get_sampled_labels(self, length: int) -> int:
        return length

    @abc.abstractmethod
    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]],
    ) -> int: ...

    @abc.abstractmethod
    def get_batch_data(self) -> Mapping[str, np.ndarray]: ...

    def cleanup(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return batch


class LabeledSubjectTask(Task):

    def __init__(self, labels: Sequence[meds.Label], observation_window: Optional[int] = None):
        super().__init__()
        self.label_map: Mapping[int, Any] = collections.defaultdict(list)
        for label in labels:
            self.label_map[label["subject_id"]].append(label)

        for k, v in self.label_map.items():
            v.sort(key=lambda a: a["prediction_time"])

        if observation_window is not None:
            assert observation_window > 0, "the feature extract observation window must be greater than 0 or None"
        self.observation_window = observation_window

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(task_type="labeled_subjects")

    def start_subject(self, subject: meds_reader.Subject, _ontology: Optional[femr.ontology.Ontology]) -> None:
        self.current_labels = self.label_map[subject.subject_id]
        self.current_label_index = 0

    def needs_exact(self) -> bool:
        return True

    def start_batch(self) -> None:
        """LabeledSubjectTask currently has no per label state."""
        pass

    def add_subject_labels(self, _subject_label_offsets: List[int]) -> None:
        """As there is no per label state, this is ignored"""
        pass

    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]] = None,
            actually_add: Optional[bool] = True,
    ) -> int:
        has_label = False

        while True:
            if self.current_label_index == len(self.current_labels):
                break

            current_label = self.current_labels[self.current_label_index]

            if self.observation_window is not None:
                observation_start_time = (
                        current_label["prediction_time"] - datetime.timedelta(days=self.observation_window)
                )
                is_valid = observation_start_time <= current_date <= current_label["prediction_time"]
                next_valid = (
                        next_date is not None and
                        observation_start_time <=next_date <= current_label["prediction_time"]
                )
            else:
                is_valid = current_date <= current_label["prediction_time"]
                next_valid = next_date is not None and next_date <= current_label["prediction_time"]

            if next_valid:
                # Next one is valid, so break early to give it a chance next time
                break

            if is_valid:
                has_label = True
                self.current_label_index += 1
            else:
                # The next label isn't valid, so we have to break here
                break

        if has_label:
            return 1
        else:
            return 0

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {}

    def get_sampled_labels(self, length: int) -> int:
        return length


class CLMBRTask(Task):
    def __init__(self, clmbr_vocab_size: int):
        self.clmbr_vocab_size = clmbr_vocab_size

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(
            task_type="clmbr", task_kwargs=dict(clmbr_vocab_size=self.clmbr_vocab_size)
        )

    def start_subject(self, _subject: meds_reader.Subject, _ontology: Optional[femr.ontology.Ontology]) -> None:
        self.per_subject_batch_labels: List[int] = []

    def needs_exact(self) -> bool:
        return False

    def start_batch(self) -> None:
        self.batch_labels: List[int] = []

    def add_subject_labels(self, subject_label_offsets: List[int]) -> None:
        self.batch_labels.extend([self.per_subject_batch_labels[i] for i in subject_label_offsets])

    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]] = None,
    ) -> int:
        if next_features is None:
            return 0

        if len(next_features) != 1:
            raise RuntimeError("Only supports one for right now")

        next_feature = next_features[0]

        if next_feature >= self.clmbr_vocab_size:
            return 0

        self.per_subject_batch_labels.append(next_feature)

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {"labels": np.array(self.batch_labels, dtype=np.int32)}


class SurvivalCalculator:
    def __init__(
            self, ontology: femr.ontology.Ontology, subject: meds_reader.Subject,
            code_whitelist: Optional[Set[str]] = None
    ):
        self.survival_events = []  # a list of tuples, each tuple is a code and a time (lab/10, 2025-01-01)
        self.final_date = subject.events[-1].time
        self.future_times = collections.defaultdict(list)   # a dictionary of lists, key is the code, value is a list of happened times

        for event in subject.events:
            if event.time is None:
                continue
            # if event.code.split('/')[0] in ('LAB', 'MEDICATION', 'INFUSION_START', 'INFUSION_END'):
            #     continue
            if event.numeric_value is not None or event.text_value is not None:
                continue
            codes = set()
            for parent in ontology.get_all_parents(event.code):
                if code_whitelist is None or parent in code_whitelist:
                    codes.add(parent)

            for code in codes:
                self.future_times[code].append(event.time)
                self.survival_events.append((code, event.time))

        for v in self.future_times.values():
            v.reverse()

        self.survival_events.reverse()

    # Advancing the cursor: removes any past or current events so that only future ones remain in future_times.
    def get_future_events_for_time(
            self, time: datetime.datetime
    ) -> Tuple[datetime.timedelta, Mapping[str, datetime.timedelta]]:
        while len(self.survival_events) > 0 and self.survival_events[-1][1] <= time:
            code = self.survival_events[-1][0]
            vals = self.future_times[code]
            vals.pop()
            if len(vals) == 0:
                del self.future_times[code]

            self.survival_events.pop()

        delta = self.final_date - time
        # k is the code, v is the list of times
        # v[-1] is the last time in the list
        # time is the current time
        # so v[-1] - time is the time until the next event of the code
        return (delta, {k: v[-1] - time for k, v in self.future_times.items()})


def _prefit_motor_map(
        subjects: Iterator[meds_reader.Subject], *, tasks: List[str], ontology: femr.ontology.Ontology
) -> Any:
    task_time_stats: List[Any] = [[0, 0, femr.stat_utils.OnlineStatistics()] for _ in range(len(tasks))]
    # Create separate ReservoirSamplers for each task/event type
    task_event_times = [femr.stat_utils.ReservoirSampler(100_000) for _ in range(len(tasks))]
    task_set = set(tasks)
    # print(f"task_set length: {len(task_set)}")

    # print(f"subject has {len(list(subjects))} objects.")
    for subject in subjects:
        calculator = SurvivalCalculator(ontology, subject, task_set)
        # print(f"calculator: {calculator}")
        # print(for )
        birth = femr.pat_utils.get_subject_birthdate(subject)

        for event, next_event in zip(subject.events, subject.events[1:]):
            # 1) Skip any “birth”‐day events or events with missing times
            if (event.time is None) or (event.time.date() == birth.date()) or (event.time.date() == next_event.time.date()):
                continue
            
              # 2) Ask the calculator: 
                #    - `censor_time`: time until end‐of‐record
                #    - `tte`: dict of per‐code next‐event deltas
            censor_time, tte = calculator.get_future_events_for_time(event.time)

            if len(tte) == 0:
                continue
            
            for i, task in enumerate(tasks):
                if task in tte:
                    time = tte[task]
                    is_censored = False
                else:
                    time = censor_time
                    is_censored = True

                if is_censored:
                    task_time_stats[i][0] += 1
                else:
                    # Add to the specific task's event time sampler
                    task_event_times[i].add(time.total_seconds(), 1)
                    task_time_stats[i][1] += 1
                task_time_stats[i][2].add(1, time.total_seconds())
    # print(f"after subject {subject.subject_id}, task_time_stats: {task_time_stats}")
    print(f"prefit_motor_map is done")
    return (task_event_times, task_time_stats)


def _prefit_motor_agg(first: Any, second: Any) -> Any:
    # first/second is (task_event_times, task_time_stats)
    # for task_event_times, we combine each task's ReservoirSampler separately
    # for task_time_stats, we add the number of events/censoring times, and combine the two groups of elements
    for a, b in zip(first[1], second[1]):
        a[0] += b[0]
        a[1] += b[1]
        a[2].combine(b[2])
    # Combine each task's event times separately
    for i in range(len(first[0])):
        first[0][i].combine(second[0][i])
    return first


class MOTORTask(Task):
    @classmethod
    def fit_pretraining_task_info(
            cls,
            db: meds_reader.SubjectDatabase,
            tokenizer: femr.models.tokenizer.HierarchicalTokenizer,
            num_tasks: int,
            num_bins: int,
            final_layer_size: int,
            codes_to_skip: List[str] = None,
    ) -> MOTORTask:
        tasks = []
        # the vocab is ordered by entropy, the first codes has highest entropy
        for dict_entry in tokenizer.dictionary["vocab"]:
            if dict_entry["type"] == "code":
                # Skip the codes that are in the codes_to_skip
                if codes_to_skip and dict_entry["code_string"] in codes_to_skip:
                    continue
                tasks.append(dict_entry["code_string"])
                if len(tasks) == num_tasks:
                    break

        if len(tasks) < num_tasks:
            warnings.warn(f"Could not find enough tasks in the provided tokenizer {len(tasks)}")

        # print(f"tasks: {tasks}")
        # apply _prefit_motor_map(subjects, tasks=tasks, ontology=tokenizer.ontology) and then use _prefit_motor_agg to aggregate the results
        print("before functools.reduce")
        task_length_samples, stats = functools.reduce(
            _prefit_motor_agg, db.map(functools.partial(_prefit_motor_map, tasks=tasks, ontology=tokenizer.ontology))
        )

        # print(f"task_length_samples samples: {[len(sampler.samples) for sampler in task_length_samples]}")
        # print(f"stats: {stats}")

        # Create time bins for each task separately, but all with the same number of bins
        # time_bins becomes a 2D array: [num_tasks, num_bins+1]
        # This allows each event type to have its own time discretization based on its specific distribution
        # Each task gets bins optimized for its time-to-event distribution
        # for i, length_samples in enumerate(task_length_samples):
        #     if len(length_samples.samples) > 0:
        #         task_time_bins = np.percentile(length_samples.samples, np.linspace(0, 100, num_bins + 1))
        #         task_time_bins[0] = 0
        #         task_time_bins[-1] = float("inf")
        #         time_bins.append(list(task_time_bins))
        #     else:
        #         # If no events for this task, use default uniform bins
        #         print(f"Warning: No events found for task {i} ({tasks[i]}), using default bins")
        #         default_bins = [0] + [float(i+1) * 86400 for i in range(num_bins-1)] + [float("inf")]  # daily bins
        #         time_bins.append(default_bins)
        
        # Convert to numpy array for easier handling
        # time_bins = np.array(time_bins)  # Shape: [num_tasks, num_bins+1]
        time_bins = []
        task_data = []

        for i,(task, task_stats) in enumerate(zip(tasks, stats)):
            frac_events = task_stats[1] / (task_stats[0] + task_stats[1])
            rate = frac_events / task_stats[2].mean()  # happening rate of the task num_points/time

            if rate == 0:
                print("Ran into task of rate 0?", task, frac_events, task_stats[0], task_stats[1], task_stats[2].mean())
                continue

            if frac_events < 1 / 1000:
                print("Ran into very rare task with less than 10 occurrences", task, frac_events, task_stats[0],
                      task_stats[1], task_stats[2].mean())
                continue

            task_data.append((task, rate, task_stats[0], task_stats[1], task_stats[2].mean()))
            task_time_bins = np.percentile(task_length_samples[i].samples, np.linspace(0, 100, num_bins + 1))
            task_time_bins[0] = 0
            task_time_bins[-1] = float("inf")
            time_bins.append(list(task_time_bins))
        time_bins = np.array(time_bins) 

        return MOTORTask(task_data, time_bins, final_layer_size)

    def __init__(self, pretraining_task_info: List[Tuple[str, float]], time_bins: np.ndarray, final_layer_size: int):
        self.pretraining_task_info = pretraining_task_info
        # Handle both numpy array and list (from config deserialization)
        self.time_bins = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins  # Now a 2D array: [num_tasks, num_bins+1]
        self.final_layer_size = final_layer_size

        self.pretraining_task_codes = set()
        self.task_to_index_map = {}
        # the self.pretraining_task_info is a list of tuples, each tuple is a task name, rate, num_events, num_censored, mean_time
        for i, task in enumerate(self.pretraining_task_info):
            self.pretraining_task_codes.add(task[0])
            self.task_to_index_map[task[0]] = i  # task_to_index_map is a dictionary, key is the task name, value is the index

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(
            task_type="motor",
            task_kwargs=dict(
                pretraining_task_info=self.pretraining_task_info,
                time_bins=self.time_bins.tolist() if isinstance(self.time_bins, np.ndarray) else self.time_bins,
                final_layer_size=self.final_layer_size,
            ),
        )

    def start_subject(self, subject: meds_reader.Subject, ontology: Optional[femr.ontology.Ontology]) -> None:
        assert ontology
        self.calculator = SurvivalCalculator(ontology, subject, self.pretraining_task_codes)

        self.per_subject_censor_time: List[float] = []
        self.per_subject_time_sparse: Dict[str, List[float]] = {
            "data": [],
            "indices": [],
            "indptr": [0],
        }

    def needs_exact(self) -> bool:
        return False

    def get_sampled_labels(self, length: int) -> int:
        desired_labels = max(5, length // 10)
        return desired_labels

    def start_batch(self) -> None:
        self.censor_time: List[float] = []

        self.time_sparse: Dict[str, List[float]] = {
            "data": [],
            "indices": [],
            "indptr": [0],
        }

    def add_subject_labels(self, subject_label_offsets: List[int]) -> None:
        """Add per-subject labels to the global task labels."""
        self.censor_time.extend([self.per_subject_censor_time[i] for i in subject_label_offsets])

        for index in subject_label_offsets:
            start = int(self.per_subject_time_sparse["indptr"][index])
            end = int(self.per_subject_time_sparse["indptr"][index + 1])

            self.time_sparse["data"].extend(self.per_subject_time_sparse["data"][start:end])
            self.time_sparse["indices"].extend(self.per_subject_time_sparse["indices"][start:end])
            self.time_sparse["indptr"].append(len(self.time_sparse["indices"]))

    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]] = None,
            actually_add: bool = True,
    ) -> int:
        if next_date is None or next_date == current_date:
            return 0
        
        if not actually_add:
            return 1

        censor_time, tte = self.calculator.get_future_events_for_time(current_date)

        if len(tte) == 0:
            return 0


        censor_seconds = censor_time.total_seconds()
        self.per_subject_censor_time.append(censor_seconds)

        for event_name, time in tte.items():
            j = self.task_to_index_map[event_name]
            seconds = time.total_seconds()

            self.per_subject_time_sparse["data"].append(seconds)
            self.per_subject_time_sparse["indices"].append(j)

        self.per_subject_time_sparse["indptr"].append(len(self.per_subject_time_sparse["data"]))
        # print(f"add event {len(self.per_subject_time_sparse['data'])}")
        # print(f"add event {len(self.per_subject_time_sparse['indptr'])}")

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        def h(a, dtype):
            return {
                "data": np.array(a["data"], dtype=dtype),
                "indices": np.array(a["indices"], dtype=np.int32),
                "indptr": np.array(a["indptr"], dtype=np.int32),
            }

        # print(f"this batch return censor_time shape: {np.array(self.censor_time, dtype=np.float32).shape}")
        # print(f"this batch return data shape: {h(self.time_sparse, dtype=np.float32)['data'].shape}")
        # print(f"this batch return indices shape: {h(self.time_sparse, dtype=np.float32)['indices'].shape}")
        # print(f"this batch return indptr shape: {h(self.time_sparse, dtype=np.float32)['indptr'].shape}")

        return {
            "censor_time": np.array(self.censor_time, dtype=np.float32),
            "time_sparse": h(self.time_sparse, dtype=np.float32),
        }

    def cleanup(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """
        Convert sparse time-to-event data into dense tensors for loss calculation.
        
        Fully vectorized implementation for efficiency at scale:
        - Handles hundreds of prediction points × thousands of tasks × tens of bins efficiently
        - Uses batch tensor operations instead of nested loops
        
        Design principle:
        - For each prediction point and task, exactly ONE time bin should have is_event=True
        - This represents either: (1) an event occurring in that interval, or (2) censoring in that interval
        - is_censored flag distinguishes between event (False) and censoring (True) cases
        
        Returns:
        - is_event: [prediction_points, time_bins, tasks] - exactly one True per prediction-task combination
        - is_censored: [prediction_points, tasks] - True if the marked bin represents censoring
        """
        num_time_bins = self.time_bins.shape[1] - 1  # Each task has same number of bins
        num_indices = len(batch["censor_time"])
        num_tasks = len(self.pretraining_task_info)

        def h(a):
            # Get actual number of tasks from the sparse data
            # if len(a["indices"]) > 0:
            #     max_task_index = torch.max(a["indices"]).item()
            #     actual_num_tasks = max_task_index + 1
            # else:
            #     # No events in this batch - use expected number of tasks
            #     actual_num_tasks = len(self.pretraining_task_info)
            
            shape = (num_indices, num_tasks)
            # print(f"batch[a] is {batch[a]}")
            a = {k: v.numpy() for k, v in batch[a].items()}
            # print(f"a['data'] is {len(a['data'])},first 10 elements: {a['data'][:10]}, last 10 elements: {a['data'][-10:]}")
            # print(f"a['indices'] is {len(a['indices'])},first 10 elements: {a['indices'][:10]}, last 10 elements: {a['indices'][-10:]}")
            # print(f"a['indptr'] is {len(a['indptr'])},first 10 elements: {a['indptr'][:10]}, last 10 elements: {a['indptr'][-10:]}")
            s = scipy.sparse.csr_array((a["data"], a["indices"], a["indptr"]), shape=shape)
            time_return = torch.from_numpy(s.toarray())
            # print(f"s is {s.shape}, {s}")
            # print(f"time_return is {time_return.shape}, {time_return}")
            return time_return

        # time shape: [pred_points, actual_task_points]
        # time[pred_idx, task_idx] = 0 means no future event for this task
        # time[pred_idx, task_idx] > 0 means time until next event for this task
        time = h("time_sparse")
        # logging.info(f"time shape: {time.shape}")
        # logging.info(f"time is {time}")
        
        # Get actual dimensions from the data
        num_tasks = time.shape[1]  # Use actual number of tasks from data
        expected_num_tasks = len(self.pretraining_task_info)
        
        
        # Convert to torch tensor for efficient operations
        time = torch.from_numpy(time) if isinstance(time, np.ndarray) else time
        censor_times = batch["censor_time"]  # [pred_points]
        # logging.info(f"censor_times.shape: {censor_times.shape}")
        
        # Initialize output tensors
        is_event = torch.zeros(size=(num_indices, num_time_bins, num_tasks), dtype=torch.bool, device=time.device)
        is_censored = torch.zeros(size=(num_indices, num_tasks), dtype=torch.bool, device=time.device)
        
        # has_future_event shape: [pred_points, task_points]
        has_future_event = time != 0
        # print(f"has_future_event.shape: {has_future_event.shape}, sum: {has_future_event.sum()}, ratio: {has_future_event.sum()/has_future_event.numel()}")
        
        # Convert time_bins to torch tensor for vectorized operations
        time_bins_tensor = torch.from_numpy(self.time_bins).to(device=time.device, dtype=time.dtype)  # [num_tasks, num_bins+1]
        # print(f"time_bins_tensor: {time_bins_tensor}, time_bins_tensor.shape: {time_bins_tensor.shape}")
        
        # Vectorized approach: process all prediction points and tasks simultaneously
        
        # Expand dimensions for broadcasting
        # time: [pred_points, task_points] -> [pred_points, 1, task_points] 
        time_expanded = time.unsqueeze(1)  # [pred_points, 1, task_points]
        
        # censor_times: [pred_points] -> [pred_points, 1, 1]
        censor_times_expanded = censor_times.unsqueeze(1).unsqueeze(2)  # [pred_points, 1, 1]
        
        # time_bins_tensor: [task_points, num_bins+1] -> [1, num_bins, task_points, 2]
        # Extract start and end of each bin
        bin_starts = time_bins_tensor[:, :-1].T.unsqueeze(0)  # [1, num_bins, task_points]
        bin_ends = time_bins_tensor[:, 1:].T.unsqueeze(0)     # [1, num_bins, task_points]
        bin_widths = bin_ends - bin_starts
        
        # bin_starts [1, 20, 8192]
        # print(f"bin_starts: {bin_starts}, bin_starts.shape: {bin_starts.shape}")
        # For events: check which bin each event time falls into
        # Shape: [pred_points, num_bins, task_points]
        event_in_bin = (has_future_event.unsqueeze(1) & 
                       (bin_starts <= time_expanded) & 
                       (time_expanded < bin_ends))
        
        # For censoring: check which bin each censor time falls into
        # Shape: [pred_points, num_bins, task_points]  
        censor_in_bin = ((~has_future_event).unsqueeze(1) & 
                        (bin_starts <= censor_times_expanded) & 
                        (censor_times_expanded < bin_ends))
        
        censor_time_ratio = ((censor_times_expanded - bin_starts) / bin_widths) * censor_in_bin
      
        # logging.info(f"event_in_bin.shape: {event_in_bin.shape}")
        # logging.info(f"distribution of event_in_bin: {event_in_bin.sum(dim=0).sum(dim=1)/event_in_bin.sum()}")
        # logging.info(f"censor_in_bin.shape: {censor_in_bin.shape}")
        # logging.info(f"distribution of censor_in_bin: {censor_in_bin.sum(dim=0).sum(dim=1)/censor_in_bin.sum()}")
        
        # Combine event and censoring cases
        # Shape: [pred_points, num_bins, task_points]
        is_event = event_in_bin | censor_in_bin
        # logging.info(f"is_event {is_event}, is_event.shape: {is_event.shape}")
        # logging.info(f"distribution of is_event: {is_event.sum(dim=0).sum(dim=1)/is_event.sum()}")
        
        # Set is_censored flag: True where we used censoring
        # Shape: [pred_points, task_points]
        is_censored = ~has_future_event
        
        # Validation: ensure exactly one bin per prediction-task combination
        bins_per_pred_task = torch.sum(is_event, dim=1)  # [pred_points, task_points]


        if not torch.all(bins_per_pred_task == 1):
            # Handle edge cases where time falls exactly on bin boundary or outside all bins
            logging.info(f"Warning: {torch.sum(bins_per_pred_task != 1)}, bins_per_pred_task: {bins_per_pred_task}, prediction-task combinations don't have exactly 1 bin marked")
            logging.info(f"event_in_bin.shape: {event_in_bin.shape}")

            # Fix cases with 0 bins marked (time falls outside all bins - put in last bin)
            zero_bins_mask = bins_per_pred_task == 0  # [pred_points, task_points]
            if torch.any(zero_bins_mask):
                # Set the last bin to True for these cases
                pred_indices, task_indices = torch.where(zero_bins_mask)
                is_event[pred_indices, -1, task_indices] = True
                logging.info(f"Fixed {len(pred_indices)} cases by assigning to last bin")
            
            # Fix cases with multiple bins marked (time falls on boundary - keep only first)
            multiple_bins_mask = bins_per_pred_task > 1  # [pred_points, task_points]
            if torch.any(multiple_bins_mask):
                pred_indices, task_indices = torch.where(multiple_bins_mask)
                for i in range(len(pred_indices)):
                    pred_idx, task_idx = pred_indices[i], task_indices[i]
                    # Find first True bin and set others to False
                    true_bins = torch.where(is_event[pred_idx, :, task_idx])[0]
                    is_event[pred_idx, true_bins[1:], task_idx] = False
                logging.info(f"Fixed {len(pred_indices)} cases by keeping only first marked bin")
        # sys.exit()

        return {"is_event": is_event, "is_censored": is_censored, "censor_time_ratio": censor_time_ratio}
    
    

