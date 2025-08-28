from __future__ import annotations

import collections
import datetime
import functools
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import datasets
import meds_reader
import numpy as np
import torch.utils.data

import femr.models.tokenizer
import femr.pat_utils
import logging
logger = logging.getLogger(__name__)

def map_preliminary_batch_stats(
    subjects: Iterable[meds_reader.Subject], *, processor: FEMRBatchProcessor, max_length: int
):
    """
    This function creates preliminary batch statistics, to be used for final batching.

    The overall problem this is trying to solve is that Subject records can be very long, so
    when we featurize subjects we actually featurize subsequences of subjects.

    AKA every subject in a batch is actually (subject_id, start_index, length), which
    specifies a subsequence of the subject.

    The trickiness becomes when a particular subject has multiple labels, some
    of which will require multiple subsequences.

    The goal of this function is to compute the list [(subject_id, start_index, length)] such
    that every label is covered by at least one batch. Note that some labels will be covered multiple times.

    Note that there is a special setting for tasks that don't need exact labeling
    (needs_exact in tasks.py returns False).

    For these subjects we only generate one tuple for them, and drop some labels.

    Later code will then take [(subject_id, start_index, length)], and create actual batches.
    """
    lengths = []

    for subject in subjects:
        data = processor.convert_subject(subject, actually_add=False)

        # There are no labels for this subject
        if data["transformer"]["label_indices"].shape[0] == 0:
            continue

        # We need exact batching, so we need an algorithm that precisely covers every label in the batch
        if data["needs_exact"]:
            current_start = 0
            current_end = None
            for label_index in data["transformer"]["label_indices"]:
                if (label_index - current_start + 1) >= max_length:
                    if current_end is not None:
                        lengths.append((subject.subject_id, current_start, current_end - current_start + 1, 1e6))
                    current_start = label_index - max_length + 1
                    current_end = label_index
                else:
                    current_end = label_index

            lengths.append((subject.subject_id, current_start, current_end - current_start + 1, 1e6))
        else:
            last_index = data["transformer"]["label_indices"][-1]
            length = min(max_length, last_index + 1)

            start_index = last_index + 1 - length

            num_tasks = len([l for l in data["transformer"]["label_indices"] if l >= start_index])
            # Maximum number of labels to sample (limited by task requirements)
            desired_tasks = min(num_tasks, processor.creator.task.get_sampled_labels(length))
            # Fraction of labels to sample
            desired_task_fraction = desired_tasks / num_tasks

            # print(num_tasks, length, desired_tasks, desired_task_fraction)

            lengths.append((subject.subject_id, start_index, length, desired_task_fraction * 1e6))
    if len(lengths) > 0:
        return np.array(lengths, dtype=np.int64)
    else:
        return np.zeros(shape=(0, 4), dtype=np.int64)


class BatchCreator:
    """The BatchCreator is designed to generate batches from subject data."""

    def __init__(self, tokenizer: femr.models.tokenizer.FEMRTokenizer, task: Optional[femr.models.tasks.Task] = None):
        """Initialize a BatchCreator, with a tokenizer, and optionally a task."""
        self.tokenizer = tokenizer
        self.task = task

    def start_batch(self):
        """Start a batch."""
        self.subject_ids = []
        self.offsets = []
        self.subject_lengths = []

        if False:
            self.tokens = []
        elif isinstance(self.tokenizer, femr.models.tokenizer.HierarchicalTokenizer):
            self.hierarchical_tokens = []
            self.hierarchical_weights = []
            self.token_indices = [0]

        self.valid_tokens = []

        self.ages = []
        self.time_data = []
        self.timestamps = []

        self.label_indices = []

        if self.task is not None:
            self.task.start_batch()

    def add_subject(self, subject: meds_reader.Subject, offset: int = 0, max_length: Optional[int] = None, subsample_task_fraction: float = 1, actually_add: bool = True):
        """Add a subject to the current batch.

        Note that the two optional parameters are used to add a subset of a subject to a batch.

        It is generally recommended to never manually use offset or max_length as
        you should rely on FEMRBatchProcessor.convert_dataset.

        Arguments:
            subject: The subject to add.
            offset: The offset into the subject to featurize.
            max_length: The maximum length of the batch sequence. There is no max when left at None.

        """
        # print(f"add_subject starts --------------------")
        current_date = None
        last_time = None

        # The overall algorithm here is a bit complex
        # First we featurize the entire subject
        # Then we slice the subject indices according to offset and max_length

        # These are the indices of the labels into the subject vectors
        per_subject_label_indices = []

        # The ages at each index for the subject
        per_subject_ages = []

        # The normalized age at index for the subject
        per_subject_time_data = []

        # The timestamps at each index for the subject
        per_subject_timestamps = []

        # For a regular tokenizer, we just have tokens
        per_subject_tokens = []

        # For a hierarchical tokenizer, we have a more complex setup
        # These are designed to match the inputs required for an EmbeddingBag.
        # See PyTorch's EmbeddingBag documentation to understand what these mean.
        per_subject_hierarchical_tokens: List[int] = []
        per_subject_hierarchical_weights: List[float] = []
        per_subject_token_indices: List[int] = [0]

        if self.task is not None:
            self.task.start_subject(subject, self.tokenizer.ontology)

        birth = femr.pat_utils.get_subject_birthdate(subject)
        self.tokenizer.start_subject()

        for event in subject.events:
            if event.time is None or (
                    event.time.date() <= birth.date()
                    and getattr(event, "table", "person") == "person"
            ):
                # Get features and weights for the current event
                features, weights = self.tokenizer.get_feature_codes(event)
                per_subject_hierarchical_tokens.extend(features)
                per_subject_hierarchical_weights.extend(weights)

        # Purpose: capture all pre‑birth or “person” table codes as a single initial timestep.
        per_subject_token_indices.append(len(per_subject_hierarchical_tokens))
        per_subject_ages.append((event.time - birth) / datetime.timedelta(days=1))
        # per_subject_ages.append((birth - birth) / datetime.timedelta(days=1))
        per_subject_time_data.append([1, 0, 0, 0, 0])
        per_subject_timestamps.append(event.time.replace(tzinfo=datetime.timezone.utc).timestamp())
                
        
        for event in subject.events:
            if event.time is None or (
                    event.time.date() <= birth.date()
                    and getattr(event, "table", "person") == "person"
            ):
                continue

            # We want to avoid duplicate codes in the same day, so we maintain codes_seen_today
            if event.time.date() != current_date:
                current_date = event.time.date()
                codes_seen_today = set()

            # Get the age of the event, delta time to last time
            age = max(datetime.timedelta(seconds=0), event.time - birth)
            # print(f"age: {age}")
            if last_time is not None:
                delta = event.time - last_time
            else:
                delta = None

            # Get features and weights for the current event
            features, weights = self.tokenizer.get_feature_codes(event)
            # print(f"for event {event} features: {features}, weights: {weights}")

            # Ignore events with no features
            if len(features) == 0:
                continue

            # Ignore events where all features have already occurred
            if all(feature in codes_seen_today for feature in features):
                continue

            codes_seen_today |= set(features)

            # print(f"iterate event {self.task}, last_time is {last_time}, event.time is {event.time}, features is {features}, actually_add is {actually_add} --------------------")
            #  Label Generation  don't understand
            if (self.task is not None) and (last_time is not None):
                # Now we have to consider whether or not to have labels for this time step
                # The add_event function returns how many labels to assign for this time
                if subsample_task_fraction == 1 or random.random() < subsample_task_fraction:
                    #Ask the task how many labels to place at the previous timestep index. 
                    # num_added = 0/1, depneding on if there is still future tte event to predict
                    #Append that many copies of the current age‐index into per_subject_label_indices
                    num_added = self.task.add_event(last_time, event.time, features, actually_add=actually_add)
                    # print(f"num_added: {num_added}")
                    # print(f"event is {event}, time is {event.time}, num_added: {num_added}")
                    for _ in range(num_added):
                        # print(f"len(per_subject_ages) - 1 is : {len(per_subject_ages) - 1}, per_subject_ages: {per_subject_ages}")
                        per_subject_label_indices.append(len(per_subject_ages) - 1)

            if False:
                assert len(features) == 1
                per_subject_tokens.append(features[0])
            elif isinstance(self.tokenizer, femr.models.tokenizer.HierarchicalTokenizer):
                assert weights is not None
                per_subject_hierarchical_tokens.extend(features)
                per_subject_hierarchical_weights.extend(weights)
                per_subject_token_indices.append(len(per_subject_hierarchical_tokens)) 
            # Time Data updates, add age in days
            per_subject_ages.append((event.time - birth) / datetime.timedelta(days=1))
            # print(f"per_subject_label_indices: {per_subject_label_indices}")
            # logger.info(f"per_subject_label_indices: {per_subject_label_indices}")
            # print(f"per_subject_ages: {per_subject_ages}")
            # logger.info(f"per_subject_ages: {per_subject_ages}")

            # update time_data with normalized time features
            if last_time is None:
                per_subject_time_data.append([-1] + self.tokenizer.get_time_data(age, delta)[:2] + [0, 0])
                # print(f"last_time is None, per_subject_time_data added: {per_subject_time_data[-1]}")
            else:
                per_subject_time_data.append([0] + self.tokenizer.get_time_data(age, delta))
                # print(f"last_time is not None, per_subject_time_data added: {per_subject_time_data[-1]}")

            per_subject_timestamps.append(event.time.replace(tzinfo=datetime.timezone.utc).timestamp())

            last_time = event.time
        # exit()
        # add labels for the last time step
        if self.task is not None and last_time is not None and last_time.date() > birth.date():
            num_added = self.task.add_event(last_time, None, None)
            for _ in range(num_added):
                # print(f"len(per_subject_ages) - 1 is : {len(per_subject_ages) - 1}, per_subject_ages: {per_subject_ages}")
                per_subject_label_indices.append(len(per_subject_ages) - 1)

        # Now we want to actually add the subject data to the batch.
        # This will involve some clever slicing.

        # First, let's get the length we are adding
        length_found = len(per_subject_ages)
        if max_length is not None:
            length_to_add = min(length_found - offset, max_length)
        else:
            length_to_add = length_found - offset

        start_index = len(self.ages)

        # Let's add the constants first
        self.valid_tokens.extend([True] * length_to_add)
        self.subject_ids.extend([subject.subject_id] * length_to_add)
        self.offsets.append(offset)
        self.subject_lengths.append(length_to_add)

        # Ages, normalized ages and timestamps are also easy to add
        self.ages.extend(per_subject_ages[offset : offset + length_to_add])
        self.time_data.extend(per_subject_time_data[offset : offset + length_to_add])
        self.timestamps.extend(per_subject_timestamps[offset : offset + length_to_add])

        # Add back the birth event
        self.ages[start_index] = per_subject_ages[0]
        self.time_data[start_index] = per_subject_time_data[0]
        self.timestamps[start_index] = per_subject_timestamps[0]

        if False: #not self.tokenizer.is_hierarchical:
            # Easy for simple tokenizer
            self.tokens.extend(per_subject_tokens[offset : offset + length_to_add])
        elif isinstance(self.tokenizer, femr.models.tokenizer.HierarchicalTokenizer):
            # Hierarchical tokenizer is more complex since we have to shift the indices as well
            # Remember, these arrays are all designed for PyTorch EmbeddingBag

            # We need to get the start and end at a particular offset
            assert offset < len(per_subject_token_indices), f'Got it {len(per_subject_token_indices)} {subject.subject_id} {offset} {max_length}'

            if offset == 0:
                actual_offset = 0
                actual_length = length_to_add
            else:    
                actual_offset = offset + 1
                actual_length = length_to_add - 1

                birth_start = per_subject_token_indices[0]
                birth_end = per_subject_token_indices[1]

                # We need to offset the token indices to account for the existing tokens
                self.token_indices.append(len(self.hierarchical_tokens) + birth_end - birth_start)
                self.hierarchical_tokens.extend(per_subject_hierarchical_tokens[birth_start:birth_end])
                self.hierarchical_weights.extend(per_subject_hierarchical_weights[birth_start:birth_end])
            
            internal_start = per_subject_token_indices[actual_offset]
            internal_end = per_subject_token_indices[actual_offset + actual_length]

            # We need to offset the token indices to account for the existing tokens
            self.token_indices.extend(
                [
                    len(self.hierarchical_tokens) - internal_start + value
                    for value in per_subject_token_indices[actual_offset + 1 : actual_offset + actual_length + 1]
                ]
            )

            self.hierarchical_tokens.extend(per_subject_hierarchical_tokens[internal_start:internal_end])
            self.hierarchical_weights.extend(per_subject_hierarchical_weights[internal_start:internal_end])

        # The label indices are also a bit tricky as they have to be offset accordingly.
        # We also need to collect good labels that should be sent to the final numpy arrays.
        labels_to_add = []
        for i, label_index in enumerate(per_subject_label_indices):
            corrected_label = label_index - offset

            if 1 <= corrected_label < length_to_add:
                labels_to_add.append(i)
                self.label_indices.append(start_index + corrected_label)

        # not executed in map_preliminary_batch_stats, but executed in _batch_generator
        if actually_add and self.task is not None:
            self.task.add_subject_labels(labels_to_add)

    def get_batch_data(self):
        """Convert the batch to numpy arrays. The data structure is defined inline in this function."""
        # print(f"get_batch_data starts --------------------")
        if self.tokenizer.vocab_size <= 2**15:
            token_dtype = np.int16
        else:
            token_dtype = np.int32

        transformer = {
            # Whether or not the token is valid at this index
            "valid_tokens": np.array(self.valid_tokens),
            # The age of the subject in days at this index
            "ages": np.array(self.ages, dtype=np.float32),
            # The normalized ages at this index
            "time_data": np.array(self.time_data, dtype=np.float16),
            # The timestamp (in seconds) at this index
            "timestamps": np.array(self.timestamps, dtype=np.int64),
            # The length of the subject
            "subject_lengths": np.array(self.subject_lengths, dtype=np.int32),
            # The indices of the labels
            "label_indices": np.array(self.label_indices, dtype=np.int32),
        }

        if False: #not self.tokenizer.is_hierarchical:
            # For a single tokenizer, these are simple the token indices
            transformer["tokens"] = np.array(self.tokens, dtype=token_dtype)
        elif isinstance(self.tokenizer, femr.models.tokenizer.HierarchicalTokenizer):
            # See PyTorch's EmbeddingBag for what these numpy arrays mean.
            transformer["hierarchical_tokens"] = np.array(self.hierarchical_tokens, dtype=token_dtype)
            transformer["hierarchical_weights"] = np.array(self.hierarchical_weights, dtype=np.float16)
            transformer["token_indices"] = np.array(self.token_indices, dtype=np.int32)

        # Some general metadata
        final = {
            "num_subjects": len(self.subject_lengths),
            "num_indices": len(self.label_indices),
            "subject_ids": np.array(self.subject_ids, dtype=np.int64),
            "offsets": np.array(self.offsets, dtype=np.int32),
            "transformer": transformer,
        }

        # Add the task data
        if self.task is not None and transformer["label_indices"].shape[0] > 0:
            # for key in final:
            #     if key == "transformer":
            #         for key2 in final[key]:
            #             print(f"transformer key: {key2}")
            #             if isinstance(final[key][key2], np.ndarray):
            #                 print(f"value shape {final[key][key2].shape}, value {final[key][key2]}")
            #             else:
            #                 print(f"value {final[key][key2]}")
            #     else:
            #         print(f"key: {key}")
            #         if isinstance(final[key], np.ndarray):
            #             print(f"shape {final[key].shape},  value {final[key]}")
            #         else:
            #             print(f"value {final[key]}")
            # exit()
            final["task"] = self.task.get_batch_data()
            final["needs_exact"] = self.task.needs_exact()
            # print(f"final: {final['task']}")
            # logger.info(f"final: {final['task']}")
            # print(f"final: {final['needs_exact']}")
            # logger.info(f"final: {final['needs_exact']}")
            # print(f"final: {final}")
            # logger.info(f"final: {final}")
            # for key in final["task"]:
            #     print(f"key: {key}")
            #     print(f"value shape: {final['task'][key].shape}, value: {final['task'][key]}")
            # exit()
        return final

    def cleanup_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a batch, applying final processing.

        This is necessary as some tasks use sparse matrices that need to be postprocessed."""

        batch["transformer"]["subject_lengths"] = np.array(batch["transformer"]["subject_lengths"])
        assert isinstance(batch["transformer"]["subject_lengths"], np.ndarray)

        if self.task is not None and "task" in batch:
            batch["task"] = self.task.cleanup(batch["task"])

        return batch


def _batch_generator(batch_data: Tuple[np.ndarray, np.ndarray], *, creator: BatchCreator, path_to_database: str):
    with meds_reader.SubjectDatabase(path_to_database) as database:
        for lengths, offsets in batch_data:
            offsets = list(offsets)
            for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
                creator.start_batch()
                for subject_index, offset, length, subsample_task_fraction in lengths[start:end, :]:
                    creator.add_subject(database[subject_index.item()], offset, length, subsample_task_fraction=float(subsample_task_fraction)/1e6)

                result = creator.get_batch_data()
                # assert "task" in result, f"No task present in {lengths[start:end, :]} {i} {start} {end}"
                if "task" not in result:
                    # Log warning but continue processing
                    print(f"Warning: No valid labels for batch {i} (subjects {lengths[start:end, 0]})")
                    continue  # Skip this batch

                yield result


def _add_dimension(data: Any) -> Any:
    if isinstance(data, collections.abc.Mapping):
        return {k: _add_dimension(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.unsqueeze(dim=0)
    elif isinstance(data, np.ndarray):
        return np.expand_dims(data, axis=0)
    elif isinstance(data, (int, float, np.number, np.bool_)):
        return data
    else:
        raise RuntimeError("Could not convert item of type " + str(type(data)))


class FEMRBatchProcessor:
    """The FEMR Batch processor creates batches for processing by a transformer."""

    def __init__(self, tokenizer, task=None):
        self.creator = BatchCreator(tokenizer, task)

    def convert_subject(
        self,
        subject: meds_reader.Subject,
        offset: int = 0,
        max_length: Optional[int] = None,
        tensor_type=None,
        actually_add: Optional[bool] = True,
        **formatter_kwargs,
    ):
        """Convert a single subject to a batch.

        Note that this can also convert parts of a subject to a batch using the offset and max_length parameters.
        This is useful for processing long subjects.

        NOTE: This function is primarily for debugging purposes. It is
        recommended to use convert_dataset for maximum correctness and efficiency.

        Arguments:
            subject: The subject to convert
            offset: The integer offset into the subject to convert
            max_length: The maximum length to convert
            tensor_type: The dataset to return
            formatter_kwargs: Arguments for a datasets formatter when converting datatypes

        Returns:
            A batch, ready to be fed into a FEMR transformer model
        """
        self.creator.start_batch()
        # print(f"convert_subject starts --------------------")
        self.creator.add_subject(subject, offset=offset, max_length=max_length, actually_add=actually_add)
        batch_data = self.creator.get_batch_data()
        # by default, tensor_type is None
        if tensor_type is not None:
            formatter = datasets.formatting.get_formatter(tensor_type, **formatter_kwargs)
            batch_data = formatter.recursive_tensorize(batch_data)
        return batch_data

    def collate(self, batches: List[Mapping[str, Any]]) -> Mapping[str, Any]:
        """A collate function that prepares batches for being fed into a dataloader."""
        # if len(batches) > 1:
        # print(f"collate batches: {len(batches)}")
        # print(f"the batch 0 is {batches[0]}")
        # print(f"the batch 1 is {batches[1]}")
        # assert len(batches) == 1, "Can only have one batch when collating"
        return {"batch": _add_dimension(self.creator.cleanup_batch(batches[0]))}

    def convert_dataset(
        self, db: meds_reader.SubjectDatabase, tokens_per_batch: int, min_subjects_per_batch: int = 1, num_proc: int = 1
    ):
        """Convert an entire dataset to batches.

        Arguments:
            dataset: A huggingface dataset containing MEDS subjects
            tokens_per_batch: The number of tokens allowed per batch
            min_subjects_per_batch: The minimum number of subjects per batch
            num_proc: The number of processers to use when converting

        Returns:
            A huggingface dataset object containing batches
        """

        max_length = tokens_per_batch // min_subjects_per_batch
        print(f"max_length: {max_length}")

        length_chunks = tuple(
            db.map(
                functools.partial(map_preliminary_batch_stats, processor=self, max_length=max_length),
            )
        )

        # is it np.array[(subject_id, start_index, length, fraction)] length=num_subjects?
        lengths = np.concatenate(length_chunks)

        # Sort by subject_id (column 0) to group same subjects together
        order = np.argsort(lengths[:, 0])
        lengths = lengths[order, :]

        # Shuffle the subjects to avoid bias in the order of the batches
        rng = np.random.default_rng(342342)
        rng.shuffle(lengths)

        assert len(lengths) != 0

        # Column 2 is sequence length
        # Column 3 is fraction of labels to sample
        # Column 1 is start_index
        # Column 0 is subject_id
        current_batch_length = 0

        batch_offsets = [0]

        # Column 2 is sequence length
        for i, length in enumerate(lengths[:, 2]):
            if current_batch_length + length > tokens_per_batch:
                batch_offsets.append(i)
                current_batch_length = 0

            current_batch_length += length

        batch_offsets.append(len(lengths))

        # print(f"batch_offsets: {batch_offsets}")

        # batches is a list of tuples (start_index, end_index) indicating the start and end subject indices of each batch
        batches = list(zip(batch_offsets, batch_offsets[1:]))
        # print(f"batches: {batches}")
        print("Got batches", len(batches))

        split_batches = np.array_split(batches, num_proc)

        final_batch_data = []

        # i denote the process id, batch_part is an array of tuples (start_index, end_index) indicating the start and end subject indices of each batch 
        #eg: batch_part=array([(0,5), (5,12), (12,18)])
        for i, batch_part in enumerate(split_batches):
            if len(batch_part) == 0:
                continue
            start = batch_part[0][0]
            end = batch_part[-1][-1]
            # lengths_part is an array of tuples (subject_id, start_index, length, fraction)
            # here lengths_part = lengths[0:18, :]
            lengths_part = lengths[start:end, :]

            for j, (a, b) in enumerate(batch_part):
                assert a != b, f'{a} {b} {i} {j}'

            offsets = [0] + [b - start for _, b in batch_part]

            final_batch_data.append(
                (
                    lengths_part,
                    np.array(offsets, dtype=np.int32),
                )
            )

        batch_func = functools.partial(
            _batch_generator,
            creator=self.creator,
            path_to_database=db.path_to_database,
        )

        batch_dataset = datasets.Dataset.from_generator(
            batch_func,
            gen_kwargs={
                "batch_data": final_batch_data,
            },
            num_proc=num_proc // 4,
            writer_batch_size=8,
        )

        return batch_dataset
