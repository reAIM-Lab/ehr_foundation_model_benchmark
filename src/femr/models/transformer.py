from __future__ import annotations

import collections
import math
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple
from dataclasses import dataclass

import meds
import meds_reader
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import xformers.ops
from torch import nn
from tqdm import tqdm
from torch.profiler import ProfilerActivity, profile

import femr.models.config
import femr.models.processor
import femr.models.rmsnorm
import femr.models.tasks
import femr.models.tokenizer
import femr.models.xformers


@dataclass(frozen=False)
class TotalFlops:
    total_flops: int = 0


# From https://github.com/kingoflolz/mesh-transformer-jax
def rotate_every_two_v2(x):
    flat_x = x.reshape(-1, x.shape[-1])

    x1 = flat_x[:, ::2]
    x2 = flat_x[:, 1::2]

    result = torch.stack((-x2, x1), axis=-1).reshape(x.shape)

    assert x.dtype == result.dtype
    return result


def fixed_pos_embedding(ages, dim, dtype):
    assert ages.dtype == torch.float32
    assert len(ages.shape) == 1

    inv_freq = 1.0 / (10000 ** (torch.linspace(0, 2, steps=dim // 2, device=ages.device)))
    inv_freq = inv_freq.reshape(1, 1, dim // 2)
    assert inv_freq.dtype == torch.float32

    ages = ages.reshape(ages.shape[0], 1)

    t = inv_freq * ages

    sin, cos = torch.sin(t), torch.cos(t)

    final_shape = (ages.shape[0], 1, dim)

    sin = torch.stack((sin, sin), axis=-1).reshape(final_shape).type(dtype)
    cos = torch.stack((cos, cos), axis=-1).reshape(final_shape).type(dtype)

    return sin, cos


def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    sin = sin.to(dtype=x.dtype)
    cos = cos.to(dtype=x.dtype)

    assert x.dtype == sin.dtype == cos.dtype, f"{x.dtype} {sin.dtype} {cos.dtype}"

    if len(sin.shape) != len(x.shape):
        new_shape = (1,) + sin.shape
        sin = sin.reshape(new_shape)
        cos = cos.reshape(new_shape)

    return (x * cos) + (rotate_every_two_v2(x) * sin)


class FEMREncoderLayer(nn.Module):
    def __init__(self, config: femr.models.config.FEMRTransformerConfig):
        super().__init__()
        self.config = config
        self.norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        if self.config.hidden_act == "swiglu":
            hidden_mult = 2
        else:
            hidden_mult = 1

        self.input_proj = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size * 3 + hidden_mult * self.config.intermediate_size,
            bias=self.config.use_bias,
        )

        self.output_proj = nn.Linear(
            self.config.hidden_size + self.config.intermediate_size, self.config.hidden_size, bias=self.config.use_bias
        )

    def forward(self, x, time_data, pos_embed, attn_bias, s):
        x = self.norm(x)

        if self.config.use_normed_ages:
            all_time = torch.concatenate((time_data, time_data ** 2), axis=-1)
            x[:, -all_time.shape[1]:] = all_time.to(dtype=x.dtype)

        transformed = self.input_proj(x)

        ff = transformed[:, : -self.config.hidden_size * 3]
        qkv = transformed[:, -self.config.hidden_size * 3:]

        head_size = self.config.hidden_size // self.config.n_heads

        qkv = qkv.reshape(x.shape[0], 3, self.config.n_heads, head_size)

        # it doesn't have absolute time as input
        q = apply_rotary_pos_emb(qkv[:, 0, :, :], pos_embed)
        k = apply_rotary_pos_emb(qkv[:, 1, :, :], pos_embed)
        v = qkv[:, 2, :, :]

        attn = femr.models.xformers.memory_efficient_attention_wrapper(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            attn_bias=attn_bias,
        )

        attn = attn.reshape(x.shape)

        if self.config.hidden_act == "gelu":
            ff = F.gelu(ff)
        elif self.config.hidden_act == "swiglu":
            x1, x2 = ff.chunk(2, dim=-1)
            ff = F.silu(x1) * x2

        combined = torch.concatenate((attn, ff), axis=-1)
        result = self.output_proj(combined)

        return result


class FEMRTransformer(nn.Module):
    def __init__(self, config: femr.models.config.FEMRTransformerConfig):
        super().__init__()
        self.config = config

        self.in_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        self.out_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)

        if not self.config.is_hierarchical:
            self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        else:
            self.embed_bag = nn.EmbeddingBag(
                num_embeddings=self.config.vocab_size,
                embedding_dim=self.config.hidden_size,
                mode="sum",
                include_last_offset=True,
            )

        self.layers = nn.ModuleList([FEMREncoderLayer(config) for i in range(self.config.n_layers)])

    def forward(self, batch, s):
        if not self.config.is_hierarchical:
            x = self.embed(batch["tokens"])
        else:
            x = self.embed_bag(batch["hierarchical_tokens"], batch["token_indices"], batch["hierarchical_weights"])

        x = self.in_norm(x)
        time_data = batch["time_data"]
        pos_embed = fixed_pos_embedding(batch["ages"], self.config.hidden_size // self.config.n_heads, x.dtype)

        attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            batch["subject_lengths"].tolist()
        ).make_local_attention(self.config.attention_width)

        for layer in self.layers:
            x = x + layer(x, time_data, pos_embed, attn_bias, s)

        final = self.out_norm(x)

        return final


class LabeledSubjectTaskHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        return 0, {}


class CLMBRTaskHead(nn.Module):
    def __init__(self, hidden_size: int, clmbr_vocab_size: int):
        super().__init__()

        self.final_layer = nn.Linear(hidden_size, clmbr_vocab_size)

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        logits = self.final_layer(features)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)

        if not return_logits:
            logits = None

        return loss, {"logits": logits}


class MOTORTaskHead(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            linear_interpolation: bool,
            pretraining_task_info: List[Tuple[str, float]],
            time_bins: np.ndarray,
            final_layer_size: int,
    ):
        super().__init__()

        # Handle both numpy array and list (from config deserialization)
        if not isinstance(time_bins, np.ndarray):
            time_bins = np.array(time_bins)
        self.num_time_bins = time_bins.shape[1] - 1  # Each task has same number of bins
        self.num_tasks = len(pretraining_task_info)
        self.time_bins = time_bins  # Store time bins for potential debugging

        self.final_layer_size = final_layer_size
        self.final_layer = nn.Linear(hidden_size, self.num_time_bins * final_layer_size)

        self.task_layer = nn.Linear(self.final_layer_size, self.num_tasks)
        self.softmax = nn.Softmax(dim=1)
        start_bias = torch.log2(torch.tensor([a[1] for a in pretraining_task_info], dtype=torch.float32))
        self.task_layer.bias.data = start_bias

      
        self.norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size)
        self.linear_interpolation = linear_interpolation

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        """
        MOTOR Task Head Forward Pass with Event-Specific Time Bins
        
        Key design principles:
        1. Each event type has its own time discretization (task-specific time bins)
        2. For each prediction point and task: exactly ONE time bin is marked as True in is_event
        3. The marked bin represents either:
           - An event occurring in that interval (is_censored=False): use f() = time_dependent_logits
           - Censoring occurring in that interval (is_censored=True): use 1-F() = integrated_logits
        4. Loss is calculated only for marked bins using appropriate likelihood function
        """
        # (num_predictions, hidden_size) -> (num_predictions, num_time_bins, final_layer_size)
        time_independent_features = self.final_layer(features).reshape(
            features.shape[0], self.num_time_bins, self.final_layer_size
        )

        # take the softmaxof the logits over the time bins, assume indenpendence between different event types conditional previous embeddings
        # time_dependent_logits: prediction_points*time_bins *event_types  [716, 8, 6100]
        
        # OPTION 1: Original approach without sigmoid (recommended)
        # This is numerically more stable than sigmoid + softmax
        task_logits = self.task_layer(self.norm(time_independent_features))
        
        # Clamp logits to prevent overflow in softmax
        # task_logits = torch.clamp(task_logits, min=-50, max=50)
        
        time_dependent_logits = self.softmax(task_logits)
        
        # OPTION 2: If you want sigmoid, use it INSTEAD of softmax, not both
        # Uncomment this block and comment out the above if you prefer sigmoid approach
        # task_logits = self.task_layer(self.norm(time_independent_features))
        # task_logits = torch.clamp(task_logits, min=-50, max=50)  # prevent overflow
        # time_dependent_logits = torch.sigmoid(task_logits)
        # # Normalize along time dimension to ensure probabilities sum to 1 over time
        # time_dependent_logits = time_dependent_logits / torch.sum(time_dependent_logits, dim=1, keepdim=True)
        
        # Debug: Check for NaN/inf values
        if torch.any(torch.isnan(time_dependent_logits)) or torch.any(torch.isinf(time_dependent_logits)):
            print(f"ERROR: NaN/inf detected in time_dependent_logits")
            print(f"  NaN count: {torch.sum(torch.isnan(time_dependent_logits))}")
            print(f"  Inf count: {torch.sum(torch.isinf(time_dependent_logits))}")
            print(f"  Raw logits stats - min: {torch.min(task_logits)}, max: {torch.max(task_logits)}")
            raise ValueError("NaN/inf detected in time_dependent_logits")
            
        assert torch.allclose(sum(time_dependent_logits[0,:,0]), torch.tensor(1.0), atol=1e-1), f" time_dependent_logits: {time_dependent_logits[0,:,0]}"
        #
        # integrated_logits = 1 - torch.cumsum(time_dependent_logits, dim=1)
        cdf = torch.cumsum(time_dependent_logits, dim=1)
        integrated_logits = torch.cat([torch.ones_like(time_dependent_logits[:, :1, :]), 1.0 - cdf[:, :-1, :]], dim=1)
        # Verify input shapes match our expectations
        assert (
                batch["is_event"].shape == time_dependent_logits.shape
        ), f"Shape mismatch: time_dependent_logits {time_dependent_logits.shape} vs is_event {batch['is_event'].shape}"
        
        # Add numerical stability - clamp values to prevent log(0)
        eps = 1e-8
        time_dependent_logits_stable = torch.clamp(time_dependent_logits, min=eps, max=1.0-eps)
        integrated_logits_stable = torch.clamp(integrated_logits, min=eps, max=1.0-eps)
        

        # Validate that exactly one bin per prediction-task combination is True
        labels_sum = torch.sum(batch["is_event"], dim=1)  # Sum along time bins dimension [prediction_points, tasks]
        if not torch.all(labels_sum == 1):
            print(f"ERROR: Expected exactly 1 True bin per prediction-task combination")
            print(f"  Found {torch.sum(labels_sum != 1)} invalid combinations")
            print(f"  Labels sum range: {torch.min(labels_sum)} to {torch.max(labels_sum)}")
            
        # Calculate loss only for the marked bins
        # For each prediction point and task, exactly one bin should be True
        # Use f() for events, 1-F() for censoring
        
        # Get the marked bins: where is_event is True
        marked_bins = batch["is_event"]  # [prediction_points, time_bins, tasks]
        event_in_bins = batch["event_in_bin"]
        censor_in_bins = batch["censor_in_bin"]

        
        # For event cases: use f() = time_dependent_logits
        # For censoring cases: use 1-F() = integrated_logits  
        # is_censored has shape [prediction_points, tasks], need to expand to match marked_bins
        is_censored_expanded = batch["is_censored"].unsqueeze(1).expand(-1, self.num_time_bins, -1)  # [prediction_points, time_bins, tasks]
        
        if self.linear_interpolation:
            censor_time_ratio = batch["censor_time_ratio"]
            integrated_logits_stable = torch.where(
                is_censored_expanded,
                integrated_logits_stable - (censor_time_ratio * time_dependent_logits_stable),
                integrated_logits_stable  # No adjustment where not censored
            )
            integrated_logits_stable = torch.clamp(integrated_logits_stable, min=eps, max=1.0-eps)

        # event probability
        event_probs = torch.where(
            event_in_bins,
            time_dependent_logits_stable,
            torch.zeros_like(event_in_bins)
        )

        censor_probs = torch.where(
            is_censored_expanded,
            integrated_logits_stable,
            torch.zeros_like(is_censored_expanded)
        )

        # Select the appropriate probability based on event vs censoring
        selected_probs = torch.where(
            is_censored_expanded,
            integrated_logits_stable,  # Use 1-F() for censoring
            time_dependent_logits_stable  # Use f() for events
        )
        
        # Calculate loss only for marked bins
        loss_values = torch.where(
            marked_bins,
            torch.log(selected_probs),
            torch.zeros_like(selected_probs)  # No contribution from unmarked bins
        )

        assert event_in_bins+censor_in_bins==marked_bins, f" event {event_in_bins} censor {censor_in_bins} take & {marked_bins}"
        assert event_in_bins.shape == censor_in_bins.shape
        assert marked_bins.shape == censor_in_bins.shape
        assert torch.sum(loss_values) == torch.sum(event_probs) + torch.sum(censor_probs), f"the loss for all, event, censor are {torch.sum(loss_values)}, {torch.sum(event_probs)},{torch.sum(censor_probs)}"
        
        print(f"total loss {-torch.sum(loss_values) / num_marked_bins}")
        print(f"event loss {-torch.sum(event_probs) / event_in_bins}")
        print(f"censor loss {-torch.sum(censor_in_bins) / censor_in_bins}")

        # Average over all marked bins (should be exactly one per prediction-task combination)
        num_marked_bins = torch.sum(marked_bins)
        if num_marked_bins > 0:
            loss = -torch.sum(loss_values) / num_marked_bins  # Negative log likelihood
        else:
            loss = torch.tensor(0.0, device=marked_bins.device)
        
        # Debug: Check for issues
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/inf detected in final loss: {loss}")
            print(f"  num_marked_bins: {num_marked_bins}")
            print(f"  time_dependent_logits range: {torch.min(time_dependent_logits_stable)} to {torch.max(time_dependent_logits_stable)}")
            print(f"  integrated_logits range: {torch.min(integrated_logits_stable)} to {torch.max(integrated_logits_stable)}")
            print(f"  selected_probs range: {torch.min(selected_probs[marked_bins])} to {torch.max(selected_probs[marked_bins])}")

        if not return_logits:
            time_dependent_logits = None

        return loss, {"time_dependent_logits": time_dependent_logits}


def remove_first_dimension(data: Any) -> Any:
    if isinstance(data, collections.abc.Mapping):
        return {k: remove_first_dimension(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        assert data.shape[0] == 1
        return data.squeeze(dim=0)
    elif isinstance(data, np.ndarray):
        assert data.shape[0] == 1
        return np.squeeze(data, axis=0)
    elif isinstance(data, (int, float, np.number, np.bool_)):
        return data
    else:
        raise RuntimeError("Could not convert item of type " + str(type(data)))


class FEMRModel(transformers.PreTrainedModel):
    config_class = femr.models.config.FEMRModelConfig
    
    def __init__(self, config: femr.models.config.FEMRModelConfig, **kwargs):
        # Extract linear_interpolation from kwargs, default to False
        self.linear_interpolation = kwargs.pop('linear_interpolation', False)
        
        # Allow the task config to be ovewritten
        if "task_config" in kwargs:
            config.task_config = kwargs["task_config"]

        super().__init__(config, **kwargs)

        self.transformer = FEMRTransformer(self.config.transformer_config)
        if self.config.task_config is not None:
            self.task_model = self.create_task_head()

        

    def create_task_head(self) -> nn.Module:
        hidden_size = self.config.transformer_config.hidden_size
        task_type = self.config.task_config.task_type
        task_kwargs = self.config.task_config.task_kwargs
        if task_type == "clmbr":
            return CLMBRTaskHead(hidden_size, **task_kwargs)
        elif task_type == "labeled_subjects":
            return LabeledSubjectTaskHead(hidden_size, **task_kwargs)
        elif task_type == "motor":
            return MOTORTaskHead(hidden_size, self.linear_interpolation, **task_kwargs)
        else:
            raise RuntimeError("Could not determine head for task " + task_type)

    def forward(self, batch: Mapping[str, Any], return_loss=True, return_logits=False, return_reprs=False):
        # Need a return_loss parameter for transformers.Trainer to work properly
        assert return_loss

        batch = remove_first_dimension(batch)
        input_device = batch['subject_ids'].device
        s = torch.zeros_like(batch['subject_ids'], device=input_device)
        # s = torch.zeros_like(batch['subject_ids'])
        s[1:] = batch['subject_ids'][1:] != batch['subject_ids'][:-1]
        s = torch.cumsum(s, dim=0).type(torch.uint8)

        # (time_steps, hidden_size)
        features = self.transformer(batch["transformer"], s)
        if "task" in batch and self.config.task_config is not None:
            features = features.reshape(-1, features.shape[-1])
            features = features[batch["transformer"]["label_indices"], :]
            # print(f"features before forward: {features.shape}")
            
            loss, result = self.task_model(features, batch["task"], return_logits=return_logits)
            if return_reprs:
                result["representations"] = features
            if return_logits or return_reprs:
                result["timestamps"] = batch["transformer"]["timestamps"][batch["transformer"]["label_indices"]]
                result["subject_ids"] = batch["subject_ids"][batch["transformer"]["label_indices"]]
            return loss, result
        else:
            loss = 0
            features = features.reshape(-1, features.shape[-1])
            if "task" in batch:
                features = features[batch["transformer"]["label_indices"], :]
                result = {
                    "timestamps": batch["transformer"]["timestamps"][batch["transformer"]["label_indices"]],
                    "subject_ids": batch["subject_ids"][batch["transformer"]["label_indices"]],
                    "representations": features,
                }
            else:
                result = {
                    "timestamps": batch["transformer"]["timestamps"],
                    "subject_ids": batch["subject_ids"],
                    "representations": features,
                }

            return loss, result


def to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, collections.abc.Mapping):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (int, float, np.number, np.bool_)):
        return data
    else:
        raise RuntimeError("Could not move item of type " + str(type(data)))


def compute_features(
        db: meds_reader.SubjectDatabase,
        model_path: str,
        labels: List[meds.Label],
        num_proc: int = 1,
        tokens_per_batch: int = 1024,
        device: Optional[torch.device] = None,
        ontology: Optional[femr.ontology.Ontology] = None,
        observation_window: Optional[int] = None,
        min_subjects_per_batch: int = 1,
        total_flops: TotalFlops = None,
        use_linear_interpolation: bool = False,
) -> Dict[str, np.ndarray]:
    """ "Compute features for a set of labels given a dataset and a model.

    Arguments:
        dataset: A HuggingFace dataset containing MEDS subjects
        model_path: A path to a saved pretrained model, including a saved tokenizer
        labels: MEDS labels to compute features for
        num_proc: The number of processors to use
        tokens_per_batch: The maximum number of tokens per batch
        device: Which type of compute to use
        ontology: A FEMR ontology object, which is necessary for models that use a hierarchical tokenizer
        observation_window: The observation window in which the features are extracted
        total_flops: TotalFlops to record the total number of flops

    Returns:
        A dictionary of numpy arrays, with three keys, "subject_ids", "feature_times" and "features"
         -  "subject_ids" and "feature_times" define the subject and time each feature refers to
         -  "features" provides the representations at each subject id and feature time
    """
    task = femr.models.tasks.LabeledSubjectTask(labels, observation_window)

    print(f"Loading model from {model_path}")
    print(f"use_linear_interpolation: {use_linear_interpolation}")
    
    # Use the new from_pretrained method that supports linear_interpolation
    model = femr.models.transformer.FEMRModel.from_pretrained(
        model_path, 
        task_config=task.get_task_config(),
        linear_interpolation=use_linear_interpolation
    )

    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(model_path, ontology=ontology)
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task=task)

    filtered_data = db.filter(list(task.label_map.keys()))

    if device:
        model = model.to(device)

    cpu_device = torch.device("cpu")

    print(f"The maximum context length is {tokens_per_batch/min_subjects_per_batch},  {min_subjects_per_batch} subjects and {tokens_per_batch} tokens per batch")
    batches = processor.convert_dataset(
        filtered_data, tokens_per_batch=tokens_per_batch, min_subjects_per_batch=min_subjects_per_batch, num_proc=num_proc
    )

    batches.set_format("pt")

    loader = torch.utils.data.DataLoader(batches, num_workers=num_proc, pin_memory=True, collate_fn=processor.collate)

    all_subject_ids = []
    all_feature_times = []
    all_representations = []

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in tqdm(loader):
                if device:
                    batch = to_device(batch, device)

                if total_flops:
                    with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            with_flops=True,
                    ) as prof:
                        _, result = model(**batch, return_reprs=True)

                    for event in prof.key_averages():
                        if hasattr(event, "flops") and event.flops > 0:
                            # Convert to GFLOPs
                            total_flops.total_flops += event.flops / 1e9
                else:
                    _, result = model(**batch, return_reprs=True)

                all_subject_ids.append(result["subject_ids"].to(cpu_device, non_blocking=True))
                all_feature_times.append(result["timestamps"].to(cpu_device, non_blocking=True))
                all_representations.append(result["representations"].to(cpu_device, non_blocking=True))

    torch.cuda.synchronize()

    all_subject_ids_np = torch.concatenate(all_subject_ids).numpy()
    all_feature_times_np = torch.concatenate(all_feature_times).numpy()
    all_representations_np = torch.concatenate(all_representations).numpy()

    return {
        "subject_ids": all_subject_ids_np,
        "feature_times": all_feature_times_np.astype("datetime64[s]"),
        "features": all_representations_np,
    }
