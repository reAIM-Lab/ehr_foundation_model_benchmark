from __future__ import annotations

import collections
import math
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
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

import femr.models_tpp.config
import femr.models_tpp.processor
import femr.models_tpp.rmsnorm
import src.femr.models_tpp.tasks_mtpp
import femr.models_tpp.tokenizer
import femr.models_tpp.xformers


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
    def __init__(self, config: femr.models_tpp.config.FEMRTransformerConfig):
        super().__init__()
        self.config = config
        self.norm = femr.models_tpp.rmsnorm.RMSNorm(self.config.hidden_size)
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

        attn = femr.models_tpp.xformers.memory_efficient_attention_wrapper(
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
    def __init__(self, config: femr.models_tpp.config.FEMRTransformerConfig):
        super().__init__()
        self.config = config

        self.in_norm = femr.models_tpp.rmsnorm.RMSNorm(self.config.hidden_size)
        self.out_norm = femr.models_tpp.rmsnorm.RMSNorm(self.config.hidden_size)

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
            non_numerical_task: List[str],
            numerical_task: List[str],
            value_bins: Union[np.ndarray, List],
            non_numerical_task_time_bins: Union[np.ndarray, List],
            numerical_task_time_bins: Union[np.ndarray, List],
            final_layer_size: int,
            # num_value_bins: int = 10,
    ):
        super().__init__()

        # Handle both numpy array and list (from config deserialization)
        if not isinstance(value_bins, np.ndarray):
            value_bins = np.array(value_bins)
        if not isinstance(non_numerical_task_time_bins, np.ndarray):
            non_numerical_task_time_bins = np.array(non_numerical_task_time_bins)
        if not isinstance(numerical_task_time_bins, np.ndarray):
            numerical_task_time_bins = np.array(numerical_task_time_bins)
            
        assert value_bins.shape[0] == numerical_task_time_bins.shape[0], "no matching dimension for number of numerical tasks"
        assert non_numerical_task_time_bins.shape[1] == numerical_task_time_bins.shape[1], "no matching dimension for time bins"

        self.num_time_bins = numerical_task_time_bins.shape[1] - 1  # Each task has same number of bins
        # self.time_bins = time_bins  # Store time bins for potential debugging
        self.non_numerical_task_time_bins = non_numerical_task_time_bins
        self.numerical_task_time_bins = numerical_task_time_bins
        self.num_value_bins = value_bins.shape[1] - 1
        self.value_bins = value_bins
        
        # Get task counts from input data structure - we'll determine from batch data
        # For now, assume we'll get the correct task counts from the batch
        self.numerical_tasks = numerical_task
        self.non_numerical_tasks = non_numerical_task

        self.final_layer_size = final_layer_size

        # print(f"non_numerical_task: {len(non_numerical_task)}")
        # print(f"numerical_task: {len(numerical_task)}")
        # print(f"value_bins: {value_bins.shape}")
        # print(f"non_numerical_task_time_bins: {non_numerical_task_time_bins.shape}")
        # print(f"numerical_task_time_bins: {numerical_task_time_bins.shape}")
        
        # Original layer for non-numerical codes: time prediction only
        self.non_numerical_final_layer = nn.Linear(hidden_size, self.num_time_bins * final_layer_size)
        self.non_numerical_task_layer = nn.Linear(self.final_layer_size, len(self.non_numerical_tasks))  # All tasks initially

        
        # Additional layer for numerical codes: time × value prediction
        self.numerical_final_layer = nn.Linear(hidden_size, self.num_time_bins * self.num_value_bins * final_layer_size)
        self.numerical_task_layer = nn.Linear(self.final_layer_size, len(self.numerical_tasks))
        # We'll determine the actual number of numerical tasks from batch data
        # self.numerical_task_layer = nn.Linear(self.final_layer_size, num_numerical_tasks).to(features.device)
        
        self.softmax = nn.Softmax(dim=1)
        self.norm = femr.models_tpp.rmsnorm.RMSNorm(self.final_layer_size)
        self.linear_interpolation = linear_interpolation
        self.H_t  = math.log(self.num_time_bins)
        self.H_tv = math.log(self.num_time_bins * self.num_value_bins)
        
        # Set initial biases
        non_numerical_start_bias = torch.log2(torch.tensor([a[1] for a in pretraining_task_info if a[0] in self.non_numerical_tasks], dtype=torch.float32))
        numerical_start_bias = torch.log2(torch.tensor([a[1] for a in pretraining_task_info if a[0] in self.numerical_tasks], dtype=torch.float32))
        assert len(non_numerical_start_bias)+len(numerical_start_bias) == len(pretraining_task_info), f"the length of non_numerical, numerical and all are: {len(non_numerical_start_bias)},{len(numerical_start_bias)},{len(pretraining_task_info)}"

        self.non_numerical_task_layer.bias.data = non_numerical_start_bias
        self.numerical_task_layer.bias.data = numerical_start_bias

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        """
        Efficient MOTOR Task Head Forward Pass
        
        Architecture follows original approach:
        1. Non-numerical codes: prediction_points × time_bins × non_numerical_tasks  
        2. Numerical codes: prediction_points × time_bins × value_bins × numerical_tasks
        3. Probability sum over time_bins × value_bins = 1 for numerical codes
        4. New loss = mean loss for non-numerical + mean loss for numerical codes
        """
        eps = 1e-8
        total_loss = torch.tensor(0.0, device=features.device)
        loss_count = 0
        result = {}
        
        # ========== NON-NUMERICAL CODES (Original Logic) ==========
            # Time-dependent features for non-numerical codes
        time_independent_features = self.non_numerical_final_layer(features).reshape(
            features.shape[0], self.num_time_bins, self.final_layer_size
        )
        
        # Non-numerical task logits: [prediction_points, time_bins, non_numerical_tasks]
        task_logits = self.non_numerical_task_layer(self.norm(time_independent_features))
        time_dependent_logits = self.softmax(task_logits)
        
        # Debug: Check for NaN/inf values  
        if torch.any(torch.isnan(time_dependent_logits)) or torch.any(torch.isinf(time_dependent_logits)):
            print(f"ERROR: NaN/inf detected in time_dependent_logits")
            raise ValueError("NaN/inf detected in time_dependent_logits")
        
        # Verify probability sums to 1 over time bins
        prob_sums = torch.sum(time_dependent_logits, dim=1)  # Sum over time bins
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-2), f"Probability sums: {prob_sums[0]}"
        
        # Calculate CDF for survival analysis
        cdf = torch.cumsum(time_dependent_logits, dim=1)
        integrated_logits = torch.cat([torch.ones_like(time_dependent_logits[:, :1, :]), 1.0 - cdf[:, :-1, :]], dim=1)
        
        # Add numerical stability
        time_dependent_logits_stable = torch.clamp(time_dependent_logits, min=eps, max=1.0-eps)
        integrated_logits_stable = torch.clamp(integrated_logits, min=eps, max=1.0-eps)
        
        # Linear interpolation adjustment if enabled
        if self.linear_interpolation:
            censor_time_ratio = batch["non_numerical_censor_time_ratio"]
            is_censored_expanded = batch["non_numerical_is_censored"].unsqueeze(1).expand(-1, self.num_time_bins, -1)
            integrated_logits_stable = torch.where(
                is_censored_expanded,
                integrated_logits_stable - (censor_time_ratio * time_dependent_logits_stable),
                integrated_logits_stable
            )
            integrated_logits_stable = torch.clamp(integrated_logits_stable, min=eps, max=1.0-eps)
        
        # Verify input shapes match expectations
        assert batch["non_numerical_is_event"].shape == time_dependent_logits.shape, \
            f"Shape mismatch: time_dependent_logits {time_dependent_logits.shape} vs is_event {batch['non_numerical_is_event'].shape}"
        
        # Validate exactly one bin per prediction-task combination
        labels_sum = torch.sum(batch["non_numerical_is_event"], dim=1)
        assert torch.all(labels_sum == 1), f"Expected exactly 1 True bin per prediction-task combination"
        
        # Loss calculation for non-numerical codes
        marked_bins = batch["non_numerical_is_event"]
        is_censored_expanded = batch["non_numerical_is_censored"].unsqueeze(1).expand(-1, self.num_time_bins, -1)
        
        # Select appropriate probabilities
        selected_probs = torch.where(
            is_censored_expanded,
            integrated_logits_stable,  # Use 1-F() for censoring
            time_dependent_logits_stable  # Use f() for events
        )
        
        # Calculate loss only for marked bins
        loss_values = torch.where(
            marked_bins,
            torch.log(selected_probs),
            torch.zeros_like(selected_probs)
        )
        
        num_marked_bins = torch.sum(marked_bins)
        if num_marked_bins > 0:
            non_numerical_loss = -torch.sum(loss_values) / (num_marked_bins*self.H_t)
            total_loss += non_numerical_loss
            loss_count += 1
        else:
            print("No non-numerical codes found in batch")
        
        if return_logits:
            result["time_dependent_logits"] = time_dependent_logits
        
        # ========== NUMERICAL CODES (New Matrix-wise Logic) ==========  
        if "numerical_time_event_bin" in batch:
            # Get dimensions from batch data 
            # print(batch.keys())
            time_event_in_bin = batch["numerical_time_event_bin"]  # [pred_points, time_bins, numerical_tasks]
            time_censor_in_bin = batch["numerical_time_censor_bin"]  # [pred_points, time_bins, numerical_tasks]  
            value_event_in_bin = batch["numerical_value_event_bin"]  # [pred_points, value_bins, numerical_tasks]
            numerical_is_censored = batch["numerical_is_censored"]  # [pred_points, numerical_tasks]
            
            batch_size, _, num_numerical_tasks = time_event_in_bin.shape
            # _, value_bins, _ = value_event_in_bin.shape
            
            # print(f"time bins is {self.num_time_bins}")
            # print(f"value bins is {self.num_value_bins}")
            # Time-value features: [pred_points, time_bins, value_bins, final_layer_size]
            numerical_features = self.numerical_final_layer(features).reshape(
                batch_size, self.num_time_bins*self.num_value_bins, self.final_layer_size
            )
            
            # Numerical task logits: [pred_points, time_bins, value_bins, numerical_tasks]
            numerical_task_logits = self.numerical_task_layer(self.norm(numerical_features))

            # Apply value-bin validity mask per numerical task before softmax so probabilities
            # are distributed only over valid value bins for each task.
            assert "value_valid_mask" in batch, "value_valid_mask is required for numerical masking"
            value_valid_mask_np = batch["value_valid_mask"]
            if isinstance(value_valid_mask_np, np.ndarray):
                value_valid_mask_tensor = torch.from_numpy(value_valid_mask_np).to(device=features.device)
            else:
                value_valid_mask_tensor = value_valid_mask_np.to(device=features.device)

            # Ensure mask is boolean per-bin: [num_tasks, num_value_bins]
            if value_valid_mask_tensor.shape[1] == self.num_value_bins + 1:
                vb = torch.as_tensor(self.value_bins, device=features.device, dtype=torch.float32)
                bin_increasing = vb[:, 1:] > vb[:, :-1]
                boundary_mask = value_valid_mask_tensor.to(torch.bool)
                value_valid_mask_bins = (boundary_mask[:, 1:] & boundary_mask[:, :-1] & bin_increasing)
            else:
                assert value_valid_mask_tensor.shape[1] == self.num_value_bins, (
                    f"value_valid_mask second dim {value_valid_mask_tensor.shape[1]} != num_value_bins {self.num_value_bins}")
                value_valid_mask_bins = value_valid_mask_tensor.to(torch.bool)

            valid_counts = value_valid_mask_bins.sum(dim=1)  # [num_tasks]
            assert torch.all(valid_counts > 0), "Each numerical task must have at least one valid value bin"

            per_task_value_mask = value_valid_mask_bins.t()  # [value_bins, num_tasks]
            tv_mask = per_task_value_mask.unsqueeze(0).expand(self.num_time_bins, -1, -1)  # [time_bins, value_bins, num_tasks]
            tv_mask_flat = tv_mask.reshape(self.num_time_bins * self.num_value_bins, num_numerical_tasks)

            tv_mask_flat = tv_mask_flat.unsqueeze(0).expand(batch_size, -1, -1)
            masked_logits = numerical_task_logits.masked_fill(~tv_mask_flat, -1e9)

            # Normalize over time_bins × value_bins to ensure probability sum = 1 (over valid bins)
            numerical_probs_flat = self.softmax(masked_logits)
            numerical_probs = numerical_probs_flat.reshape(batch_size, self.num_time_bins, self.num_value_bins, num_numerical_tasks)
            numerical_probs = torch.clamp(numerical_probs, min=eps, max=1.0-eps)
            
            # Verify probability sums to 1 over time_bins × value_bins
            prob_sums = torch.sum(numerical_probs, dim=(1, 2))  # Sum over time and value bins
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-2), f"Numerical probability sums: {prob_sums[0]}"
            # Assert that labeled value bins fall only on valid bins
            value_event_in_bin_bool = value_event_in_bin.to(torch.bool)
            valid_mask_broadcast = value_valid_mask_bins.t().unsqueeze(0)  # [1, value_bins, num_tasks]
            assert torch.all((~value_event_in_bin_bool) | valid_mask_broadcast), \
                "Found value events assigned to invalid value bins per task"

            # === MATRIX-WISE LOSS COMPUTATION (NO NESTED LOOPS) ===
            
            # Case 1: Censored cases - sum over value bins, only predict time
            # Use time_censor_in_bin directly for censored events
            numerical_loss = torch.tensor(0.0, device=features.device)
            num_numerical_losses = 0

            if torch.any(time_censor_in_bin):
                # Sum probabilities over value bins: [pred_points, time_bins, numerical_tasks]
                time_only_probs = torch.sum(numerical_probs, dim=2)
                cdf = torch.cumsum(time_only_probs, dim=1)
                integrated_logits = torch.cat([torch.ones_like(time_only_probs[:, :1, :]), 1.0 - cdf[:, :-1, :]], dim=1)
                integrated_logits_stable = torch.clamp(integrated_logits, min=eps, max=1.0-eps)

                # Censored loss using matrix operations
                # it is 1-F at censor positions otherwise
                censored_loss_values = torch.where(
                    time_censor_in_bin,
                    torch.log(integrated_logits_stable),
                    torch.zeros_like(time_only_probs)
                )
                
                censored_count = torch.sum(time_censor_in_bin)
                if censored_count > 0:
                    censored_loss = -torch.sum(censored_loss_values) / (censored_count*self.H_t)
                    numerical_loss += censored_loss
                    num_numerical_losses += 1
            # Case 2: Event cases - predict both time and value using outer product
            # Construct event_bins = time_event_in_bin ⊗ value_event_in_bin (outer product)  
            if torch.any(time_event_in_bin) and torch.any(value_event_in_bin):
                # Matrix-wise outer product: [pred_points, time_bins, numerical_tasks] ⊗ [pred_points, value_bins, numerical_tasks]
                # Result: [pred_points, time_bins, value_bins, numerical_tasks]
                event_bins = time_event_in_bin.unsqueeze(2) & value_event_in_bin.unsqueeze(1)
                
                # Event loss using matrix operations  
                event_loss_values = torch.where(
                    event_bins,
                    torch.log(numerical_probs),
                    torch.zeros_like(numerical_probs)
                )
                
                event_count = torch.sum(event_bins)
                if event_count > 0:
                    denom = torch.log(valid_counts.to(numerical_probs.dtype) + 1e-8)
                    denom = torch.clamp(denom, min=1.0)
                    denom_broadcast = denom.reshape(1, 1, 1, -1)
                    event_loss = -torch.sum(event_loss_values / denom_broadcast) / event_count
                    numerical_loss += event_loss
                    num_numerical_losses += 1
            
            if return_logits:
                result["value_dependent_logits"] = numerical_probs
            if num_numerical_losses < 2:
                print(f"censor cases are {torch.any(time_censor_in_bin)}")
                print(f"event cases are {torch.any(time_event_in_bin) and torch.any(value_event_in_bin)}")

            loss_count += 1
            numerical_loss_mean = numerical_loss / num_numerical_losses
            total_loss += numerical_loss_mean
        else:
            print("No numerical codes found in batch")
        # Final loss: mean of non-numerical and numerical losses
        # print(f"ratio {non_numerical_loss/numerical_loss}, non_numerical_loss is {non_numerical_loss} : numerical loss is {numerical_loss}, censor {censored_loss}, event {event_loss}")
        # loss = total_loss / loss_count
        loss = non_numerical_loss*0.75+numerical_loss_mean*0.25
        
        # Debug: Check for issues
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/inf detected in final loss: {loss}")
            print(f"  num_losses: {loss_count}")
            print(f"  total_loss: {total_loss}")

        return loss, result


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
    config_class = femr.models_tpp.config.FEMRModelConfig
    
    def __init__(self, config: femr.models_tpp.config.FEMRModelConfig, **kwargs):
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
    task = femr.models_tpp.tasks_mtpp.LabeledSubjectTask(labels, observation_window)

    print(f"Loading model from {model_path}")
    # print(f"use_linear_interpolation: {use_linear_interpolation}")
    
    # Use the new from_pretrained method that supports linear_interpolation
    model = femr.models_tpp.transformer.FEMRModel.from_pretrained(
        model_path, 
        task_config=task.get_task_config(),
    )

    tokenizer = femr.models_tpp.tokenizer.HierarchicalTokenizer.from_pretrained(model_path, ontology=ontology)
    processor = femr.models_tpp.processor.FEMRBatchProcessor(tokenizer, task=task)

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
