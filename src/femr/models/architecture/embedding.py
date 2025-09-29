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

import femr.models.config
import femr.models.processor
import femr.models.rmsnorm
import femr.models.tasks.motor
import femr.models.tokenizer
import femr.models.architecture.xformers


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


# class FEMREncoderLastLayer(nn.Module):
#     def __init__(self, config: femr.models.config.FEMRTransformerConfig):
#         super().__init__()
#         self.config = config
#         self.norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
#         if self.config.hidden_act == "swiglu":
#             hidden_mult = 2
#         else:
#             hidden_mult = 1

#         self.input_proj = nn.Linear(
#             self.config.hidden_size,
#             self.config.hidden_size * 3 + hidden_mult * self.config.intermediate_size,
#             bias=self.config.use_bias,
#         )

#         self.output_proj = nn.Linear(
#             self.config.hidden_size + self.config.intermediate_size, self.config.hidden_size*4, bias=self.config.use_bias
#         )

#     def forward(self, x, time_data, pos_embed, attn_bias, s):
#         x = self.norm(x)

#         if self.config.use_normed_ages:
#             all_time = torch.concatenate((time_data, time_data ** 2), axis=-1)
#             x[:, -all_time.shape[1]:] = all_time.to(dtype=x.dtype)

#         transformed = self.input_proj(x)

#         ff = transformed[:, : -self.config.hidden_size * 3]
#         qkv = transformed[:, -self.config.hidden_size * 3:]

#         head_size = self.config.hidden_size // self.config.n_heads

#         qkv = qkv.reshape(x.shape[0], 3, self.config.n_heads, head_size)

#         # it doesn't have absolute time as input
#         q = apply_rotary_pos_emb(qkv[:, 0, :, :], pos_embed)
#         k = apply_rotary_pos_emb(qkv[:, 1, :, :], pos_embed)
#         v = qkv[:, 2, :, :]

#         attn = femr.models.architecture.xformers.memory_efficient_attention_wrapper(
#             q.unsqueeze(0),
#             k.unsqueeze(0),
#             v.unsqueeze(0),
#             attn_bias=attn_bias,
#         )

#         attn = attn.reshape(x.shape)

#         if self.config.hidden_act == "gelu":
#             ff = F.gelu(ff)
#         elif self.config.hidden_act == "swiglu":
#             x1, x2 = ff.chunk(2, dim=-1)
#             ff = F.silu(x1) * x2

#         combined = torch.concatenate((attn, ff), axis=-1)
#         result = self.output_proj(combined)

#         return result

class FEMREncoderLayer(nn.Module):
    def __init__(self, config: femr.models.config.FEMRTransformerConfig,last_layer_bool=False):
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

        # if last_layer_bool == True:
        #     self.output_proj = nn.Linear(
        #         self.config.hidden_size + self.config.intermediate_size, self.config.hidden_size*4, bias=self.config.use_bias
        #     )
        # else:
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

        attn = femr.models.architecture.xformers.memory_efficient_attention_wrapper(
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
        # self.last_layer = FEMREncoderLayer(config,True)

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
        # x = x + self.last_layer(x, time_data, pos_embed, attn_bias, s)

        final = self.out_norm(x)
        return final
    
class FEMRMamba(nn.Module):
    def __init__(self, config: femr.models.config.FEMRMambaConfig):
        super().__init__()
        self.config = config

        # Build embeddings like transformer for identical input pipeline
        hidden_size = config.hidden_size
        emb_dim = hidden_size if hidden_size is not None else 768
        if not self.config.is_hierarchical:
            self.embed = nn.Embedding(self.config.vocab_size, emb_dim)
        else:
            self.embed_bag = nn.EmbeddingBag(
                num_embeddings=self.config.vocab_size,
                embedding_dim=emb_dim,
                mode="sum",
                include_last_offset=True,
            )

        self.in_norm = femr.models.rmsnorm.RMSNorm(emb_dim)
        self.out_norm = femr.models.rmsnorm.RMSNorm(emb_dim)

        # Load HF Mamba config and apply overrides
        hf_config = transformers.AutoConfig.from_pretrained(self.config.hf_name, trust_remote_code=True)

        # Adopt model size defaults from HF unless explicitly overridden
        # if self.config.hidden_size is not None:
        #     hf_config.d_model = self.config.hidden_size
        # if self.config.n_layers is not None:
        #     if hasattr(hf_config, "n_layer"):
        #         hf_config.n_layer = self.config.n_layers
        #     if hasattr(hf_config, "num_hidden_layers"):
        #         hf_config.num_hidden_layers = self.config.n_layers
        # if self.config.d_state is not None and hasattr(hf_config, "d_state"):
        #     hf_config.d_state = self.config.d_state
        # if hasattr(hf_config, "vocab_size"):
        #     hf_config.vocab_size = self.config.vocab_size

        hf_config.intermediate_size = self.config.intermediate_size
        hf_config.d_model = self.config.hidden_size
        hf_config.n_layer = self.config.n_layers
        hf_config.num_hidden_layers = self.config.n_layers
        hf_config.d_state = self.config.d_state
        hf_config.vocab_size = self.config.vocab_size

        # Freeform overrides
        # for key, val in (self.config.config_kwargs or {}).items():
        #     if hasattr(hf_config, key):
        #         setattr(hf_config, key, val)

        # Update embed dims and norms to match final d_model
        d_model = getattr(hf_config, "d_model", emb_dim)
        if not self.config.is_hierarchical:
            self.embed = nn.Embedding(self.config.vocab_size, d_model)
        else:
            self.embed_bag = nn.EmbeddingBag(
                num_embeddings=self.config.vocab_size,
                embedding_dim=d_model,
                mode="sum",
                include_last_offset=True,
            )
        self.in_norm = femr.models.rmsnorm.RMSNorm(d_model)
        self.out_norm = femr.models.rmsnorm.RMSNorm(d_model)

        # Instantiate backbone; use CausalLM variant for maximum compatibility
        print(f"the hf_config is {hf_config}")
        backbone_model = transformers.AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
        if hasattr(backbone_model, "backbone"):
            self.backbone = backbone_model.backbone
        elif hasattr(backbone_model, "model"):
            self.backbone = backbone_model.model
        else:
            self.backbone = backbone_model

    def forward(self, batch: Mapping[str, torch.Tensor], s):
        # Embedding path identical to transformer
        if not self.config.is_hierarchical:
            x = self.embed(batch["tokens"])  # (T, D)
        else:
            x = self.embed_bag(batch["hierarchical_tokens"], batch["token_indices"], batch["hierarchical_weights"])  # (T, D)

        x = self.in_norm(x)

        # Inject normalized ages/time features as in transformer
        if self.config.use_normed_ages:
            time_data = batch["time_data"]
            all_time = torch.concatenate((time_data, time_data ** 2), axis=-1)
            x[:, -all_time.shape[1]:] = all_time.to(dtype=x.dtype)

        # Efficient per-subject batching: pad sequences and process in parallel
        lengths = torch.tensor(batch["subject_lengths"],device=x.device)  # (S,)
        seqs = torch.split(x, lengths.tolist())  # List[(L_i, D)]
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # (S, L_max, D)
        attn_mask = torch.arange(padded.size(1), device=padded.device)[None, :].expand(len(seqs), -1) < lengths[:, None]


        # Forward through backbone using inputs_embeds; attention_mask may be ignored by some Mamba impls
        out = self.backbone(inputs_embeds=padded, attention_mask=attn_mask)
        # try:
        #     out = self.backbone(inputs_embeds=padded, attention_mask=attn_mask)
        # except TypeError:
        #     out = self.backbone(inputs_embeds=padded)
        # except Exception:
        #     out = self.backbone(padded)

        last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

        # Unpad back to flattened order
        chunks: List[torch.Tensor] = [last_hidden[i, : lengths[i], :] for i in range(len(seqs))]
        final = torch.cat(chunks, dim=0)  # (T, D)
        final = self.out_norm(final)
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
            pretraining_task_info: List[Tuple[str, float]],
            time_bins: List[float],
            final_layer_size: int,
    ):
        super().__init__()

        self.num_time_bins = len(time_bins) - 1
        self.num_tasks = len(pretraining_task_info)

        self.final_layer_size = final_layer_size
        self.final_layer = nn.Linear(hidden_size, self.num_time_bins * final_layer_size)

        self.task_layer = nn.Linear(self.final_layer_size, self.num_tasks)
        start_bias = torch.log2(torch.tensor([a[1] for a in pretraining_task_info], dtype=torch.float32))
        self.task_layer.bias.data = start_bias

        self.task_time_bias = nn.Parameter(torch.zeros(1, self.num_time_bins, self.num_tasks))

        self.norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size)

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        # b*hidden_size ^ (hidden_size,self.final_layer_size * self.num_time_bins) -> b*self.num_time_bins (8)*self.final_layer_size (512)
        time_independent_features = self.final_layer(features).reshape(
            features.shape[0], self.num_time_bins, self.final_layer_size
        )


        # self.final_layer_size (512) *(self.final_layer_size, K) -> K (non=5000, num=1500)
        time_dependent_logits = self.task_layer(self.norm(time_independent_features)) + self.task_time_bias
        # time_dependent_logits = self.task_layer(time_independent_features)

        assert (
                batch["log_time"].shape == time_dependent_logits.shape
        ), f"{time_dependent_logits.shape} {batch['log_time'].shape}"
        assert (
                batch["is_event"].shape == time_dependent_logits.shape
        ), f"{time_dependent_logits.shape} {batch['is_event'].shape}"

        # Force to always be negative
        # time_dependent_logits = -F.softplus(-time_dependent_logits)

        survival_loss = torch.exp2(time_dependent_logits + batch["log_time"]).mean()
        event_loss = -math.log(2) * torch.where(batch["is_event"], time_dependent_logits, 0).mean()

        def stats(a):
            a = a[torch.isfinite(a)]
            print(torch.mean(a), torch.std(a), torch.max(a), torch.min(a))

        loss = survival_loss + event_loss

        if not return_logits:
            time_dependent_logits = None

        return loss, {"time_dependent_logits": time_dependent_logits}

class TPPTaskHead(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            pretraining_task_info: List[Tuple[str, float]],
            time_bins: np.ndarray,
            final_layer_size: int,
            linear_interpolation: bool = False,
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
            torch.log(time_dependent_logits_stable),
            torch.zeros_like(event_in_bins)
        )

        censor_probs = torch.where(
            censor_in_bins,
            torch.log(integrated_logits_stable),
            torch.zeros_like(censor_in_bins)
        )

        # Select the appropriate probability based on event vs censoring
        selected_probs = torch.where(
            is_censored_expanded,  # true for all time bins for censor patients
            integrated_logits_stable,  # Use 1-F() for censoring
            time_dependent_logits_stable  # Use f() for events
        )
        
        # Calculate loss only for marked bins
        loss_values = torch.where(
            marked_bins,
            torch.log(selected_probs),
            torch.zeros_like(selected_probs)  # No contribution from unmarked bins
        )

        # print(f"event_in_bins {event_in_bins},{event_in_bins.shape}")
        # print(f"censor_in_bins {censor_in_bins},{censor_in_bins.shape}")
        # print(f"marked_bins {marked_bins},{marked_bins.shape}")
        assert torch.all(event_in_bins+censor_in_bins==marked_bins), f" event {event_in_bins} censor {censor_in_bins} take & {marked_bins}"
        assert event_in_bins.shape == censor_in_bins.shape
        assert marked_bins.shape == censor_in_bins.shape
        assert torch.sum(loss_values) - (torch.sum(event_probs) + torch.sum(censor_probs))<5, f"the loss for all, event, censor are {torch.sum(loss_values)}, {torch.sum(event_probs)},{torch.sum(censor_probs)}"
        

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

class MTPPTaskHead(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            pretraining_task_info: List[Tuple[str, float]],
            non_numerical_task: List[str],
            numerical_task: List[str],
            value_bins: Union[np.ndarray, List],
            non_numerical_task_time_bins: Union[np.ndarray, List],
            numerical_task_time_bins: Union[np.ndarray, List],
            final_layer_size: int,
            linear_interpolation: bool = False,
            # num_value_bins: int = 10,
    ):
        super().__init__()

        self.use_uncertainty = True
        self.logvars = nn.Parameter(torch.zeros(2))   # [non, num]
        self._logvar_clamp = (-3.0, 3.0)   

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
        
        # Original layer for non-numerical codes: time prediction only
        # print(f"hidden size is {hidden_size}")
        self.non_numerical_final_layer = nn.Linear(hidden_size, self.num_time_bins * final_layer_size)
        self.non_numerical_task_layer = nn.Linear(self.final_layer_size, len(self.non_numerical_tasks))  # All tasks initially

        
        # Additional layer for numerical codes: time × value prediction
        self.numerical_final_layer = nn.Linear(hidden_size, self.num_time_bins * self.num_value_bins * final_layer_size)
        self.numerical_task_layer = nn.Linear(self.final_layer_size, len(self.numerical_tasks))
        # We'll determine the actual number of numerical tasks from batch data
        # self.numerical_task_layer = nn.Linear(self.final_layer_size, num_numerical_tasks).to(features.device)
        
        self.softmax = nn.Softmax(dim=1)
        self.norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size)
        # self.linear_interpolation = linear_interpolation
        self.H_t  = math.log(self.num_time_bins)
        self.H_tv = math.log(self.num_time_bins * self.num_value_bins)
        
        # Set initial biases
        self.non_numerical_task_layer.bias.data.zero_()
        self.numerical_task_layer.bias.data.zero_()

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        """
        Efficient MOTOR Task Head Forward Pass
        
        Architecture follows original approach:
        1. Non-numerical codes: prediction_points × time_bins × non_numerical_tasks  
        2. Numerical codes: prediction_points × time_bins × value_bins × numerical_tasks
        3. Probability sum over time_bins × value_bins = 1 for numerical codes
        4. New loss = mean loss for non-numerical + mean loss for numerical codes
        """
        eps = 1e-9
        total_loss = torch.tensor(0.0, device=features.device)
        total_count = 0
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
        # prob_sums = torch.sum(time_dependent_logits, dim=1)  # Sum over time bins
        # assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-2), f"Probability sums: {prob_sums[0]}"
        
        # Calculate CDF for survival analysis
        cdf = torch.cumsum(time_dependent_logits, dim=1)
        integrated_logits = torch.cat([torch.ones_like(time_dependent_logits[:, :1, :]), 1.0 - cdf[:, :-1, :]], dim=1)
        
        # Add numerical stability
        time_dependent_logits_stable = torch.clamp(time_dependent_logits, min=eps)
        integrated_logits_stable = torch.clamp(integrated_logits, min=eps)
        
        # Verify input shapes match expectations
        assert batch["non_numerical_is_event"].shape == time_dependent_logits.shape, \
            f"Shape mismatch: time_dependent_logits {time_dependent_logits.shape} vs is_event {batch['non_numerical_is_event'].shape}"
        
        # Validate exactly one bin per prediction-task combination
        # labels_sum = torch.sum(batch["non_numerical_is_event"], dim=1)
        # assert torch.all(labels_sum == 1), f"Expected exactly 1 True bin per prediction-task combination"
        
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
            L_non = -torch.sum(loss_values) / (num_marked_bins)  # normalized by log B_t
            total_loss += -torch.sum(loss_values)
            total_count += num_marked_bins
        else:
            L_non = torch.tensor(0.0, device=features.device)

        
        if return_logits:
            result["time_dependent_logits"] = time_dependent_logits
        
        # ========== NUMERICAL CODES (New Matrix-wise Logic) ==========  
        if "numerical_time_event_bin" in batch:
            # Get dimensions from batch data 
            # print(batch.keys())
            time_event_in_bin = batch["numerical_time_event_bin"]  # [pred_points, time_bins, numerical_tasks]
            time_censor_in_bin = batch["numerical_time_censor_bin"]  # [pred_points, time_bins, numerical_tasks]
            value_event_in_bin = batch["numerical_value_event_bin"]  # [pred_points, value_bins, numerical_tasks]
            # numerical_is_censored = batch["numerical_is_censored"]  # [pred_points, numerical_tasks]
            
            batch_size, _, num_numerical_tasks = time_event_in_bin.shape
            # Time-value features: [pred_points, time_bins, value_bins, final_layer_size]
            numerical_features = self.numerical_final_layer(features).reshape(
                batch_size, self.num_time_bins*self.num_value_bins, self.final_layer_size
            )
            
            # Numerical task logits: [pred_points, time_bins*value_bins, numerical_tasks]
            numerical_task_logits = self.numerical_task_layer(self.norm(numerical_features))

            # Apply value_valid_mask to mask out invalid value bins before softmax
            # value_valid_mask: [numerical_tasks, value_bins] -> need to expand to [time_bins*value_bins, numerical_tasks]
            device = numerical_task_logits.device
            value_valid_mask = torch.tensor(batch["value_valid_mask"],device=device,dtype=torch.bool)

            if value_valid_mask.shape[1] - value_event_in_bin.shape[1] == 1:
                value_valid_mask = value_valid_mask[:,1:]
            
            # Expand valid mask to match value_event_in_bin shape: [1, value_bins, numerical_tasks]
            valid_mask_expanded = value_valid_mask.T.unsqueeze(0) 
            # Check that events only occur in valid bins: invalid events = events & ~valid_mask
            invalid_events = value_event_in_bin & (~valid_mask_expanded)  # [pred_points, value_bins, numerical_tasks]
            assert not torch.any(invalid_events), f"Events in bin found in invalid value bins {invalid_events}"



            mask_tv = valid_mask_expanded.expand(self.num_time_bins,-1,-1) #[num_time_bins,value_bins, numerical_tasks]
            expanded_mask = mask_tv.reshape(1,self.num_time_bins * self.num_value_bins, num_numerical_tasks)

            # Use large negative value instead of -inf
            mask_value = -1e10  # This effectively zeros out in softmax but avoids -inf
            numerical_task_logits = numerical_task_logits.masked_fill(~expanded_mask, mask_value)

            numerical_probs_flat = self.softmax(numerical_task_logits)

            # torch.logsumexp()
            # torch.exp(torch.logsum(numerical_log_probs_flat))

            # numerical_log_probs = numerical_log_probs_flat.reshape(batch_size, self.num_time_bins, self.num_value_bins, num_numerical_tasks)
            numerical_probs = numerical_probs_flat.reshape(batch_size, self.num_time_bins, self.num_value_bins, num_numerical_tasks)

            numerical_probs = numerical_probs.clamp_min(eps)

            # # assert check
            expanded_valid_mask_assert = mask_tv.unsqueeze(0).expand(batch_size,-1,-1,-1)
            numerical_probs = numerical_probs*expanded_valid_mask_assert

            assert torch.all(numerical_probs[~expanded_valid_mask_assert] <= eps), \
            "Non-zero probabilities found in invalid bins."
            assert torch.all(numerical_probs[expanded_valid_mask_assert] >= eps), \
            "zero probabilities found in valid bins."
            
            # === MATRIX-WISE LOSS COMPUTATION (NO NESTED LOOPS) ===
            
            # Case 1: Censored cases - sum over value bins, only predict time
            # Use time_censor_in_bin directly for censored events
            numerical_loss = torch.tensor(0.0, device=features.device)
            total_num_count = 0
            # event_count = torch.tensor(0.0,device=features.device)
            # censored_count = torch.tensor(0.0,device=features.device)
            L_num = torch.tensor(0.0, device=features.device)
            L_num_event = torch.tensor(0.0, device=features.device)
            L_num_cens  = torch.tensor(0.0, device=features.device)


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
                # if censored_count > 0:
                L_num_cens = -torch.sum(censored_loss_values) 
                numerical_loss += L_num_cens
                total_num_count += censored_count

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

                    L_num_event = -torch.sum(event_loss_values)
                    numerical_loss += L_num_event
                    total_num_count += event_count
            
            # ne  = event_count.float()
            # nc  = censored_count.float()
            # den = ne + nc
            if total_num_count > 0:
                L_num = numerical_loss/total_num_count
                total_count += total_num_count
                total_loss += numerical_loss
            else:
                L_num = torch.tensor(0.0, device=features.device)

            # if den > 0:
            #     w_evt = ne / den
            #     w_cen = nc / den
            #     L_num = w_evt * L_num_event + w_cen * L_num_cens
            # else:
            #     L_num = torch.tensor(0.0, device=features.device)

            if return_logits:
                result["loss_components"] = {
                    "L_non_num":       L_non.detach(),
                    "L_num_event":   L_num_event.detach(),
                    "L_num_censor":  L_num_cens.detach(),
                }
                result["value_dependent_logits"] = numerical_probs
        else:
            print("No numerical codes found in batch")

        # logvars = torch.clamp(self.logvars, *self._logvar_clamp)
        # w = torch.exp(-self.logvars)  # [w_non, w_num] = [exp(-s1), exp(-s2)]

        # if self.use_uncertainty:
        #     loss = w[0] * L_non + self.logvars[0] + w[1] * L_num + self.logvars[1]
        # else:
        #     # simple fallback (e.g., warmup for first epoch)

        # if args.
        #     loss = alpha * L_non + (1 - alpha) * L_num
        # else:
        loss = total_loss / total_count
        # print(f"loss is {loss}, L_non {L_non} L_num {L_num}")

        return loss, result
    
class MTPPSharedTaskHead(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            pretraining_task_info: List[Tuple[str, float]],
            non_numerical_task: List[str],
            numerical_task: List[str],
            value_bins: Union[np.ndarray, List],
            non_numerical_task_time_bins: Union[np.ndarray, List],
            numerical_task_time_bins: Union[np.ndarray, List],
            final_layer_size: int,
            linear_interpolation: bool = False,
            # num_value_bins: int = 10,
    ):
        super().__init__()

        self.use_uncertainty = True
        self.logvars = nn.Parameter(torch.zeros(2))   # [non, num]
        self._logvar_clamp = (-3.0, 3.0)   

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
        
        # Original layer for non-numerical codes: time prediction only
        # print(f"hidden size is {hidden_size}")
        # self.non_numerical_final_layer = nn.Linear(hidden_size, self.num_time_bins * final_layer_size)
        self.final_layer = nn.Linear(hidden_size, self.num_time_bins * self.num_value_bins * final_layer_size)

        self.non_numerical_task_layer = nn.Linear(self.final_layer_size*self.num_value_bins, len(self.non_numerical_tasks))  # All tasks initially
        
        # Additional layer for numerical codes: time × value prediction
        # self.numerical_final_layer = nn.Linear(hidden_size, self.num_time_bins * self.num_value_bins * final_layer_size)
        self.numerical_task_layer = nn.Linear(self.final_layer_size, len(self.numerical_tasks))
        # We'll determine the actual number of numerical tasks from batch data
        # self.numerical_task_layer = nn.Linear(self.final_layer_size, num_numerical_tasks).to(features.device)
        
        self.softmax = nn.Softmax(dim=1)
        self.numerical_norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size)
        self.non_numerical_norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size*self.num_value_bins)
        # self.linear_interpolation = linear_interpolation
        self.H_t  = math.log(self.num_time_bins)
        self.H_tv = math.log(self.num_time_bins * self.num_value_bins)
        
        # Set initial biases
        self.non_numerical_task_layer.bias.data.zero_()
        self.numerical_task_layer.bias.data.zero_()

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        """
        Efficient MOTOR Task Head Forward Pass
        
        Architecture follows original approach:
        1. Non-numerical codes: prediction_points × time_bins × non_numerical_tasks  
        2. Numerical codes: prediction_points × time_bins × value_bins × numerical_tasks
        3. Probability sum over time_bins × value_bins = 1 for numerical codes
        4. New loss = mean loss for non-numerical + mean loss for numerical codes
        """
        eps = 1e-9
        total_loss = torch.tensor(0.0, device=features.device)
        total_count = 0
        result = {}

        # print("using task shared")
        # ========== NON-NUMERICAL CODES (Original Logic) ==========
            # Time-dependent features for non-numerical codes
        time_independent_features = self.final_layer(features).reshape(
            features.shape[0], self.num_time_bins, self.num_value_bins*self.final_layer_size
        )
        
        # Non-numerical task logits: [prediction_points, time_bins, non_numerical_tasks]
        task_logits = self.non_numerical_task_layer(self.non_numerical_norm(time_independent_features))
        time_dependent_logits = self.softmax(task_logits)

        # Debug: Check for NaN/inf values  
        if torch.any(torch.isnan(time_dependent_logits)) or torch.any(torch.isinf(time_dependent_logits)):
            print(f"ERROR: NaN/inf detected in time_dependent_logits")
            raise ValueError("NaN/inf detected in time_dependent_logits")
        
        # Verify probability sums to 1 over time bins
        # prob_sums = torch.sum(time_dependent_logits, dim=1)  # Sum over time bins
        # assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-2), f"Probability sums: {prob_sums[0]}"
        
        # Calculate CDF for survival analysis
        cdf = torch.cumsum(time_dependent_logits, dim=1)
        integrated_logits = torch.cat([torch.ones_like(time_dependent_logits[:, :1, :]), 1.0 - cdf[:, :-1, :]], dim=1)
        
        # Add numerical stability
        time_dependent_logits_stable = torch.clamp(time_dependent_logits, min=eps)
        integrated_logits_stable = torch.clamp(integrated_logits, min=eps)
        
        # Verify input shapes match expectations
        assert batch["non_numerical_is_event"].shape == time_dependent_logits.shape, \
            f"Shape mismatch: time_dependent_logits {time_dependent_logits.shape} vs is_event {batch['non_numerical_is_event'].shape}"
        
        # Validate exactly one bin per prediction-task combination
        # labels_sum = torch.sum(batch["non_numerical_is_event"], dim=1)
        # assert torch.all(labels_sum == 1), f"Expected exactly 1 True bin per prediction-task combination"
        
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
            L_non = -torch.sum(loss_values) / (num_marked_bins)  # normalized by log B_t
            total_loss += -torch.sum(loss_values)
            total_count += num_marked_bins
        else:
            L_non = torch.tensor(0.0, device=features.device)

        
        if return_logits:
            result["time_dependent_logits"] = time_dependent_logits
        
        # ========== NUMERICAL CODES (New Matrix-wise Logic) ==========  
        if "numerical_time_event_bin" in batch:
            # Get dimensions from batch data 
            # print(batch.keys())
            time_event_in_bin = batch["numerical_time_event_bin"]  # [pred_points, time_bins, numerical_tasks]
            time_censor_in_bin = batch["numerical_time_censor_bin"]  # [pred_points, time_bins, numerical_tasks]
            value_event_in_bin = batch["numerical_value_event_bin"]  # [pred_points, value_bins, numerical_tasks]
            # numerical_is_censored = batch["numerical_is_censored"]  # [pred_points, numerical_tasks]
            
            batch_size, _, num_numerical_tasks = time_event_in_bin.shape
            # Time-value features: [pred_points, time_bins, value_bins, final_layer_size]
            numerical_features = self.final_layer(features).reshape(
                batch_size, self.num_time_bins*self.num_value_bins, self.final_layer_size
            )
            
            # Numerical task logits: [pred_points, time_bins*value_bins, numerical_tasks]
            numerical_task_logits = self.numerical_task_layer(self.numerical_norm(numerical_features))

            # Apply value_valid_mask to mask out invalid value bins before softmax
            # value_valid_mask: [numerical_tasks, value_bins] -> need to expand to [time_bins*value_bins, numerical_tasks]
            device = numerical_task_logits.device
            value_valid_mask = torch.tensor(batch["value_valid_mask"],device=device,dtype=torch.bool)

            if value_valid_mask.shape[1] - value_event_in_bin.shape[1] == 1:
                value_valid_mask = value_valid_mask[:,1:]
            
            # Expand valid mask to match value_event_in_bin shape: [1, value_bins, numerical_tasks]
            valid_mask_expanded = value_valid_mask.T.unsqueeze(0) 
            # Check that events only occur in valid bins: invalid events = events & ~valid_mask
            invalid_events = value_event_in_bin & (~valid_mask_expanded)  # [pred_points, value_bins, numerical_tasks]
            assert not torch.any(invalid_events), f"Events in bin found in invalid value bins {invalid_events}"



            mask_tv = valid_mask_expanded.expand(self.num_time_bins,-1,-1) #[num_time_bins,value_bins, numerical_tasks]
            expanded_mask = mask_tv.reshape(1,self.num_time_bins * self.num_value_bins, num_numerical_tasks)

            # Use large negative value instead of -inf
            mask_value = -1e10  # This effectively zeros out in softmax but avoids -inf
            numerical_task_logits = numerical_task_logits.masked_fill(~expanded_mask, mask_value)

            numerical_probs_flat = self.softmax(numerical_task_logits)

            # torch.logsumexp()
            # torch.exp(torch.logsum(numerical_log_probs_flat))

            # numerical_log_probs = numerical_log_probs_flat.reshape(batch_size, self.num_time_bins, self.num_value_bins, num_numerical_tasks)
            numerical_probs = numerical_probs_flat.reshape(batch_size, self.num_time_bins, self.num_value_bins, num_numerical_tasks)

            numerical_probs = numerical_probs.clamp_min(eps)

            # # assert check
            expanded_valid_mask_assert = mask_tv.unsqueeze(0).expand(batch_size,-1,-1,-1)
            numerical_probs = numerical_probs*expanded_valid_mask_assert

            assert torch.all(numerical_probs[~expanded_valid_mask_assert] <= eps), \
            "Non-zero probabilities found in invalid bins."
            assert torch.all(numerical_probs[expanded_valid_mask_assert] >= eps), \
            "zero probabilities found in valid bins."
            
            # === MATRIX-WISE LOSS COMPUTATION (NO NESTED LOOPS) ===
            
            # Case 1: Censored cases - sum over value bins, only predict time
            # Use time_censor_in_bin directly for censored events
            numerical_loss = torch.tensor(0.0, device=features.device)
            total_num_count = 0
            # event_count = torch.tensor(0.0,device=features.device)
            # censored_count = torch.tensor(0.0,device=features.device)
            L_num = torch.tensor(0.0, device=features.device)
            L_num_event = torch.tensor(0.0, device=features.device)
            L_num_cens  = torch.tensor(0.0, device=features.device)


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
                # if censored_count > 0:
                L_num_cens = -torch.sum(censored_loss_values) 
                numerical_loss += L_num_cens
                total_num_count += censored_count

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

                    L_num_event = -torch.sum(event_loss_values)
                    numerical_loss += L_num_event
                    total_num_count += event_count
            
            # ne  = event_count.float()
            # nc  = censored_count.float()
            # den = ne + nc
            if total_num_count > 0:
                L_num = numerical_loss/total_num_count
                total_count += total_num_count
                total_loss += numerical_loss
            else:
                L_num = torch.tensor(0.0, device=features.device)

            # if den > 0:
            #     w_evt = ne / den
            #     w_cen = nc / den
            #     L_num = w_evt * L_num_event + w_cen * L_num_cens
            # else:
            #     L_num = torch.tensor(0.0, device=features.device)

            if return_logits:
                result["loss_components"] = {
                    "L_non_num":       L_non.detach(),
                    "L_num_event":   L_num_event.detach(),
                    "L_num_censor":  L_num_cens.detach(),
                }
                result["value_dependent_logits"] = numerical_probs
        else:
            print("No numerical codes found in batch")

        loss = total_loss / total_count

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
    config_class = femr.models.config.FEMRModelConfig

    def __init__(self, config: femr.models.config.FEMRModelConfig, **kwargs):
        if "task_config" in kwargs:
            config.task_config = kwargs["task_config"]
        super().__init__(config, **kwargs)
        self.linear_interpolation = kwargs.pop("linear_interpolation", False)
        self.loss_type = kwargs.pop("loss_type",False)


        # Choose backbone
        if self.config.model_type == "mamba":
            # self.config.mamba_config should be model_config in pretrain_motor.py
            assert self.config.mamba_config is not None
            self.backbone = FEMRMamba(self.config.mamba_config)
            self.hidden_size = self.config.mamba_config.hidden_size
            
            # if self.hidden_size is None:
            #     self.hidden_size = self.backbone.out_norm.weight.shape[0]
        elif self.config.model_type == "transformer":
            assert self.config.transformer_config is not None
            self.backbone = FEMRTransformer(self.config.transformer_config)
            self.hidden_size = self.config.transformer_config.hidden_size
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")
        assert self.hidden_size is not None
        print(f"the hidden size is {self.hidden_size}")

        # Task head (shared across architectures)
        if self.config.task_config is not None:
            # print(f"task_config is {self.config.task_config}")
            self.task_model = self.create_task_head()

    def create_task_head(self) -> nn.Module:
        # hidden_size = self.config.transformer_config.hidden_size
        task_type = self.config.task_config.task_type
        task_kwargs = self.config.task_config.task_kwargs
        # if task_type == "clmbr":
        #     return CLMBRTaskHead(hidden_size, **task_kwargs)
        # elif task_type == "labeled_subjects":
        #     return LabeledSubjectTaskHead(hidden_size, **task_kwargs)
        # elif task_type == "motor":
        #     return MOTORTaskHead(hidden_size, **task_kwargs)
        # else:
        #     raise RuntimeError("Could not determine head for task " + task_type)
        print(f"create loss type is {self.loss_type}")
        if self.loss_type == "clmbr":
            return CLMBRTaskHead(self.hidden_size, **task_kwargs)
        elif self.loss_type == "labeled_subjects":
            return LabeledSubjectTaskHead(self.hidden_size, **task_kwargs)
        elif self.loss_type == "motor":
            return MOTORTaskHead(self.hidden_size, **task_kwargs)
        elif self.loss_type == "tpp":
            return TPPTaskHead(self.hidden_size, **task_kwargs)
        elif self.loss_type == "mtpp":
            return MTPPTaskHead(self.hidden_size, **task_kwargs)
        elif self.loss_type == "mtpp_shared":
            return MTPPSharedTaskHead(self.hidden_size, **task_kwargs)
        else:
            raise RuntimeError("Could not determine head for task " + self.loss_type)

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
        features = self.backbone(batch["transformer"], s)
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
        loss_type: str = None,
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
    task = femr.models.tasks.motor.LabeledSubjectTask(labels, observation_window)

    print(f"Loading model from {model_path}")
    print(f"use_linear_interpolation: {use_linear_interpolation}")
    print(f"loss type is {loss_type}")
    # Use the new from_pretrained method that supports linear_interpolation
    model = femr.models.architecture.embedding.FEMRModel.from_pretrained(
        model_path, 
        task_config=task.get_task_config(),
        loss_type=loss_type,
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
