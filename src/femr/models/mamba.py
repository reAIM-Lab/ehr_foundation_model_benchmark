from __future__ import annotations

import collections
from typing import Any, Dict, List, Mapping, Optional

import meds
import meds_reader
import numpy as np
import torch
import transformers
from torch import nn
from tqdm import tqdm

import femr.models.config
import femr.models.processor
import femr.models.rmsnorm
import femr.models.tasks
import femr.models.tokenizer


class FEMRMambaConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 32768,
        is_hierarchical: bool = False,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        n_layers: int = 24,
        use_normed_ages: bool = False,
        use_bias: bool = True,
        hidden_act: str = "silu",
        # Mamba-specific parameters
        hf_name: str = "state-spaces/mamba-130m-hf",
        d_state: int = 16,
        # d_conv: int = 4,
        # expand_factor: int = 2,
        config_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Configuration for a FEMR Mamba model.

        Arguments:
            vocab_size: The number of tokens in the vocabulary
            is_hierarchical: Whether to use a hierarchical vocabulary
            hidden_size: The internal representation size (d_model in Mamba)
            intermediate_size: Not used in Mamba but kept for compatibility
            n_layers: The number of Mamba layers
            use_normed_ages: Whether to provide normalized ages as a feature
            use_bias: Whether to use bias terms
            hidden_act: Unused for Mamba (kept for compatibility)
            hf_name: HuggingFace model name for Mamba configuration
            d_state: State dimension for Mamba
            d_conv: Convolution dimension for Mamba
            expand_factor: Expansion factor for Mamba
            config_kwargs: Additional configuration overrides
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.is_hierarchical = is_hierarchical
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size  # Keep for compatibility
        self.n_layers = n_layers
        self.use_normed_ages = use_normed_ages
        self.use_bias = use_bias
        # hidden_act is intentionally unused in Mamba; kept for API compatibility

        # Mamba-specific parameters
        self.hf_name = hf_name
        self.d_state = d_state
        # self.d_conv = d_conv
        # self.expand_factor = expand_factor
        self.config_kwargs = config_kwargs or {}


class FEMRMamba(nn.Module):
    def __init__(self, config: FEMRMambaConfig):
        super().__init__()
        self.config = config

        self.in_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        self.out_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)

        # Embedding layers (same as transformer)
        if not self.config.is_hierarchical:
            self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        else:
            self.embed_bag = nn.EmbeddingBag(
                num_embeddings=self.config.vocab_size,
                embedding_dim=self.config.hidden_size,
                mode="sum",
                include_last_offset=True,
            )

        # Configure HuggingFace Mamba model
        # Load base HF config and allow remote code (required by Mamba HF refs)
        model_config = transformers.AutoConfig.from_pretrained(
            self.config.hf_name,
            trust_remote_code=True,
        )
        
        # Override configuration parameters
        model_config.vocab_size = self.config.vocab_size
        model_config.d_model = self.config.hidden_size
        model_config.n_layer = self.config.n_layers
        model_config.d_state = self.config.d_state
        # model_config.d_conv = self.config.d_conv
        # model_config.expand = self.config.expand_factor
        
        # Apply additional config overrides
        for key, val in self.config.config_kwargs.items():
            assert hasattr(model_config, key), (
                f"Config for HF model {self.config.hf_name if hasattr(self.config, 'hf_name') else ''} "
                f"does not have attribute {key}"
            )
            setattr(model_config, key, val)
        
        self.model_config = model_config
        self.hidden_size = model_config.d_model

        # Initialize Mamba model without pre-trained weights
        self.mamba_model = transformers.AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=True,
        )
        print(f"mamba config is {model_config}")
        
        # We only need the backbone, not the LM head
        if hasattr(self.mamba_model, 'backbone'):
            self.mamba_backbone = self.mamba_model.backbone
        elif hasattr(self.mamba_model, 'model'):
            self.mamba_backbone = self.mamba_model.model
        else:
            # For some Mamba models, the backbone is the model itself
            self.mamba_backbone = self.mamba_model

    def forward(self, batch, s):
        # Note: 's' parameter kept for interface compatibility with transformer
        # Input embedding (same logic as transformer)
        if not self.config.is_hierarchical:
            x = self.embed(batch["tokens"])
        else:
            x = self.embed_bag(batch["hierarchical_tokens"], batch["token_indices"], batch["hierarchical_weights"])

        # Input normalization
        x = self.in_norm(x)
        
        # Handle time data if configured (same as transformer)
        if self.config.use_normed_ages:
            time_data = batch["time_data"]
            all_time = torch.concatenate((time_data, time_data ** 2), axis=-1)
            x[:, -all_time.shape[1]:] = all_time.to(dtype=x.dtype)

        # Process each subject independently to avoid cross-subject leakage
        lengths = batch["subject_lengths"].tolist()
        assert sum(lengths) == x.shape[0], (
            f"Sum of subject_lengths {sum(lengths)} != token len {x.shape[0]}"
        )
        outputs = []
        start = 0
        for L in lengths:
            end = start + L
            seg = x[start:end, :]
            seg_batched = seg.unsqueeze(0)  # (1, L, D)
            # Run through the HF Mamba backbone using inputs_embeds
            if hasattr(self.mamba_backbone, 'forward'):
                try:
                    seg_out = self.mamba_backbone(inputs_embeds=seg_batched)
                except Exception:
                    seg_out = self.mamba_backbone(seg_batched)
            else:
                seg_out = self.mamba_backbone(seg_batched)

            if hasattr(seg_out, 'last_hidden_state'):
                seg_hidden = seg_out.last_hidden_state  # (1, L, D)
            elif isinstance(seg_out, tuple):
                seg_hidden = seg_out[0]
            else:
                seg_hidden = seg_out

            outputs.append(seg_hidden.squeeze(0))  # (L, D)
            start = end

        final = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        
        # Output normalization
        final = self.out_norm(final)

        return final


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


class FEMRMambaModel(transformers.PreTrainedModel):
    config_class = femr.models.config.FEMRModelConfig
    
    def __init__(self, config: femr.models.config.FEMRModelConfig, **kwargs):
        # Extract linear_interpolation from kwargs, default to False
        self.linear_interpolation = kwargs.pop('linear_interpolation', False)
        
        # Extract Mamba-specific parameters from kwargs
        mamba_model_name = kwargs.pop('mamba_model_name', 'state-spaces/mamba-130m-hf')
        d_state = kwargs.pop('d_state', 16)
        # mamba_config_overrides = kwargs.pop('mamba_config_overrides', {})
        
        # Allow the task config to be overwritten
        if "task_config" in kwargs:
            config.task_config = kwargs["task_config"]

        super().__init__(config, **kwargs)

        # Replace transformer with mamba
        # Convert transformer config to mamba config with additional parameters
        mamba_config = FEMRMambaConfig(
            vocab_size=config.transformer_config.vocab_size,
            is_hierarchical=config.transformer_config.is_hierarchical,
            hidden_size=config.transformer_config.hidden_size,
            intermediate_size=config.transformer_config.intermediate_size,
            n_layers=config.transformer_config.n_layers,
            use_normed_ages=config.transformer_config.use_normed_ages,
            use_bias=config.transformer_config.use_bias,
            hidden_act=config.transformer_config.hidden_act,
            # Mamba-specific parameters
            hf_name=mamba_model_name,
            d_state=d_state,
            # config_kwargs=mamba_config_overrides,
        )
        
        self.mamba = FEMRMamba(mamba_config)
        if self.config.task_config is not None:
            self.task_model = self.create_task_head()

    def create_task_head(self) -> nn.Module:
        from femr.models.transformer import CLMBRTaskHead, LabeledSubjectTaskHead, MOTORTaskHead
        
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
        # Same logic as FEMRModel but use mamba instead of transformer
        assert return_loss

        batch = remove_first_dimension(batch)
        input_device = batch['subject_ids'].device
        s = torch.zeros_like(batch['subject_ids'], device=input_device)
        s[1:] = batch['subject_ids'][1:] != batch['subject_ids'][:-1]
        s = torch.cumsum(s, dim=0).type(torch.uint8)

        # Use mamba instead of transformer
        features = self.mamba(batch["transformer"], s)
        
        if "task" in batch and self.config.task_config is not None:
            features = features.reshape(-1, features.shape[-1])
            features = features[batch["transformer"]["label_indices"], :]
            
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
        total_flops=None,
        use_linear_interpolation: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute features using FEMRMamba model.
    
    Same interface as transformer.compute_features but uses Mamba architecture.
    Note: total_flops parameter currently unused but kept for interface compatibility.
    """
    task = femr.models.tasks.LabeledSubjectTask(labels, observation_window)

    print(f"Loading Mamba model from {model_path}")
    print(f"use_linear_interpolation: {use_linear_interpolation}")
    
    model = FEMRMambaModel.from_pretrained(
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

    print(f"The maximum context length is {tokens_per_batch/min_subjects_per_batch}, {min_subjects_per_batch} subjects and {tokens_per_batch} tokens per batch")
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
