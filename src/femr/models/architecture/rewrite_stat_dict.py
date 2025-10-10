from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import torch
from transformers.modeling_utils import load_state_dict, PreTrainedModel
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    renamed_keys: List[Tuple[str, str]]
    dropped_keys: List[str]
    missing_keys: List[str]
    unexpected_keys: List[str]

    def log(self) -> None:
        if self.renamed_keys:
            logger.info(
                "Renamed %d keys (first 5 shown): %s",
                len(self.renamed_keys),
                ", ".join(f"{src}->{dst}" for src, dst in self.renamed_keys[:5]),
            )
        if self.dropped_keys:
            logger.info(
                "Dropped %d task-head keys (first 5 shown): %s",
                len(self.dropped_keys),
                ", ".join(self.dropped_keys[:5]),
            )
        if self.missing_keys:
            logger.info(
                "load_state_dict reported %d missing keys (first 5 shown): %s",
                len(self.missing_keys),
                ", ".join(self.missing_keys[:5]),
            )
        if self.unexpected_keys:
            logger.info(
                "load_state_dict reported %d unexpected keys (first 5 shown): %s",
                len(self.unexpected_keys),
                ", ".join(self.unexpected_keys[:5]),
            )


def _load_raw_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    for filename in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME):
        candidate = os.path.join(checkpoint_dir, filename)
        if os.path.exists(candidate):
            return load_state_dict(candidate)
    raise FileNotFoundError(
        f"Could not find {SAFE_WEIGHTS_NAME} or {WEIGHTS_NAME} under {checkpoint_dir}"
    )


def _rewrite_legacy_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, str]]]:
    renamed: List[Tuple[str, str]] = []
    adapted: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("transformer."):
            new_key = "backbone." + key[len("transformer."):]
            adapted[new_key] = value
            renamed.append((key, new_key))
        else:
            adapted[key] = value

    return adapted, renamed


def _drop_task_head(
    state_dict: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    dropped = [k for k in state_dict if k.startswith("task_model.")]
    if not dropped:
        return state_dict, []
    filtered = {
        key: value for key, value in state_dict.items() if not key.startswith("task_model.")
    }
    return filtered, dropped


def load_femr_model_with_compat(
    checkpoint_dir: str,
    *,
    model_cls: Type[PreTrainedModel],
    drop_task_head: bool = True,
    map_location: str | torch.device = "cpu",
    **model_kwargs: Any,
) -> Tuple[PreTrainedModel, LoadReport]:
    """
    Load a FEMRModel checkpoint while remapping legacy parameter names.

    Args:
        checkpoint_dir: path containing config + weights.
        model_cls: FEMRModel class (pass the class to avoid circular imports).
        drop_task_head: remove task head parameters from the checkpoint.
        map_location: passed to torch.load when reading weights (default CPU).
        model_kwargs: forwarded to model constructor (e.g., task_config, loss_type).

    Returns:
        (model, LoadReport)
    """
    raw_state = _load_raw_state_dict(checkpoint_dir)
    state_dict, renamed = _rewrite_legacy_keys(raw_state)
    dropped: List[str] = []

    if drop_task_head:
        state_dict, dropped = _drop_task_head(state_dict)

    config = model_cls.config_class.from_pretrained(checkpoint_dir)
    config._name_or_path = checkpoint_dir  # preserve provenance

    # Create model instance from config using constructor
    model = model_cls(config, **model_kwargs)
    state_dict = {k: v.to(map_location) for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    report = LoadReport(
        renamed_keys=renamed,
        dropped_keys=dropped,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
    report.log()
    return model, report