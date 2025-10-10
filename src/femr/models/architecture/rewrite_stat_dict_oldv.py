from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import torch
from transformers.modeling_utils import load_state_dict
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    renamed_keys: List[Tuple[str, str]]
    dropped_keys: List[str]

    def log(self) -> None:
        if self.renamed_keys:
            logger.info(
                "Renamed %d keys (first 5): %s",
                len(self.renamed_keys),
                ", ".join(f"{src}->{dst}" for src, dst in self.renamed_keys[:5]),
            )
        if self.dropped_keys:
            logger.info(
                "Dropped %d task-head keys (first 5): %s",
                len(self.dropped_keys),
                ", ".join(self.dropped_keys[:5]),
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
) -> Tuple[Dict[str, torch.Tensor], LoadReport]:
    renamed: List[Tuple[str, str]] = []
    dropped: List[str] = []
    adapted: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("transformer."):
            new_key = "backbone." + key[len("transformer."):]
            adapted[new_key] = value
            renamed.append((key, new_key))
        elif key.startswith("task_model."):
            dropped.append(key)
        else:
            adapted[key] = value

    return adapted, LoadReport(renamed_keys=renamed, dropped_keys=dropped)


def load_femr_model_with_compat(
    checkpoint_dir: str,
    *,
    model_cls: Type[Any],
#   drop_task_head: bool = False,
    **model_kwargs: Any,
) -> Tuple[Any, LoadReport]:
    """Load a FEMRModel while adapting legacy checkpoint key layouts."""
    if model_cls is None:
        raise ValueError("model_cls must be provided to avoid circular imports")

    raw_state_dict = _load_raw_state_dict(checkpoint_dir)
    state_dict = raw_state_dict
    report = LoadReport(renamed_keys=[], dropped_keys=[])

    if any(key.startswith("transformer.") for key in raw_state_dict):
        state_dict, report = _rewrite_legacy_keys(raw_state_dict)

#   if drop_task_head:
#       filtered = {k: v for k, v in state_dict.items() if not k.startswith("task_model.")}
#       report.dropped_keys.extend(
#           [k for k in state_dict.keys() if k.startswith("task_model.")]
#       )
#       state_dict = filtered

    model = model_cls.from_pretrained(
        checkpoint_dir,
        state_dict=state_dict,
        ignore_mismatched_sizes=True,
        **model_kwargs,
    )
    report.log()
    return model, report