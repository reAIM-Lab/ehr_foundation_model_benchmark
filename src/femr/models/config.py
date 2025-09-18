from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import transformers


class FEMRTransformerConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 32768,
        is_hierarchical: bool = False,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        n_heads: int = 12,
        n_layers: int = 6,
        attention_width: int = 496,
        use_normed_ages: bool = False,
        use_bias: bool = True,
        hidden_act: str = "gelu",
        **kwargs,
    ) -> None:
        """Defined a configuration for a FEMR Transformer.

        Arguments:
            vocab_size: The number of tokens in the vocabulary
            is_hierarchical: Whether to use a hierarchical vocabulary. See FEMRTokenizer for more information
            hidden_size: The internal representation size
            intermediate_size: The size of the FFN in the transformer layers
            n_heads: The number of attention heads
            n_layers: The number of transformer encoder layers
            attention_width: FEMR by default uses a local attention transformer with a width defined here
            use_normed_ages: Whether or not to provide normalized ages as a feature to the model
            use_bias: Whether or not to use bias terms in the transformer layers
            hidden_act: The type of activation function to use in the transformer
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.is_hierarchical = is_hierarchical

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_width = attention_width

        self.use_normed_ages = use_normed_ages

        self.use_bias = use_bias
        self.hidden_act = hidden_act

class FEMRMambaConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 32768,
        is_hierarchical: bool = False,
        # Allow None to use HF defaults when building backbone
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        n_layers: Optional[int] = None,
        use_normed_ages: bool = False,
        use_bias: bool = True,
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
        self.intermediate_size = intermediate_size  # Keep for compatibility; may be unused by backbone
        self.n_layers = n_layers
        self.use_normed_ages = use_normed_ages
        self.use_bias = use_bias

        # Mamba-specific parameters
        self.hf_name = hf_name
        self.d_state = d_state
        # self.d_conv = d_conv
        # self.expand_factor = expand_factor
        self.config_kwargs = config_kwargs or {}

class FEMRTaskConfig(transformers.PretrainedConfig):
    def __init__(self, task_type: str = "", task_kwargs: Mapping[str, Any] = {}, **kwargs):
        """A generic FEMR task definition. This holds state used for initalizing a tasks.py class.

        Task.get_task_config returns the task type and kwargs used to initialize this.

        Arguments:
            task_type: The name of the task.
            task_kwargs: Arbitrary arguments used to store state for that task.
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.task_kwargs = task_kwargs


class FEMRModelConfig(transformers.PretrainedConfig):
    """A model config composed of an architecture config (transformer or mamba) and an optional task config."""

    def __init__(
        self,
        transformer_config: Optional[Dict[str, Any]] = None,
        mamba_config: Optional[Dict[str, Any]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        model_type: Optional[str] = None,  # "transformer" or "mamba"
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Determine model_type if not provided
        if model_type is None:
            if mamba_config is not None:
                model_type = "mamba"
            else:
                model_type = "transformer"
        self.model_type = model_type

        # Initialize sub-configs
        self.transformer_config: Optional[FEMRTransformerConfig] = None
        self.mamba_config: Optional[FEMRMambaConfig] = None

        if transformer_config is not None:
            self.transformer_config = FEMRTransformerConfig(**transformer_config)
        if mamba_config is not None:
            self.mamba_config = FEMRMambaConfig(**mamba_config)

        # Task config
        self.task_config: Optional[FEMRTaskConfig]
        if task_config is not None:
            self.task_config = FEMRTaskConfig(**task_config)
        else:
            self.task_config = None

    @classmethod
    def from_task_configs(
        cls,
        model_config: "FEMRTransformerConfig | FEMRMambaConfig",
        task_config: Optional[FEMRTaskConfig],
    ) -> "FEMRModelConfig":
        """Combine an architecture configuration and a task configuration into a model configuration.

        Accepts either a FEMRTransformerConfig or a FEMRMambaConfig for the architecture.
        """
        task_config_dict = task_config.to_dict() if task_config is not None else None

        # Detect which architecture config is provided
        if isinstance(model_config, FEMRTransformerConfig):
            return cls(
                transformer_config=model_config.to_dict(),
                mamba_config=None,
                task_config=task_config_dict,
                model_type="transformer",
            )
        elif isinstance(model_config, FEMRMambaConfig):
            return cls(
                transformer_config=None,
                mamba_config=model_config.to_dict(),
                task_config=task_config_dict,
                model_type="mamba",
            )
        else:
            raise TypeError("model_config must be FEMRTransformerConfig or FEMRMambaConfig")
