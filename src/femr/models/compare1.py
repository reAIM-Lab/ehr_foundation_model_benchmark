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
        if self.config.hidden_size is not None:
            hf_config.d_model = self.config.hidden_size
        if self.config.n_layers is not None:
            if hasattr(hf_config, "n_layer"):
                hf_config.n_layer = self.config.n_layers
            if hasattr(hf_config, "num_hidden_layers"):
                hf_config.num_hidden_layers = self.config.n_layers
        if self.config.d_state is not None and hasattr(hf_config, "d_state"):
            hf_config.d_state = self.config.d_state
        if hasattr(hf_config, "vocab_size"):
            hf_config.vocab_size = self.config.vocab_size

        # Freeform overrides
        for key, val in (self.config.config_kwargs or {}).items():
            if hasattr(hf_config, key):
                setattr(hf_config, key, val)

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
        lengths = batch["subject_lengths"]  # (S,)
        seqs = torch.split(x, lengths.tolist())  # List[(L_i, D)]
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # (S, L_max, D)
        attn_mask = torch.arange(padded.size(1), device=padded.device)[None, :].expand(len(seqs), -1) < lengths[:, None]

        # Forward through backbone using inputs_embeds; attention_mask may be ignored by some Mamba impls
        try:
            out = self.backbone(inputs_embeds=padded, attention_mask=attn_mask)
        except TypeError:
            out = self.backbone(inputs_embeds=padded)
        except Exception:
            out = self.backbone(padded)

        last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

        # Unpad back to flattened order
        chunks: List[torch.Tensor] = [last_hidden[i, : lengths[i], :] for i in range(len(seqs))]
        final = torch.cat(chunks, dim=0)  # (T, D)
        final = self.out_norm(final)
        return final
