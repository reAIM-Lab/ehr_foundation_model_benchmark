class DescTokenizer(BaseTokenizer):
    """Converts codes => textual descriptions, then tokenizes using a normal text tokenizer (e.g. BERT)
    """
    def __init__(self, 
                 path_to_tokenizer_config: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        assert metadata is not None, f"ERROR - `metadata` must be provided, but got {metadata}"
        assert 'desc_emb_tokenizer' in metadata, f"ERROR - `metadata` must contain a 'desc_emb_tokenizer' key, but got {metadata}"
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)
        
        # Set metadata
        self.metadata: Dict[str, Any] = {} if metadata is None else dict(metadata)
        self.metadata['cls'] = 'DescTokenizer'

        # Load underlying textual tokenizer
        self.desc_emb_tokenizer: str = metadata['desc_emb_tokenizer']
        self.tokenizer = AutoTokenizer.from_pretrained(self.desc_emb_tokenizer)
        self.event_separator: str = ' ' # character that gets inserted between Events when transformed into their textual descriptions
        
        # Metadata
        metadata = {} if metadata is None else metadata
        self.excluded_vocabs: Optional[Set[str]] = { x.lower() for x in metadata.get('excluded_vocabs', {}) } if metadata.get('excluded_vocabs', {}) else None # type: ignore
        self.min_code_occurrence_count: Optional[int] = metadata.get('min_code_occurrence_count', None)
        self.keep_n_max_occurrence_codes: Optional[int] = metadata.get('keep_n_max_occurrence_codes', None)
        
        # Apply filtering
        self.tokenizer_config, self.excluded_tokens = filter_tokenizer_config(self.tokenizer_config, 
                                                                              self.excluded_vocabs, 
                                                                              self.min_code_occurrence_count,
                                                                              self.keep_n_max_occurrence_codes)
        
        # Preprocess tokenizer config for quick access
        self.code_2_desc: Dict[str, str] = {}
        # initialize non special tokens list
        self.non_special_tokens: List[str] = []
        for entry in self.tokenizer_config:
            if entry.description is not None:
                self.code_2_desc[entry.code] = entry.description
                self.non_special_tokens.append(entry.description)

        # Define special tokens 
        self.special_tokens: List[str] = [
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
            self.tokenizer.cls_token,
            self.tokenizer.mask_token,
        ]
        
        super().__init__(
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            unk_token=self.tokenizer.unk_token,
            sep_token=self.tokenizer.sep_token,
            pad_token=self.tokenizer.pad_token,
            cls_token=self.tokenizer.cls_token,
            mask_token=self.tokenizer.mask_token
        )
    
    def convert_event_to_token(self, e: Event, **kwargs) -> Optional[str]:
        if e.code not in self.code_2_desc:
            return None
        return self.code_2_desc[e.code]

    def __call__(self, 
                 batch_of_events: Union[List[Event], List[List[Event]]],
                 is_truncation_random: bool = False,
                 seed: int = 1,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
            Tokenize a batch of patient timelines, where each timeline is a list of event codes.
            We add the ability to truncate seqs at random time points.
            
            Expects as input a list of Events
        """
        if not isinstance(batch_of_events[0], list):
            # List[Event] => List[List[Event]]
            batch_of_events = [ batch_of_events ] # type: ignore
        
        # First, convert all Events => ProtoTokens
        batch: List[List[str]] = [ self.convert_events_to_tokens(x) for x in batch_of_events ] 
        
        # Second, add special tokens (if applicable)
        if kwargs.get("add_special_tokens", False):
            batch = [ [ self.cls_token, self.bos_token ] + x + [ self.eos_token ] for x in batch ]

        # Concatenate all strings together for tokenization by traditional HF tokenizer
        # List[List[str]] => List[str]
        #batch = [ self.event_separator.join(x) for x in batch ] # type: ignore
        # Ensure proper filtering of None values
        batch = [self.event_separator.join(filter(None, x)) for x in batch]

        if is_truncation_random:
            max_length: int = kwargs.get("max_length")
            if not max_length:
                raise ValueError(f"If you specify `is_truncation_random`, then you must also provide a non-None value for `max_length`")

            # Tokenize without truncation
            if 'max_length' in kwargs:
                del kwargs['max_length']
            if 'truncation' in kwargs:
                del kwargs['truncation']
            tokenized_batch: Dict[str, torch.Tensor] = self.tokenizer.__call__(batch, **kwargs, truncation=None)

            # Truncate at random positions
            random.seed(seed)
            start_idxs: List[int] = []
            for timeline in tokenized_batch['input_ids']:
                length: int = (timeline != self.pad_token_id).sum() # count of non-PAD tokens
                if length > max_length:
                    # Calculate a random start index
                    start_index: int = random.randint(0, length - max_length)
                    start_idxs.append(start_index)
                else:
                    start_idxs.append(0)
                    
            for key in tokenized_batch.keys():
                truncated_batch: List[List[int]] = []
                for idx, timeline in enumerate(tokenized_batch[key]):
                    new_timeline = timeline[start_idxs[idx]:start_idxs[idx] + max_length]
                    assert new_timeline.shape[0] <= max_length, f"Error in truncating by random positions: new_timeline.shape = {new_timeline.shape[0]} !<= max_length={max_length}"
                    truncated_batch.append(new_timeline)
                if kwargs.get('return_tensors') == 'pt':
                    tokenized_batch[key] = torch.stack(truncated_batch, dim=0)
                else:
                    tokenized_batch[key] = truncated_batch
        else:
            tokenized_batch: Dict[str, torch.Tensor] = self.tokenizer.__call__(batch, **kwargs)

        return tokenized_batch

    """Mandatory overwrites of base class"""
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.get_vocab())

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def _tokenize(self, text: str, **kwargs):
        return self.tokenizer._tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer._convert_id_to_token(index)