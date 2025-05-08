class CookbookTokenizer(BaseCodeTokenizer):
    """
        Settings:
            is_remap_numerical_codes_to_quantiles: bool
                - If TRUE, then remap numericals to buckets based on quantile of value
            excluded_vocabs: Optional[List[str]]
                - List of vocabs to exclude from the tokenizer. Determined by the first part of the code before the '/' (e.g. "STANFORD_OBS" in "STANFORD_OBS/1234")
            min_code_occurrence_count: Optional[int]
                - Only keep tokens with >= `min_code_occurrence_count` total occurrences in our dataset
    """
    def __init__(self, 
                 path_to_tokenizer_config: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)
        
        # Set metadata
        self.metadata: Dict[str, Any] = {} if metadata is None else dict(metadata)
        self.metadata['cls'] = 'CookbookTokenizer'

        # Fetches ATT metadata - By default, it'll be done in CEHR-GPT style
        self.is_add_visit_start: bool = metadata.get('is_add_visit_start', True)
        self.is_add_visit_end: bool = metadata.get('is_add_visit_end', True)
        self.is_add_day_att: bool = metadata.get('is_add_day_att', True)
        self.is_add_day_week_month_att: bool = metadata.get('is_add_day_week_month_att', False)

        # Pop the flags from metadata after initializing them to avoid recreating the config
        for key in ['is_add_visit_start', 'is_add_visit_end', 'is_add_day_att', 'is_add_day_week_month_att']:
            if key in self.metadata:
                self.metadata.pop(key)
        
        #  # Initialize special tokens
        # # TODO -- prepend all these attributes with 'token_' for readibility
        # self.visit_start = "[VISIT START]"
        # self.visit_end = "[VISIT END]"
        # self.day_atts_cehr_gpt = [f"[DAY {i}]" for i in range(1, 1081)]
        # self.long_att_cehr_gpt = "[LONG TERM]"
        # self.day_atts_cehr_bert = [f"[DAY {i}]" for i in range(1, 7)]
        # self.week_atts = [f"[WEEK {i}]" for i in range(1, 4)]
        # self.month_atts = [f"[MONTH {i}]" for i in range(1, 12)]
        # self.long_att_cehr_bert = "[LONG TERM]"

        # # Add special tokens to the vocabulary
        # self.special_tokens.extend(self.day_atts_cehr_gpt)
        # self.special_tokens.append(self.long_att_cehr_gpt)
        # self.special_tokens.extend(self.day_atts_cehr_bert)
        # self.special_tokens.extend(self.week_atts)
        # self.special_tokens.extend(self.month_atts)
        # self.special_tokens.append(self.long_att_cehr_bert)
        # self.special_tokens.extend([self.visit_start, self.visit_end])

        # Metadata
        print(f"metadata is {metadata}")
        self.is_remap_numerical_codes_to_quantiles: bool = metadata.get('is_remap_numerical_codes_to_quantiles', False)
        self.excluded_vocabs: Optional[Set[str]] = { x.lower() for x in metadata.get('excluded_vocabs', {}) } if metadata.get('excluded_vocabs', {}) else None # type: ignore
        self.min_code_occurrence_count: Optional[int] = metadata.get('min_code_occurrence_count', None)
        self.keep_n_max_occurrence_codes: Optional[int] = metadata.get('keep_n_max_occurrence_codes', None)

        # Apply filtering
        # print(f"path to token is {path_to_tokenizer_config}")
        print(f" occur count is {self.min_code_occurrence_count}")
        self.tokenizer_config, self.excluded_tokens = filter_tokenizer_config(self.tokenizer_config, 
                                                                              self.excluded_vocabs, 
                                                                              self.min_code_occurrence_count,
                                                                              self.keep_n_max_occurrence_codes)
        # Tokens
        self.code_2_token = {} # [key] = token; [val] = { 'type' : str, 'tokenization' : dict, 'token' : str }
        self.non_special_tokens: List[str] = []
        
        # Preprocess tokenizer config for quick access
        for entry in self.tokenizer_config: # NOTE: Takes ~10 seconds for 1.5M tokens
            if entry.code not in self.code_2_token: self.code_2_token[entry.code] = {}
            if entry.type not in self.code_2_token[entry.code]: self.code_2_token[entry.code][entry.type] = []
            self.code_2_token[entry.code][entry.type].append({
                'tokenization': entry.tokenization,
                'token' : entry.to_token(),
            })
            # print(f"entry.code is {entry.code}")
            # print(f"entry.type is {entry.type}")
            # print(f"entry.tokenization is {entry.tokenization}")
            # print(f"entry.to_token is {entry.to_token}")
            self.non_special_tokens.append(entry.to_token())
            # count+=1
            # if count >= 100:
            #     sys.exit(0)
        
        # Create tokenizer
        # sys.exit(0)
        super().__init__()

    def convert_event_to_token(self, e: Event, **kwargs) -> Optional[str]:
        """NOTE: This is basically the same as the CLMBR tokenizer's version."""
        event_code = e.code
        # If code isn't in vocab => ignore
        if event_code not in self.code_2_token:
            return None
        
        event_value = e.value
        # If numerical code...
        if (
            'numerical_range' in self.code_2_token[event_code] # `numerical_range` is a valid type for this code
            and event_value is not None # `value` is not None
            and ( # `value` is numeric
                isinstance(event_value, float)
                or isinstance(event_value, int)
            )
        ):
            for token_range in self.code_2_token[event_code]['numerical_range']:
                assert 'token' in token_range, f"ERROR - Missing 'token' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                assert 'tokenization' in token_range, f"ERROR - Missing 'tokenization' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                token: str = token_range['token']
                unit: str = token_range['tokenization']['unit']
                range_start: float = token_range['tokenization']['range_start']
                range_end: float = token_range['tokenization']['range_end']
                if range_start <= event_value <= range_end and e.unit == unit:
                    # print(f"numerical token is {token}") 
                    return token
            return None

        # If textual code...
        if (
            'categorical' in self.code_2_token[event_code] # `categorical` is a valid type for this code
            and event_value is not None # `value` is not None
            and event_value != '' # `value` is not blank
            and ( # `value` is textual
                isinstance(event_value, str)
            )
        ):
            for categorical_value in self.code_2_token[event_code]['categorical']:
                assert 'token' in categorical_value, f"ERROR - Missing 'token' for code={event_code},type=categorical in self.code_2_token: {self.code_2_token[event_code]['categorical']}"
                assert 'tokenization' in categorical_value, f"ERROR - Missing 'tokenization' for code={event_code},type=categorical in self.code_2_token: {self.code_2_token[event_code]['categorical']}"
                if event_value in categorical_value['tokenization']['categories']:
                    token: str = categorical_value['token']
                    # print(f"categorical token is {token}") 
                    return token
            return None

        # If just vanilla code...
        if (
            'code' in self.code_2_token[event_code] # `code` is a valid type for this code
        ):
            token: str = self.code_2_token[event_code]['code'][0]['token']
            # print(f"vanilla token is {token}") 
            return token

        return None
    
    def convert_events_to_tokens(self, events: List[Event], **kwargs) -> List[str]:
        tokens: List[str] = []
        current_visit_end: Optional[datetime.datetime] = None # track the end time of the currently active visit
        previous_visit_end: Optional[datetime.datetime] = None # track the end time of the immediately preceding visit

        for e in events:
            # print(f"The code is {e.code}")

            # Check if we need to add a visit end token
            # if current_visit_end is not None and (
            #     e.start > current_visit_end # If we have [VISIT A = { TOKEN 1, TOKEN 2 }] [TOKEN 3], then add a visit end token before [TOKEN 3]
            #     or "Visit" in e.code # If we have [VISIT A = { TOKEN 1, TOKEN 2 }] [VISIT B = { TOKEN 3, TOKEN 4 }], then add a visit end token after [VISIT A]
            # ):
            #     # This token occurs after the currently active visit ends, so end it (if exists)
            #     if self.is_add_visit_end:
            #         tokens.append(self.visit_end)
            #     current_visit_end = None

            # Check if the event is a visit
            # if "Visit" in e.code:
                    
            #     # Add ATT Tokens, if applicable
            #     # This will be inserted between the prior visit and the current visit
            #     if previous_visit_end is not None:
            #         interval: float = (e.start - previous_visit_end).days # Time (in days) between this visit's start and the immediately preceding visit's end
            #         assert interval >= 0, f"Interval has value = {interval} but should always be positive, but fails on {e}."
                    
            #         if self.is_add_day_att:
            #             if interval <= 1080:
            #                 att = self.day_atts_cehr_gpt[interval - 1]
            #             else:
            #                 att = self.long_att_cehr_gpt
            #             tokens.append(att)
            #         elif self.is_add_day_week_month_att:
            #             if interval < 7:
            #                 att = self.day_atts_cehr_bert[interval - 1]
            #             elif 7 <= interval < 30:
            #                 att = self.week_atts[(interval // 7) - 1]
            #             elif 30 <= interval < 360:
            #                 att = self.month_atts[(interval // 30) - 1]
            #             else:
            #                 att = self.long_att_cehr_bert
            #             tokens.append(att)

            #     # Add visit start token, if applicable
            #     # if self.is_add_visit_start:
            #     #     tokens.append(self.visit_start)
            
            #     # Add token itself
            #     token = self.convert_event_to_token(e, **kwargs)
            #     if token:
            #         tokens.append(token)

            #     # Keep track of this visit's end
            #     current_visit_end = e.end
            #     previous_visit_end = e.end
            # else:
            #     token = self.convert_event_to_token(e, **kwargs)
            #     if token:
            #         tokens.append(token)
        
            token = self.convert_event_to_token(e, **kwargs)
            if token:
                tokens.append(token)
    
        return tokens