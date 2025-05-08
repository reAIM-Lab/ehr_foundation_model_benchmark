import functools
from hf_ehr.data.tokenization_new import BaseTokenizer, collate_femr_timelines

def collate_femr_timelines_collate(
    batch,
    tokenizer,
    dataset_name,
    max_length,
    is_truncation_random,
    is_mlm,
    mlm_prob,
    seed
):
    return collate_femr_timelines(
        batch, tokenizer, dataset_name, max_length,
        is_truncation_random, is_mlm, mlm_prob, seed
    )