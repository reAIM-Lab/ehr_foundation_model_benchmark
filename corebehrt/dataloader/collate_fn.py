import torch

def dynamic_padding(data: list)->dict:
    print('check dynamic padding')
    print(type(data))
    print(len(data))
    print(data[1:10])
    max_len = max([len(patient["input_ids"]) for patient in data])
    for patient in data:
        difference = max_len - len(patient["input_ids"])
        for key, values in patient.items():
            if key in ["target"]:
                if isinstance(values, float):  # 0D: For finetuning
                    patient[key] = torch.tensor(values)
                    continue
                elif values.ndim == 1:  # 1D: For normal pretraining
                    filler = torch.ones(difference, dtype=values.dtype) * -100
            else:
                filler = torch.zeros(difference, dtype=values.dtype)
            patient[key] = torch.cat((values, filler), dim=0)

    padded_data = {}
    for key in data[0].keys():
        padded_data[key] = torch.stack([patient[key] for patient in data])

    return padded_data

class CustomMLMDataCollatorWithPadding:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )

    def __call__(self, features: list) -> dict:
        # Dynamically determine max sequence length
        max_len = max(len(f["input_ids"]) for f in features)

        padded_batch = {}
        for key in features[0].keys():
            # Prepare a list of padded tensors for each key
            padded_tensors = []
            for f in features:
                values = f[key]
                pad_len = max_len - len(values)

                if isinstance(values, torch.Tensor):
                    values = values.clone()
                else:
                    values = torch.tensor(values)

                if key == "target":
                    if values.ndim == 0:  # scalar float
                        values = values.unsqueeze(0)
                    pad_val = -100
                else:
                    pad_val = 0

                padded = torch.cat(
                    [values, torch.full((pad_len,), pad_val, dtype=values.dtype)],
                    dim=0
                )
                padded_tensors.append(padded)

            padded_batch[key] = torch.stack(padded_tensors)

        # Apply MLM to get labels
        mlm_inputs = self.mlm_collator(
            [{"input_ids": ids} for ids in padded_batch["input_ids"]]
        )

        # Overwrite input_ids and add labels
        padded_batch["input_ids"] = mlm_inputs["input_ids"]
        padded_batch["labels"] = mlm_inputs["labels"]
        # attention_mask is created by the MLM collator and padded to match
        padded_batch["attention_mask"] = mlm_inputs["attention_mask"]

        return padded_batch

