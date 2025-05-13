from collections import defaultdict
from pathlib import Path
from typing import Dict
import datetime
import os
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pyl
from transformers import AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from datetime import timedelta
import sys
import pandas as pd
from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, record_function, ProfilerActivity

from hf_ehr.config import Event
from hf_ehr.data.tokenization_new import CLMBRTokenizer
from hf_ehr.utils import load_config_from_path, load_tokenizer_from_path, load_model_from_path

# Model-specific parameters
BATCH_SIZE=12
CONTEXT_LENGTH=8192


class LRModelLightning(pyl.LightningModule):
    def __init__(self, input_dim):
        super(LRModelLightning, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Assuming binary classification
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = self.criterion(outputs, labels)

        self.log("train_loss", loss, prog_bar=False, logger=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        self.log("val_loss", loss, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer

    def predict_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)

        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        return preds, probs

    def test_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = self.criterion(outputs, labels)

        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        probs = torch.softmax(outputs, dim=1)[:, 1]
        auroc = roc_auc_score(labels.cpu(), probs.cpu())

        precision, recall, _ = precision_recall_curve(labels.cpu(), probs.cpu())
        pr_auc = auc(recall, precision)

        # Log test loss and accuracy
        self.log("test_loss", loss, prog_bar=False, logger=True)
        self.log("test_acc", acc, prog_bar=False, logger=True)
        self.log("test_auroc", auroc, prog_bar=False, logger=True)
        self.log("test_pr_auc", pr_auc, prog_bar=False, logger=True)

        return {"test_loss": loss, "test_acc": acc, "test_auroc": auroc, "test_pr_auc": pr_auc}
    
def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def load_task_embeddings(args, labels, data_path, device,split):
    # Load model and tokenizer
    if args.model_type == 'mamba-ehrshot':
        model = AutoModelForCausalLM.from_pretrained(f"StanfordShahLab/{args.model}").to(device)
        tokenizer = CLMBRTokenizer.from_pretrained(f"StanfordShahLab/{args.model}")
    else:
        model_path = args.model_path
        assert os.path.exists(model_path), f"No model exists @ `{model_path}`"
        config = load_config_from_path(model_path)
        tokenizer = load_tokenizer_from_path(model_path)
        model = load_model_from_path(model_path)
    
    # print(f"model is {model}")
    # elif 

    embeddings, labels, ids, times = load_meds(labels, data_path, model, tokenizer, device, args,split)

    return embeddings, labels, ids, times

'''
data_path: original meds data for training/tuning/testing, including dozens of .parquet
data: downstram task csv, including subject_id, prediction_time, boolean value
'''
def load_meds(data, data_path, model, tokenizer, device, args,split):
    files = sorted(data_path.glob("*.parquet"))

    base_path = Path(args.input_meds)
    # save_dir = args.out_dir
    save_dir = base_path / args.model_type / args.task / data_path.name
    save_dir.mkdir(parents=True, exist_ok=True)

    subjects = (
        data.group_by("subject_id")
        .agg(pl.max("prediction_time")
        .alias("prediction_time"))  # Take max prediction_time per subjectƒ
    )

    # Load and filter each parquet file in data_path
    batch_embeddings = []
    batch_labels = []
    batch_ids = []
    batch_times = []
    for file in files:
        save_path = save_dir / f"{Path(file).stem}.pt"

        # If embedding is already saved, then load from save_path
        if save_path.exists():
            with torch.serialization.safe_globals([datetime.datetime]):
                embeddings_data = torch.load(save_path, weights_only=False)
            # embeddings_data = torch.load(save_path)
            embeddings = embeddings_data['features']
            labels = embeddings_data['boolean_value']
            ids = embeddings_data['subject_id']
            prediction_times = embeddings_data['prediction_time']
            if isinstance(batch_embeddings,torch.Tensor):
                batch_embeddings = batch_embeddings.tolist()
            if isinstance(labels,torch.Tensor):
                labels = labels.tolist()
        else:
            df = pl.read_parquet(file)
            
            df_joined = df.join(subjects, on="subject_id", how="inner")
            df_filtered = df_joined.filter(df_joined["time"] <= df_joined["prediction_time"]) # filter by max prediction time for efficiency
            # print(f"df are {df}")
            # print(f"subjects are {subjects}")
            # print(f"df_join are {df_joined}")
            # # print(f"df_filtered are {df_filtered}")
            # sys.exit(0)

            if args.model_type == 'mamba-ehrshot':
                # Revert unit concatenation for Stanford-trained model
                df_filtered = df_filtered.with_columns(
                    pl.col("code").map_elements(
                        lambda x: x.split('//')[0] if isinstance(x, str) and x.startswith('LOINC') else x,
                        return_dtype=pl.Utf8  # Ensure the return is a string
                    ).alias("code")
                )

            # Convert each row to Event object
            events = convert_to_events(df_filtered)

            # Generate embedding from pretrained model
            embeddings, labels, ids, prediction_times = get_embeddings(model, tokenizer, events, data, device)

            # Save embeddings for linear probing & evaluation
            torch.save({
                'subject_id': ids,
                'prediction_time': prediction_times,
                'features': embeddings,
                'boolean_value': labels
            }, save_path)
        
        # batch_embeddings.append(embeddings)
        # batch_labels.append(labels)
        # batch_ids.append(ids)
        # batch_times.append(prediction_times)
        if isinstance(ids, list):
            for i in range(len(ids)):
                batch_ids.append(ids[i])
                batch_times.append(prediction_times[i])
                batch_labels.append(labels[i] if isinstance(labels, list) else labels[i].item())
                batch_embeddings.append(embeddings[i].cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings[i])
        print(f"file {Path(file)} is finished")
    
    # Concatenate all filtered DataFrames into one
    # batch_embeddings = torch.cat(batch_embeddings, dim=0)
    # batch_labels = torch.cat(batch_labels, dim=0)
    df = pd.DataFrame({
    "subject_id":      batch_ids,
    "prediction_time": batch_times,
    "boolean_value":   batch_labels,
    "features":        batch_embeddings,
    })


    df = df.sort_values(["subject_id", "prediction_time"])

    print("DataFrame info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())

    output_dir = base_path / args.model_type / args.task/ "features_with_label" 
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir/f"{split}.parquet", index=False)

    print("Wrote combined embeddings to:", output_dir/f"{split}.parquet")


    return batch_embeddings, batch_labels, batch_ids, batch_times

def convert_to_events(tokens):
    subject_data = defaultdict(list)

    for row in tokens.iter_rows(named=True):
        subject_id = row["subject_id"]
        event, time = create_event(row)
        subject_data[subject_id].append((event, time))

    return subject_data

def get_embeddings(model, tokenizer, subject_data, labels, device):
    batch_events = []
    batch_embedding = []
    batch_ids = []
    batch_times = []
    batch_labels = []

    # Extract events for each prediction sample
    for row in labels.iter_rows(named=True):
        subject_id = row["subject_id"]
        prediction_time = row["prediction_time"]
        label = row["boolean_value"]

        # Extract events for this subject_id
        if subject_id in subject_data:
            events_data = subject_data.get(subject_id, [])  # Get events; default to empty list if not found

            if events_data:  # Only process if data exists
                two_years_ago = prediction_time - timedelta(days=2*365)

                sorted_events = sorted(events_data, key=lambda x: x[1], reverse=True)
                filtered_events = [event for event, event_time in sorted_events if event_time <= prediction_time]
                # filtered_events = [event for event, event_time in sorted_events if two_years_ago <= event_time <= prediction_time]

                #filtered_eventtimes = [event_time for event, event_time in sorted_events if two_years_ago <= event_time < prediction_time]

                if filtered_events:

                    batch_ids.append(subject_id)
                    batch_times.append(prediction_time)
                    batch_events.append(filtered_events)
                    batch_labels.append(label)

    tokenized = [tokenizer(e, truncation=True, max_length=CONTEXT_LENGTH) for e in batch_events]
    lengths = [len(t['input_ids'][0]) for t in tokenized]

    # Step 2: Sort by length
    sorted_data = sorted(zip(batch_events, tokenized, lengths, batch_ids, batch_times, batch_labels), key=lambda x: x[2], reverse=True)

    # Step 3: Bucket into mini-batches with similar lengths
    batches = [
        sorted_data[i:i+BATCH_SIZE] for i in range(0, len(sorted_data), BATCH_SIZE)
    ]

    sorted_ids = []
    sorted_times = []
    sorted_labels = []

    total_flops = []
    # print(f"really signature is {model.forward.__code__.co_varnames}")
    for batch in batches:
        batch_events_, _, lengths_, batch_ids_, batch_times_, batch_labels_ = zip(*batch)
        batch_max_len = max(lengths_)
        batch_dict = tokenizer(batch_events_, padding=True, truncation=True, max_length=batch_max_len, return_tensors='pt')

        sorted_ids.extend(batch_ids_)
        sorted_times.extend(batch_times_)
        sorted_labels.extend(batch_labels_)

        # print()
        batch_dict['input_ids'] = batch_dict['input_ids'].flip(dims=[1])
        batch_dict['attention_mask'] = batch_dict['attention_mask'].flip(dims=[1])

        batch_dict.pop("token_type_ids", None)

        input_ids = batch_dict['input_ids'].to(device)
        attention_mask = batch_dict['attention_mask'].to(device)
        model = model.to(device)

        with torch.no_grad():
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": True
            }

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_flops=True) as prof:
            outputs = model.model(**model_inputs)

            # print(prof.key_averages().table(sort_by="flops", row_limit=10))
            # total_gflops = 0
            # for event in prof.key_averages():
            #     if hasattr(event, 'flops') and event.flops > 0:
            #         total_gflops += event.flops / 1e9  # Convert to GFLOPs
            # # print(f"Total GFLOPs: {total_gflops:.3f}")
            # total_flops.append(total_gflops)
            representations = outputs.hidden_states[-1][:, -1, :]

            if representations.dim() == 3:  # (batch_size, sequence_len, embedding_dim)
                batch_embedding.append(representations.squeeze(1).cpu())  # Squeeze sequence_len dimension if batch size is 1
            else:
                batch_embedding.append(representations.cpu())
    # print(f"sum of flops is: {sum(total_flops)}, the list of total_flops are {total_flops}")
    batch_embedding = torch.cat(batch_embedding, dim=0).tolist()
    # sorted_labels = torch.tensor(sorted_labels, dtype=torch.long)

    return batch_embedding, sorted_labels, sorted_ids, sorted_times


def create_event(row):
    return Event(
        code=row["code"],
        value=row["value"] if "value" in row else None,
        unit=row["unit"] if "unit" in row else None,
        start=row["start"] if "start" in row else None,
        end=row["end"] if "end" in row else None,
        omop_table=row["omop_table"] if "omop_table" in row else None,
    ), row['time']


# def count_events(labels, files):
#     subjects = (
#         labels.group_by("subject_id")
#         .agg(pl.max("prediction_time")
#         .alias("prediction_time"))  # Take max prediction_time per subject
#     )

#     # Load and filter each parquet file in data_path
#     all_events = []
#     batch_labels = []
#     for file in files[0:10]:
#         df = pl.read_parquet(file)
#         df_joined = df.join(subjects, on="subject_id", how="inner")
#         df_filtered = df_joined.filter(df_joined["time"] < df_joined["prediction_time"])

#         # Revert unit concatenation
#         df_filtered = df_filtered.with_columns(
#             pl.col("code").map_elements(
#                 lambda x: x.split('//')[0] if isinstance(x, str) and x.startswith('LOINC') else x,
#                 return_dtype=pl.Utf8  # Ensure the return is a string
#             ).alias("code")
#         )

#         subject_data = convert_to_events(df_filtered)

#         batch_events = []
#         batch_embedding = []
#         batch_labels = []

#         for row in labels.iter_rows(named=True):
#             subject_id = row["subject_id"]
#             prediction_time = row["prediction_time"]
#             label = row["boolean_value"]

#             # Extract events for this subject_id
#             if subject_id in subject_data:
#                 events_data = subject_data.get(subject_id, [])  # Get events; default to empty list if not found

#                 if events_data:  # Only process if data exists
#                     sorted_events = sorted(events_data, key=lambda x: x[1], reverse=True)
#                     filtered_events = [event for event, event_time in sorted_events if event_time.date() < prediction_time]

#                     if filtered_events:
#                         batch_events.append(filtered_events)
#                         batch_labels.append(label)

#         all_events.extend(batch_events)
    
#     return all_events