from collections import defaultdict
from pathlib import Path
from typing import Dict

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pyl
from transformers import AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from hf_ehr.config import Event
from hf_ehr.data.tokenization import CLMBRTokenizer


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

        self.log("train_loss", loss, prog_bar=False, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        self.log("val_loss", loss, prog_bar=False, logger=True)
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

def load_task_embeddings(args, labels, data_path, device):
    model = AutoModelForCausalLM.from_pretrained(f"StanfordShahLab/{args.model}").to(device)
    tokenizer = CLMBRTokenizer.from_pretrained(f"StanfordShahLab/{args.model}")

    files = sorted(data_path.glob("*.parquet"))

    labels = labels.with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us")),
    )

    base_path = Path(args.base_path)
    save_dir = base_path / "context_clues_embeddings" / args.task / data_path.name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(save_dir)

    embeddings, labels = load_meds(labels, files, model, tokenizer, save_dir, device)
    return embeddings, labels
    
def load_meds(data, files, model, tokenizer, save_dir, device):
    subjects = (
        data.group_by("subject_id")
        .agg(pl.max("prediction_time")
        .alias("prediction_time"))  # Take max prediction_time per subject
    )

    # Load and filter each parquet file in data_path
    batch_embeddings = []
    batch_labels = []
    for file in files:
        save_path = save_dir / f"{Path(file).stem}.pt"

        # If embedding is already saved, then load from save_path
        if save_path.exists():
            embeddings_data = torch.load(save_path)
            embeddings = embeddings_data['embeddings']
            labels = embeddings_data['labels']
        else:
            df = pl.read_parquet(file)

            df = df.with_columns(
                pl.col("time").cast(pl.Datetime("us")), # Convert to Datetime for consistent comparisons
            )
            
            df_joined = df.join(subjects, on="subject_id", how="inner")
            df_filtered = df_joined.filter(df_joined["time"] < df_joined["prediction_time"])

            # Revert unit concatenation
            df_filtered = df_filtered.with_columns(
                pl.col("code").map_elements(
                    lambda x: x.split('//')[0] if isinstance(x, str) and x.startswith('LOINC') else x,
                    return_dtype=pl.Utf8  # Ensure the return is a string
                ).alias("code")
            )

            events = convert_to_events(df_filtered)
            embeddings, labels = get_embeddings(model, tokenizer, events, data, device)

            torch.save({
                'embeddings': embeddings,
                'labels': labels
            }, save_path)
        
        batch_embeddings.append(embeddings)
        batch_labels.append(labels)
    
    # Concatenate all filtered DataFrames into one
    batch_embeddings = torch.cat(batch_embeddings, dim=0)
    batch_labels = torch.cat(batch_labels, dim=0)

    return batch_embeddings, batch_labels

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
    batch_labels = []

    for row in labels.iter_rows(named=True):
        subject_id = row["subject_id"]
        prediction_time = row["prediction_time"]
        label = row["boolean_value"]

        # Extract events for this subject_id
        if subject_id in subject_data:
            events_data = subject_data.get(subject_id, [])  # Get events; default to empty list if not found

            if events_data:  # Only process if data exists
                sorted_events = sorted(events_data, key=lambda x: x[1], reverse=True)
                filtered_events = [event for event, event_time in sorted_events if event_time < prediction_time]

                if filtered_events:
                    batch_events.append(filtered_events)
                    batch_labels.append(label)

    batch_size = 64

    batch_dict = tokenizer(batch_events, max_length=4096, padding=True, truncation=True, return_tensors='pt')
    batch_dict['input_ids'] = batch_dict['input_ids'].flip(dims=[1])  # Reverse along sequence dimension
    batch_dict['attention_mask'] = batch_dict['attention_mask'].flip(dims=[1]) 
    batch_dict.pop("token_type_ids", None)

    padding_token_id = tokenizer.pad_token_id

    input_ids = batch_dict['input_ids']
    # Find the index of the first non-padding token
    start_indices = (input_ids != padding_token_id).int().argmax(dim=1)

    dataset = TensorDataset(batch_dict['input_ids'], batch_dict['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if batch_events:
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                representations = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]

            # Store data
            if representations.dim() == 3:  # (batch_size, sequence_len, embedding_dim)
                batch_embedding.append(representations.squeeze(1).cpu())  # Squeeze sequence_len dimension if batch size is 1
            else:
                batch_embedding.append(representations.cpu())

    batch_embedding = torch.cat(batch_embedding, dim=0)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return batch_embedding, batch_labels

def count_events(labels, files):
    subjects = (
        labels.group_by("subject_id")
        .agg(pl.max("prediction_time")
        .alias("prediction_time"))  # Take max prediction_time per subject
    )

    # Load and filter each parquet file in data_path
    all_events = []
    batch_labels = []
    for file in files[0:10]:
        df = pl.read_parquet(file)
        df_joined = df.join(subjects, on="subject_id", how="inner")
        df_filtered = df_joined.filter(df_joined["time"] < df_joined["prediction_time"])

        # Revert unit concatenation
        df_filtered = df_filtered.with_columns(
            pl.col("code").map_elements(
                lambda x: x.split('//')[0] if isinstance(x, str) and x.startswith('LOINC') else x,
                return_dtype=pl.Utf8  # Ensure the return is a string
            ).alias("code")
        )

        subject_data = convert_to_events(df_filtered)

        batch_events = []
        batch_embedding = []
        batch_labels = []

        for row in labels.iter_rows(named=True):
            subject_id = row["subject_id"]
            prediction_time = row["prediction_time"]
            label = row["boolean_value"]

            # Extract events for this subject_id
            if subject_id in subject_data:
                events_data = subject_data.get(subject_id, [])  # Get events; default to empty list if not found

                if events_data:  # Only process if data exists
                    sorted_events = sorted(events_data, key=lambda x: x[1], reverse=True)
                    filtered_events = [event for event, event_time in sorted_events if event_time.date() < prediction_time]

                    if filtered_events:
                        batch_events.append(filtered_events)
                        batch_labels.append(label)

        all_events.extend(batch_events)
    
    return all_events

def standardize(train, val, test):
    train_mean = train.mean(dim=0, keepdim=True)
    train_std = train.std(dim=0, keepdim=True)

    # Avoid division by zero
    train_std[train_std == 0] = 1.0

    train = (train - train_mean) / train_std
    val = (val - train_mean) / train_std
    test = (test - train_mean) / train_std

    return train, val, test

def create_event(row):
    return Event(
        code=row["code"],
        value=row["value"] if "value" in row else None,
        unit=row["unit"] if "unit" in row else None,
        start=row["start"] if "start" in row else None,
        end=row["end"] if "end" in row else None,
        omop_table=row["omop_table"] if "omop_table" in row else None,
    ), row['time']
