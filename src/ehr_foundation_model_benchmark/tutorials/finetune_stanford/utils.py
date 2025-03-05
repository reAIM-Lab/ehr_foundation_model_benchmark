from hf_ehr.config import Event
from typing import Dict
import torch.nn as nn
from pathlib import Path
import polars as pl
from collections import defaultdict
import pytorch_lightning as pyl
import torch.optim as optim
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

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

        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

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
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        self.log("test_auroc", auroc, prog_bar=True, logger=True)
        self.log("test_pr_auc", pr_auc, prog_bar=True, logger=True)

        return {"test_loss": loss, "test_acc": acc, "test_auroc": auroc, "test_pr_auc": pr_auc}
    
def load_meds(data, path, model, tokenizer, save_dir, device):
    subjects = (
        data.group_by("subject_id")
        .agg(pl.max("prediction_time")
        .alias("prediction_time"))  # Take max prediction_time per subject
    )

    # Load and filter each parquet file in data_path
    batch_embeddings = []
    batch_labels = []
    for file in sorted(path.glob("*.parquet")):
        save_path = save_dir / f"{Path(file).stem}.pt"

        # If embedding is already saved, then load from save_path
        if save_path.exists():
            embeddings_data = torch.load(save_path)
            embeddings = embeddings_data['embeddings']
            labels = embeddings_data['labels']
        else:
            df = pl.read_parquet(file)
            df_joined = df.join(subjects, on="subject_id", how="inner")
            df_filtered = df_joined.filter(df_joined["time"] < df_joined["prediction_time"])

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
            events_times = subject_data.get(subject_id, [])  # Get events; default to empty list if not found

            if events_times:  # Only process if data exists
                filtered_events = [event for event, event_time in events_times if event_time < prediction_time]

                if filtered_events:
                    batch_events.append(filtered_events)
                    batch_labels.append(label)

    batch_size = 40

    if batch_events:
        for i in range(0, len(batch_events), batch_size):
            sub_batch_events = batch_events[i:i+batch_size]
            tokenized = tokenizer(
                sub_batch_events, 
                add_special_tokens=True, 
                return_tensors='pt', 
                padding=True, 
                truncation="only_first", 
                max_length=2048  # Ensure truncation at 2048 tokens
            )
            tokenized.pop("token_type_ids", None)  # Remove unused token IDs
            tokenized = {key: val.to(device) for key, val in tokenized.items()}

            # Get model embeddings
            with torch.no_grad():
                representations = model(**tokenized).logits[:, -1, :]
            # Store data
            if representations.dim() == 3:  # (batch_size, sequence_len, embedding_dim)
                batch_embedding.append(representations.squeeze(1).cpu())  # Squeeze sequence_len dimension if batch size is 1
            else:
                batch_embedding.append(representations.cpu())

    batch_embedding = torch.cat(batch_embedding, dim=0)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return batch_embedding, batch_labels

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
