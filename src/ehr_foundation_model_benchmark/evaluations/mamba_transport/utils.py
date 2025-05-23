from collections import defaultdict
from pathlib import Path
from typing import Dict

import polars as pl
import torch
from transformers import AutoModelForCausalLM
from datetime import timedelta

from hf_ehr.config import Event
from hf_ehr.data.tokenization import CLMBRTokenizer

# Model-specific parameters
BATCH_SIZE=8
CONTEXT_LENGTH=8192
    
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
    # Load model and tokenizer
    if args.model_type == 'mamba-ehrshot':
        model = AutoModelForCausalLM.from_pretrained(f"StanfordShahLab/{args.model}").to(device)
        tokenizer = CLMBRTokenizer.from_pretrained(f"StanfordShahLab/{args.model}")
    # elif 

    embeddings, labels, ids, times = load_meds(labels, data_path, model, tokenizer, device, args)

    return embeddings, labels, ids, times
    
def load_meds(data, data_path, model, tokenizer, device, args):
    files = sorted(data_path.glob("*.parquet"))

    base_path = Path(args.input_meds)
    save_dir = base_path / args.model_type / args.task / data_path.name
    save_dir.mkdir(parents=True, exist_ok=True)

    subjects = (
        data.group_by("subject_id")
        .agg(pl.max("prediction_time")
        .alias("prediction_time"))  # Take max prediction_time per subject
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
            embeddings_data = torch.load(save_path)
            embeddings = embeddings_data['embeddings']
            labels = embeddings_data['labels']
            ids = embeddings_data['ids']
            prediction_times = embeddings_data['prediction_times']
        else:
            df = pl.read_parquet(file)
            
            df_joined = df.join(subjects, on="subject_id", how="inner")
            df_filtered = df_joined.filter(df_joined["time"] <= df_joined["prediction_time"]) # filter by max prediction time for efficiency

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
            embeddings, labels, ids, prediction_times = get_embeddings(model, tokenizer, events, data, device, args)

            # Save embeddings for linear probing & evaluation
            torch.save({
                'ids': ids,
                'prediction_times': prediction_times,
                'embeddings': embeddings,
                'labels': labels
            }, save_path)
        
        batch_embeddings.append(embeddings)
        batch_labels.append(labels)
        batch_ids.extend(ids)
        batch_times.extend(prediction_times)
    
    # Concatenate all filtered DataFrames into one
    batch_embeddings = torch.cat(batch_embeddings, dim=0)
    batch_labels = torch.cat(batch_labels, dim=0)

    return batch_embeddings, batch_labels, batch_ids, batch_times

def convert_to_events(tokens):
    subject_data = defaultdict(list)

    for row in tokens.iter_rows(named=True):
        subject_id = row["subject_id"]
        event, time = create_event(row)
        subject_data[subject_id].append((event, time))

    return subject_data

def get_embeddings(model, tokenizer, subject_data, labels, device, args):
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

                if args.task in ["AMI", "Celiac", "CLL", "HTN", "Ischemic_Stroke", "MASLD", "Osteoporosis", "Pancreatic_Cancer", "SLE", "T2DM"]:
                    filtered_events = [event for event, event_time in sorted_events if two_years_ago <= event_time <= prediction_time]
                else:
                    filtered_events = [event for event, event_time in sorted_events if event_time <= prediction_time]

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

    for batch in batches:
        batch_events_, _, lengths_, batch_ids_, batch_times_, batch_labels_ = zip(*batch)
        batch_max_len = max(lengths_)
        batch_dict = tokenizer(batch_events_, padding=True, truncation=True, max_length=batch_max_len, return_tensors='pt')

        sorted_ids.extend(batch_ids_)
        sorted_times.extend(batch_times_)
        sorted_labels.extend(batch_labels_)

        batch_dict['input_ids'] = batch_dict['input_ids'].flip(dims=[1])
        batch_dict['attention_mask'] = batch_dict['attention_mask'].flip(dims=[1])
        batch_dict.pop("token_type_ids", None)

        input_ids = batch_dict['input_ids'].to(device)
        attention_mask = batch_dict['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            representations = outputs.hidden_states[-1][:, -1, :]

            if representations.dim() == 3:  # (batch_size, sequence_len, embedding_dim)
                batch_embedding.append(representations.squeeze(1).cpu())  # Squeeze sequence_len dimension if batch size is 1
            else:
                batch_embedding.append(representations.cpu())

    batch_embedding = torch.cat(batch_embedding, dim=0)
    sorted_labels = torch.tensor(sorted_labels, dtype=torch.long)

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