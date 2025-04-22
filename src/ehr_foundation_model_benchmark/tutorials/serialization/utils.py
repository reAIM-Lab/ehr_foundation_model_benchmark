import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from pathlib import Path
import pytorch_lightning as pyl
import torch.optim as optim
import torch.nn as nn

from datetime import timedelta

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class LRModelLightning(pyl.LightningModule):
    def __init__(self, input_dim, seed=42):
        super(LRModelLightning, self).__init__()
        
        torch.manual_seed(seed)

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

normal_range = {'Body Weight': [350, 10000],  
            'Body Height': [5, 100],
            'BMI': [18.5, 24.9],
            'Body Surface Area': [0.1, 10],
            'Heart Rate': [60, 100],
            'Systolic blood pressure': [90, 140],
            'Diastolic blood pressure': [60, 90],
            'Body temperature': [95, 100.4],
            'Respiratory rate': [12, 18],
            'Oxygen saturation': [95, 100],
            'Hemoglobin': [12, 17],
            'Hematocrit': [36, 51],
            'Erythrocytes': [4.2, 5.9],  # LOINC/789-8, LOINC/26453-1
            'Leukocytes': [4, 10],  # LOINC/20584-9, LOINC/6690-2
            'Platelets': [150, 350],  # LOINC/777-3, SNOMED/61928009
            'Sodium': [136, 145],  # LOINC/2951-2, LOINC/2947-0, SNOMED/25197003
            'Potassium': [3.5, 5.0],  # LOINC/2823-3, SNOMED/312468003, LOINC/6298-4, SNOMED/59573005
            'Chloride': [98, 106],  # LOINC/2075-0, SNOMED/104589004, LOINC/2069-3
            'Carbon dioxide, total': [10, 100],  # LOINC/2028-9 (mmol/L), 23-28 (Integer)
            'Calcium': [9, 10.5],  # LOINC/17861-6, SNOMED/271240001
            'Glucose': [70, 100],  # LOINC/2345-7, SNOMED/166900001, LOINC/2339-0, SNOMED/33747003, LOINC/14749-6
            'Urea nitrogen': [8, 20],  # LOINC/3094-0, SNOMED/105011006
            'Creatinine': [0.7, 1.3],  # LOINC/2160-0, SNOMED/113075003
            'Anion gap': [3, 11]}

# Example function to categorize values (you can expand this)
def categorize_value(category, value):
    if category in normal_range:
        lower, upper = normal_range[category]
        if value < lower:
            return f"{value:.2f} (low)"
        elif value > upper:
            return f"{value:.2f} (high)"
        else:
            return f"{value:.2f} (normal)"
    else:
        return f"{value:.2f}"

def normalize_icd10(df: pl.DataFrame, column: str) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(column).is_not_null() & pl.col(column).str.starts_with("ICD10CM/"))
        .then(
            pl.col(column)
            .str.replace(r"ICD10CM/", "", literal=True)  # Remove prefix
            .str.replace(r"\.?0+$", "", literal=False)  # Remove trailing zeros & period
            .map_elements(lambda x: "ICD10CM/" + x)  # Add prefix back
        )
        .otherwise(pl.col(column))
        .alias(f"normalized_{column}")  # New column with normalized values
    )

def extract_labs(df_filtered):
    codes_to_keep = {'Body Weight': ['LOINC/29463-7'],  
                     'Body Height': ['LOINC/8302-2'],
                     'BMI': ['LOINC/39156-5'],
                     'Body Surface Area': ['LOINC/8277-6', 'SNOMED/301898006'],
                     'Heart Rate': ['LOINC/8867-4', 'SNOMED/364075005'],
                     'Systolic blood pressure': ['LOINC/8480-6', 'SNOMED/271649006'],
                     'Diastolic blood pressure': ['LOINC/8462-4', 'SNOMED/271650006'],
                     'Body temperature': ['LOINC/8310-5'],
                     'Respiratory rate': ['LOINC/9279-1'],
                     'Oxygen saturation': ['LOINC/LP21258-6'],
                     'Hemoglobin': ['LOINC/718-7', 'SNOMED/271026005', 'SNOMED/441689006'],
                     'Hematocrit': ['LOINC/4544-3', 'LOINC/20570-8', 'LOINC/48703-3', 'SNOMED/28317006'],
                     'Erythrocytes': ['LOINC/789-8', 'LOINC/26453-1'],
                    'Leukocytes': ['LOINC/20584-9', 'LOINC/6690-2'],
                    'Platelets': ['LOINC/777-3', 'SNOMED/61928009'],
                    'Sodium': ['LOINC/2951-2', 'LOINC/2947-0', 'SNOMED/25197003'],
                    'Potassium': ['LOINC/2823-3', 'SNOMED/312468003', 'LOINC/6298-4', 'SNOMED/59573005'],
                    'Chloride': ['LOINC/2075-0', 'SNOMED/104589004', 'LOINC/2069-3'],
                    'Carbon dioxide, total': ['LOINC/2028-9'],
                    'Calcium': ['LOINC/17861-6', 'SNOMED/271240001'],
                    'Glucose': ['LOINC/2345-7', 'SNOMED/166900001', 'LOINC/2339-0', 'SNOMED/33747003', 'LOINC/14749-6'],
                    'Urea nitrogen': ['LOINC/3094-0', 'SNOMED/105011006'],
                    'Creatinine': ['LOINC/2160-0', 'SNOMED/113075003'],
                    'Anion gap': ['LOINC/33037-3', 'LOINC/41276-7', 'SNOMED/25469001']}
    
    code_to_category = {code: category for category, codes in codes_to_keep.items() for code in codes}
    
    df_filtered = df_filtered.with_columns(
        pl.col("code").map_elements(
            lambda x: x.split('//')[0] if isinstance(x, str) and x.startswith('LOINC') else x,
            return_dtype=pl.Utf8  # Ensure the return is a string
        ).alias("code")
    )

    df_filtered = df_filtered.with_columns(
        pl.col("code").map_elements(
            lambda x: x.split('//')[0] if isinstance(x, str) and x.startswith('SNOMED') else x,
            return_dtype=pl.Utf8  # Ensure the return is a string
        ).alias("code")
    )

    df_labs = df_filtered.filter(df_filtered["code"].is_in(code_to_category.keys()))
    df_labs = df_labs.filter(df_labs["numeric_value"].is_not_null())

    # Map 'code' to its corresponding 'category'
    df_labs = df_labs.with_columns(
        pl.col("code").replace(code_to_category).alias("category")
    )

    # Keep only the last three records per category
    df_labs = (
        df_labs.sort("time", descending=True)  # Ensure sorting by time
        .group_by(["subject_id", "prediction_time", "category"], maintain_order=True)
        .head(3)
    )
    return df_labs

def extract_demo(df_filtered):
    person = df_filtered.filter(pl.col("table").str.starts_with("person"))

    gender = (
        person.filter(pl.col("normalized_code").str.starts_with("Gender/"))
        .select(
            pl.coalesce([pl.col("concept_name"), pl.col("normalized_code").str.split("/").list.get(1)])
            .str.to_lowercase()
            .alias("gender")
        )
    )
    race = (
        person.filter(pl.col("normalized_code").str.starts_with("Race/"))
        .select(
            pl.coalesce([pl.col("concept_name"), pl.col("normalized_code").str.split("/").list.get(1)])
            .str.to_lowercase()
            .alias("race")
        )
    )
    ethnicity = (
        person.filter(pl.col("normalized_code").str.starts_with("Ethnicity/"))
        .select(
            pl.coalesce([pl.col("concept_name"), pl.col("normalized_code").str.split("/").list.get(1)])
            .str.to_lowercase()
            .alias("ethnicity")
        )
    )
    df = df_filtered.with_columns([
        ((pl.col("prediction_time") - pl.col("time")).dt.total_days() / 365.25)
        .round(0)
        .cast(pl.Int32)
        .alias("age")
    ])
    age = df.filter(pl.col("normalized_code") == "MEDS_BIRTH").select(pl.col("age").alias("age"))
    df_demographics = pl.concat([age, ethnicity, gender, race], how="horizontal")

    return df_demographics

# Generate markdown for each patient
def generate_markdown(df_labs, df_visit, subjects):

    markdowns = []
    labels = []
    
    for (subject_id, prediction_time), subject_df in df_labs.group_by(["subject_id", "prediction_time"]):
        #print(subject_df)
        markdown_str = f"\n # Electronic Health Record\n"
        markdown_str += f"## Prediction Time: {prediction_time}\n"
        subject = subjects.filter(
            (pl.col("subject_id") == subject_id) & (pl.col("prediction_time") == prediction_time)
        )
        label = subject['boolean_value'].to_list()[0]
        labels.append(label)

        visit_codes = df_visit.filter(
            (pl.col("subject_id") == subject_id) & (pl.col("prediction_time") == prediction_time)
        )
        body_metrics = []
        vitals = []
        labs = []

        for category, records in subject_df.group_by("category"):
            category = category[0]
            values = [categorize_value(category, v) for v in records["numeric_value"].to_list() if v is not None]
            values_str = ", ".join(values)

            # Organizing into body metrics and vital signs
            if category in {"Body Weight", "Body Height", "BMI", "Body Surface Area"}:
                body_metrics.append(f"- {category}: {values_str}")
            elif category in {"Heart Rate", 'Systolic blood pressure', 'Diastolic blood pressure', 'Body temperature', 'Respiratory rate', 'Oxygen saturation'}:
                vitals.append(f"- {category}: {values_str}")
            else:
                labs.append(f"- {category}: {values_str}")

        demographics = extract_demo(visit_codes)

        # Append formatted sections
        age_value = str(demographics['age'].to_list()[0])
        markdown_str += "## Demographics\n"
        markdown_str += "Patient age: " + age_value + "\n"
        markdown_str += "Patient gender: " + demographics['gender'].to_list()[0] + "\n"
        markdown_str += "Patient race: " + demographics['race'].to_list()[0] + "\n"

        markdown_str += "## Recent Body Metrics\n"
        if body_metrics:
            markdown_str += "\n".join(body_metrics) + "\n"
        markdown_str += "## Recent Vital Signs\n"
        if vitals:
            markdown_str += "\n".join(vitals) + "\n"
        markdown_str += "## Recent Lab Results\n"
        if labs:
            markdown_str += "\n".join(labs) + "\n"

        filtered_visit_types = (
            visit_codes
            .filter(pl.col("table").str.starts_with("visit"))
            .filter(pl.col("code").str.starts_with("Visit/"))
            .filter(pl.col("time") >= (pl.col("prediction_time") - timedelta(days=100))) 
            .sort("time", descending=True)  # Sort by time in descending order
        )

        filtered_conditions = (
            visit_codes
            .filter(pl.col("table").str.starts_with("condition"))
            .filter(pl.col("concept_name").is_not_null())
            .filter(pl.col("time") >= (pl.col("prediction_time") - timedelta(days=100))) 
            .sort("time", descending=True)  # Ensure sorting by time
        )

        filtered_medications = (
            visit_codes
            .filter(pl.col("table").str.starts_with("drug_exposure"))
            .filter(pl.col("concept_name").is_not_null())
            .filter(pl.col("time") >= (pl.col("prediction_time") - timedelta(days=100))) 
            .sort("time", descending=True)
        )

        filtered_proc = (
            visit_codes
            .filter(pl.col("table").str.starts_with("procedure"))
            .filter(pl.col("concept_name").is_not_null())
            .filter(pl.col("time") >= (pl.col("prediction_time") - timedelta(days=100))) 
            .sort("time", descending=True)
        )

        markdown_str += "## Past Medical Visits (Most recent first)\n"
        if not filtered_visit_types.is_empty():
            visits = filtered_visit_types["concept_name"].to_list()  # Extract visit names
            #visit_ids = filtered_visit_types["visit_id"].to_list() 

            filtered_visit_types = filtered_visit_types.with_columns(
                ((pl.col("end") - pl.col("time")).dt.total_hours()).alias("duration")
            )
            filtered_visit_types = filtered_visit_types.with_columns(
                pl.col("time").dt.date().alias("date")
            )
            #associated_conditions = filtered_conditions.filter(pl.col("visit_id").is_in(visit_ids))
            #associated_proc = filtered_proc.filter(pl.col("visit_id").is_in(visit_ids))
            times = filtered_visit_types["date"].to_list()
            durations = filtered_visit_types["duration"].to_list()
            #markdown_str += "\n".join([f"- {visit} on {time} (Duration: {duration} hours)" for visit, time, duration in zip(visits, times, durations)]) + "\n"
            markdown_str += "\n".join([
                f"- {visit} on [{time}]({time}) (Duration: {duration} hours)" 
                if visit not in ["Outpatient Visit", "Office Visit"] 
                else f"- {visit} on [{time}]({time})"  # Skip adding duration for specific visits
                for visit, time, duration in zip(visits, times, durations)
            ]) + "\n"
            markdown_str += "## Conditions\n"

            if not filtered_conditions.is_empty():
                conditions = filtered_conditions["concept_name"].unique().to_list()  # Extract condition names
                markdown_str += "\n".join([f"- {cond}" for cond in conditions]) + "\n"
            # markdown_str += "## Medications\n"
            # if not filtered_medications.is_empty():
            #     medications = filtered_medications["concept_name"].to_list()  # Extract condition names
            #     markdown_str += "\n".join([f"- {cond}" for cond in medications]) + "\n"
            markdown_str += "## Procedures\n"
            if not filtered_proc.is_empty():
                procedures = filtered_proc["concept_name"].unique().to_list()  # Extract condition names
                markdown_str += "\n".join([f"- {cond}" for cond in procedures]) + "\n"

            print(markdown_str)

        markdowns.append(markdown_str)

    return markdowns, labels

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def last_token_pool(last_hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths, :]
    
# def mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     """Computes mean pooling over valid token positions based on attention mask."""
#     # Ensure masked positions are not included in the mean
#     masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)  # Expand mask for broadcasting
#     print(masked_embeddings[0])
#     sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over the sequence dimension
#     valid_lengths = attention_mask.sum(dim=1, keepdim=True)  # Get valid token lengths
#     return sum_embeddings / valid_lengths.clamp(min=1) 

def generate_embeddings(markdown_output, model, tokenizer, task, device, args):
    
    if args.task == 'hf_readmission_meds':
        query = get_detailed_instruct(task, 'Will this patient with heart failure be readmitted within 30 days?')
    elif args.task == 'cad_cabg_meds':
        query = get_detailed_instruct(task, 'Will this patient with coronary artery disease receive coronary artery bypass graft treatment within a year?')
    elif args.task == 'afib_ischemic_stroke_meds':
        query = get_detailed_instruct(task, 'Will this patient with atrial fibrilation develop ischemic stroke within a year?')

    # Create batch by appending the same query to each EHR record
    input_texts = [query + ehr for ehr in markdown_output]

    # Tokenize the batch
    max_length = 2048
    batch_size = 50

    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

    dataset = TensorDataset(batch_dict['input_ids'], batch_dict['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize list for embeddings and labels
    batch_embeddings = []

    # Process input texts in batches
    for batch in dataloader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        #embeddings = mean_pool(outputs.last_hidden_state, attention_mask)
        batch_embeddings.append(embeddings.cpu())

    # Combine all the batch embeddings into a single tensor
    batch_embeddings = torch.cat(batch_embeddings, dim=0)
    return batch_embeddings

def load_embeddings(subjects, concepts, files, embedding_path, model, tokenizer, task, device, args):
    embeddings = []
    labels = []

    for file in files:
        save_path = embedding_path / f"{Path(file).stem}.pt"

        if save_path.exists():
            embeddings_data = torch.load(save_path)
            batch_embeddings = embeddings_data['embeddings']
            batch_labels = embeddings_data['labels']
        else:
            df = pl.read_parquet(file)
            df_joined = subjects.join(df, on=["subject_id"], how="left")
            df_filtered = df_joined.filter(df_joined["time"] < df_joined["prediction_time"])

            df_labs = extract_labs(df_filtered)

            df_visit = df_filtered.filter(
                pl.col("table").str.starts_with("condition") |
                pl.col("table").str.starts_with("drug_exposure") |
                pl.col("table").str.starts_with("procedure") |
                pl.col("table").str.starts_with("visit") |
                pl.col("table").str.starts_with("person")
            )

            df_visit = normalize_icd10(df_visit, 'code')
            df_visit = df_visit.join(concepts, left_on="normalized_code", right_on="normalized_vocabulary_concept", how="left")

            # Example: Convert to markdown
            markdown_output, batch_labels = generate_markdown(df_labs, df_visit, subjects)
            # print(markdown_output[0])
            # print(markdown_output[1])
            # print(markdown_output[2])

            batch_embeddings = generate_embeddings(markdown_output, model, tokenizer, task, device, args)

            print("Saved: ", file)

            torch.save({
                'embeddings': batch_embeddings,
                'labels': batch_labels
            }, save_path)

        embeddings.append(batch_embeddings)
        labels.append(batch_labels)

    embeddings = torch.cat(embeddings, dim=0)
    labels = [torch.tensor(label) if isinstance(label, list) else label for label in labels]
    labels = torch.cat(labels, dim=0)

    return embeddings, labels

def standardize(train, val, test):
    train_mean = train.mean(dim=0, keepdim=True)
    train_std = train.std(dim=0, keepdim=True)

    # Avoid division by zero
    train_std[train_std == 0] = 1.0

    train = (train - train_mean) / train_std
    val = (val - train_mean) / train_std
    test = (test - train_mean) / train_std

    return train, val, test