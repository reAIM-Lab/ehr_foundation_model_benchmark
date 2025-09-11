# FEMR TPP (Temporal Point Process) Models Documentation

## Overview

The `models_tpp` directory contains optimized implementations for marked temporal point process modeling with dual task architecture supporting both non-numerical (without mark) and numerical codes (with mark). This implementation separates traditional event prediction from lab value prediction using matrix-wise operations for efficiency.

## Architecture Overview

### Dual Task Design
1. **Non-numerical codes**: Traditional time-to-event prediction with deephit loss
2. **Numerical codes**: Combined time-value prediction for lab values with deephit loss

### Loss Computation
- **New loss = mean(original mean loss for non-numerical codes, mean loss for lab test codes)**
- each loss consists of event loss (logf()) and censor loss (log(1-F()))
- Separate task layers and vocabularies for each type

---

## Key Files

### `tasks.py` - Core Task Processing

#### Main Classes

##### `MOTORTask`
Primary task class handling both non-numerical and numerical codes with separate vocabularies.

**Key Attributes:**
- `task_to_index_map`: Dictionary mapping non-numerical task names to indices
- `numerical_task_to_index_map`: Dictionary mapping numerical task names to indices  
- `task_value_bins`: Value binning information for numerical tasks

**Key Methods:**

###### `fit_pretraining_task_info()` - Static Factory Method
```python
@classmethod
def fit_pretraining_task_info(cls, db, tokenizer, num_tasks, num_bins, 
                              final_layer_size, codes_to_skip=None, num_value_bins=10)
```
**Purpose:** Analyzes dataset to determine task statistics and create optimal time/value bins
**Process:**
1. Extracts most frequent codes from tokenizer
2. Collects survival statistics per task using `SurvivalCalculator`
3. Filters tasks based on event frequency and value sample count
4. Creates robust time bins using percentiles with `-inf`/`inf` boundaries
5. Creates value bins for numerical tasks with valid bin detection

**Returns:** Configured `MOTORTask` instance

###### `start_subject()` - Subject Initialization
```python  
def start_subject(self, subject, ontology)
```
**Purpose:** Initialize per-subject sparse data structures
**Creates:**
- `per_subject_time_sparse`: Time events for non-numerical codes
- `per_subject_value_sparse`: Time+value events for numerical codes

**Structure:**
- `["data"]`: Event times (seconds)
- `["value"]`: Lab values (numerical tasks only) - **initialized with NaN, not 0**
- `["indices"]`: Task indices in respective vocabularies
- `["indptr"]`: Sparse matrix pointers

###### `add_event()` - Event Processing
```python
def add_event(self, current_date, next_date, next_features=None, actually_add=True)
```
**Purpose:** Process individual events and route to appropriate vocabulary
**Logic:**
1. Gets future events using `SurvivalCalculator`
2. Routes non-numerical codes to `task_to_index_map`
3. Routes numerical codes to `numerical_task_to_index_map`
4. Stores time and value information separately

###### `cleanup()` - Tensor Preparation
```python
def cleanup(self, batch) -> Mapping[str, torch.Tensor]
```
**Purpose:** Convert sparse data to dense tensors for model training
**Returns:**
- **Non-numerical codes:**
  - `"is_event"`: `[pred_points, time_bins, non_numerical_tasks]` 
  - `"is_censored"`: `[pred_points, non_numerical_tasks]`
  - `"censor_time_ratio"`: `[pred_points, time_bins, non_numerical_tasks]`
  
- **Numerical codes:**
  - `"time_event_in_bin"`: `[pred_points, time_bins, numerical_tasks]`
  - `"time_censor_in_bin"`: `[pred_points, time_bins, numerical_tasks]`  
  - `"value_event_in_bin"`: `[pred_points, value_bins, numerical_tasks]`
  - `"time_is_censored "`: `[pred_points, numerical_tasks]`

#### Helper Methods

###### `sparse_to_dense()` - Sparse Conversion
**Purpose:** Convert sparse matrices to dense tensors with proper initialization
**Key Feature:** Value tensors initialized with `NaN` (not 0) since values range from `-inf` to `+inf`

###### `_create_time_bins()` - Time Discretization  
```python
def _create_time_bins(self, time_data, censor_times, num_time_bins)
```
**Purpose:** Vectorized time binning using matrix operations
**Algorithm:**
1. Uses `torch.searchsorted` for efficient bin assignment
2. Handles events and censoring cases separately
3. Ensures exactly one bin per prediction-task combination
4. **Asserts:** `torch.all(bins_per_pred_task == 1)` - no edge case handling

###### `_create_time_value_bins()` - Combined Binning
```python  
def _create_time_value_bins(self, time_numerical, values_numerical, censor_times, num_time_bins)
```
**Purpose:** Matrix-wise creation of time and value bins for numerical tasks
**Key Features:**
- **Future event detection:** `(time != 0) or (~isnan(value))` in theory, (time != 0) should be the same as (~isnan(value))
- **Vectorized value binning:** 
1. Time bins: Same logic as `_create_time_bins` 
2. Value bins: Vectorized assignment using valid bin mapping
3. Returns dictionary with separate event/censor matrices

---

### `transformer.py` - Neural Network Architecture

#### `MOTORTaskHead` Class

##### `__init__()` - Architecture Setup
```python
def __init__(self, hidden_size, linear_interpolation, pretraining_task_info, 
             time_bins, final_layer_size, num_value_bins=10)
```
**Architecture:**
- **`non_numerical_final_layer`**: Time features for non-numerical codes
- **`on_numerical_task_layer`**: Prediction head for non-numerical codes  
- **`numerical_final_layer`**: Time×value features for numerical codes
- **`numerical_task_layer`**: Dynamically sized based on batch data

**Note:** No `value_bins_dict` parameter - determined from batch dimensions

##### `forward()` - Dual Loss Computation
```python
def forward(self, features, batch, return_logits=False)
```

**Non-Numerical Codes (Original Logic):**
1. **Time-dependent features:** `[pred_points, time_bins, final_layer_size]`
2. **Logits:** `[pred_points, time_bins, non_numerical_tasks]`
3. **Probability constraint:** `sum(time_bins) = 1`
4. **Survival analysis:** Uses CDF for censored cases
5. **Loss:** Cross entropy on time horizon only

**Numerical Codes (New Logic):**
1. **Time-value features:** `[pred_points, time_bins, value_bins, final_layer_size]`  
2. **Logits:** `[pred_points, time_bins, value_bins, numerical_tasks]`
3. **Probability constraint:** `sum(time_bins × value_bins) = 1`
4. **Matrix-wise loss computation:**

**Case 1 - Censored Events:**
```python
# Sum over value bins, predict time only
time_only_probs = torch.sum(numerical_probs, dim=2)
censored_loss = log(1-F())
```

**Case 2 - Actual Events:**
```python
# Outer product: time_event ⊗ value_event
event_bins = time_event_in_bin.unsqueeze(2) & value_event_in_bin.unsqueeze(1)
event_loss = f()  
```

**Final Loss:**
```python
loss = mean(non_numerical_loss, numerical_loss)
```

---

## Key Optimizations Implemented

### 1. Value Initialization Fix
- **Problem:** Values initialized with 0, but can range `-inf` to `+inf`
- **Solution:** Initialize with `NaN` (None equivalent) for proper detection

### 2. Matrix-wise Operations  
- **Eliminated:** Nested loops over prediction points and tasks
- **Implemented:** Vectorized tensor operations using broadcasting
- **Performance:** Scales efficiently with batch size and task count

### 3. Separate Task Vocabularies
- **`task_to_index_map`**: Only non-numerical codes
- **`numerical_task_to_index_map`**: Only numerical codes  
- **Benefit:** Clean separation of concerns and vocabulary management

### 4. Flexible Loss Architecture
- **Outer product approach:** `time_event_in_bin ⊗ value_event_in_bin`
- **No nested loops:** Pure matrix operations in transformer
- **Separate handling:** Events vs censored cases with different loss computation

---

## Usage Example

```python
# 1. Fit task from dataset
motor_task = MOTORTask.fit_pretraining_task_info(
    db=subject_database,
    tokenizer=hierarchical_tokenizer, 
    num_tasks=1000,
    num_bins=20,
    final_layer_size=256,
    num_value_bins=10
)

# 2. Initialize model
model = FEMRModel(config, task_config=motor_task.get_task_config())

# 3. Process batch
loss, result = model(batch, return_loss=True)
```

---

## Data Flow Summary

1. **Dataset Analysis** → Task statistics and bin creation
2. **Event Processing** → Separate routing to vocabularies  
3. **Sparse Storage** → Efficient memory usage during batch construction
4. **Matrix Conversion** → Dense tensors with proper initialization
5. **Dual Prediction** → Separate logits for time-only vs time-value
6. **Loss Computation** → Matrix-wise operations, no loops
7. **Mean Loss** → Combined non-numerical and numerical losses

This architecture efficiently handles mixed event types while maintaining clean separation between traditional survival analysis and lab value prediction.