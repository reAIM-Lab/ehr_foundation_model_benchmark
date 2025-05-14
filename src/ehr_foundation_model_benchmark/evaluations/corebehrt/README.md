# CORE-BEHRT benchmark Pipeline
CORE-BEHRT uses the OMOP data as the input directly.

```bash
conda create -n corebehrt python=3.10
```
Install corebehrt and the evaluation packages
```bash
conda activate corebehrt
pip install corebehrt-0.1.0-py3-none-any.whl
pip install meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl
```

Let's set up some environment variables
```bash
export OMOP_DIR=""
export CORE_BEHRT_DATA_DIR=""
export CORE_BEHRT_MODEL_DIR=""
```