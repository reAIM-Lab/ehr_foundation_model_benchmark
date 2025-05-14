# Tools for the EHR benchmarking pipeline

## Unit harmonization - How to prepare units for EHR Foundation Models

You must create a Python file `path.py` with two variables `concepts_path` (for the Parquet OMOP concept table) and `files` (list of the OMOP files to process).
Generated files will be located in the same directory as the initial file.

Step-by-step instructions:
- Add your custom `path.py` in this directory (see example below).
- Run `python analyze_labs.py` to create the summary lab csv table.
- Run `python convert_units.py` to run the conversion - it may take some hours.
- Run `python stats.py` to compute the conversion summary.

Example of `path.py` file:
```python
# separate file to avoid sharing paths on Github
import glob

concepts_path = '/omop_folder/concept/part-00000-xxxx.snappy.parquet'
files = sorted(glob.glob('/omop_folder/part-*-xxx-c000.snappy.parquet'))
```
