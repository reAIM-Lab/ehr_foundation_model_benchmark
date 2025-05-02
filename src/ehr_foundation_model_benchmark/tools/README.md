## How to prepare units for EHR Foundation Models

You must provide a `path.py` with two variables `concepts_path` (for the Parquet OMOP concept table) and `files` (list of the files to process).
Generated files will be located in the same directory as the initial file.

- Run `analyze_labs.py` to create the summary lab csv table.
- Run `convert_units.py` to run the conversion.
- Run `stats.py` to compute the conversion summary.

Example of `path.py` file:
```python
# separate file to avoid sharing paths on Github
import glob

concepts_path = '/omop_folder/concept/part-00000-xxxx.snappy.parquet'
files = sorted(glob.glob('/omop_folder/part-*-xxx-c000.snappy.parquet'))
```
