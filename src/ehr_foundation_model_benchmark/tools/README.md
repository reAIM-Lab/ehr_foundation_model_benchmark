## How to prepare units for EHR Foundation Models

You must provide a `path.py` with two variables `concepts_path` (for the Parquet OMOP concept table) and `files` (list of the files to process).
Generated files will be located in the same directory as the initial file.

- Run `analyze_labs.py` to create the summary lab csv table.
- Run `convert_units.py` to run the conversion.
- Run `stats.py` to compute the conversion summary.