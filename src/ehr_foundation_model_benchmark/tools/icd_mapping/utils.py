from pathlib import Path

import polars as pl
import requests
from zipfile import ZipFile
import re
import io

def add_dot(code: pl.Expr, position: int) -> pl.Expr:
    return (
        pl.when(code.str.len_chars() > position)
        .then(code.str.slice(0, position) + "." + code.str.slice(position))
        .otherwise(code)
    )

def add_prefix(code: pl.Expr, icd_version: str, type: str) -> pl.Expr:
    """
    Adds a prefix (e.g., "ICD9CM/" or "ICD10CM/") to the ICD code based on the version and type.

    Args:
        code (pl.Expr): The ICD code column as a Polars expression.
        icd_version (str): The version of the ICD code ("9" or "10").
        type (str): The type of the ICD code (e.g., "diagnosis" or "procedure").

    Returns:
        pl.Expr: The ICD code with the added prefix.
    """

    if icd_version == "9":
        code = pl.when(type == "diagnosis").then("ICD9CM/" + code).otherwise("ICD9Proc/" + code)
    elif icd_version == "10":
         code = pl.when(type == "diagnosis").then("ICD10CM/" + code).otherwise("ICD10PCS/" + code)

    return code

def format_diagnosis_code(icd_version: str, icd_code: pl.Expr) -> pl.Expr:
    # Consistent zero padding for ICD9
    icd_code = (
        pl.when(icd_code.str.len_chars() == 3).then(icd_code + "00")
        .when(icd_code.str.len_chars() == 4).then(icd_code + "0")
        .otherwise(icd_code)
    )

    icd9_code = (
        pl.when(icd_code.str.starts_with("E")).then(add_dot(icd_code, 4)).otherwise(add_dot(icd_code, 3))
    )

    icd10_code = add_dot(icd_code, 3)

    icd9_code = add_prefix(icd9_code, "9", "diagnosis")
    icd10_code = add_prefix(icd10_code, "10", "diagnosis")

    return pl.when(icd_version == "9").then(icd9_code).otherwise(icd10_code)

def format_procedure_code(icd_version: str, icd_code: pl.Expr) -> pl.Expr:
    icd_code = (
        pl.when(icd_code.str.len_chars() == 3).then(icd_code + "0")
        .otherwise(icd_code)
    )

    icd9_code = add_dot(icd_code, 2)
    icd10_code = icd_code

    icd9_code = add_prefix(icd9_code, "9", "procedure")
    icd10_code = add_prefix(icd10_code, "10", "procedure")

    return pl.when(icd_version == "9").then(icd9_code).otherwise(icd10_code)

def try_loading_gem_file(target_omop_vocabulary: str) -> pl.LazyFrame:
    """ Extract the specific GEM conversion files from cms.gov depending on vocabulary.
    Combine into a single parquet file and output a dataframe.

    Args:
        gem_cache_dir: The directory to cache the GEM files.
        target_omop_vocabularies: A tuple of vocabulary keys to extract.

    Raises:
        ValueError: If any item in target_omop_vocabularies is not a valid key in file_mappings.

    Returns: the dataframe containing mappings between source and target vocabularies

        Examples:
        >>> import os
        >>> import tempfile
        >>> import polars as pl
        >>> from pathlib import Path

        >>> target_vocab = Vocabulary("ICD12", ["ICD12CM"])
        >>> gem_df = try_loading_gem_file(target_vocab.omop_vocabularies)

        >>> target_vocab = Vocabulary("ICD10", ["ICD10CM"])
        >>> gem_df = try_loading_gem_file(target_vocab.omop_vocabularies)
        >>> print(gem_df.collect().head())
    """

    if target_omop_vocabulary == 'ICD10':
        target_omop_vocabularies = ["ICD10CM", "ICD10PCS"]
    elif target_omop_vocabulary == 'ICD9':
        target_omop_vocabularies = ["ICD9CM", "ICD9PCS"]
    
    file_mappings = {
        "ICD9CM": "2018_I10gem.txt", 
        "ICD9PCS": "gem_pcsi9.txt",
        "ICD10CM": "2018_I9gem.txt",
        "ICD10PCS": "gem_i9pcs.txt"
    }

    function_mappings = {
        "ICD9CM": format_diagnosis_code, 
        "ICD9PCS": format_procedure_code,
        "ICD10CM": format_diagnosis_code,
        "ICD10PCS": format_procedure_code,
    }
    
    invalid_keys = set(target_omop_vocabularies) - set(file_mappings.keys())
    if invalid_keys:
        raise ValueError(f"Invalid vocabulary: {invalid_keys}."
                         f"Allowed target vocabularies are: {list(file_mappings.keys())}.")
    
    # Specify location of ZIP files
    zip_urls = ["https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-cm-general-equivalence-mappings.zip",
                "https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-pcs-general-equivalence-mappings.zip"]
    
    all_data = []

    # Loop through each URL in zip_urls
    for url in zip_urls:
        zip_response = requests.get(url)
        zip_response.raise_for_status()

        # Extract the ZIP file contents
        with ZipFile(io.BytesIO(zip_response.content)) as zip_file:
            # List all files in the zip
            file_names = zip_file.namelist()

            # Extract and save only the files in target_omop_vocabularies
            for vocab in target_omop_vocabularies:
                file_name = file_mappings[vocab]
                if file_name in file_names:
                    # Extract the relevant file from the zip
                    with zip_file.open(file_name) as file:
                        file_content = file.read().decode("utf-8")
                        rows = [re.split(r'\s+', line.strip()) for line in file_content.splitlines()]
                        df = pl.DataFrame(rows, schema=["source_code", "target_code", "flags"], orient="row")

                        if vocab.startswith('ICD10'):
                            df = (df
                            .with_columns(function_mappings[vocab]("9", pl.col("source_code")).alias("source_code"))
                            .with_columns(function_mappings[vocab]("10", pl.col("target_code")).alias("target_code"))          
                            .drop("flags"))
                        elif vocab.startswith('ICD9'):
                            df = (df
                            .with_columns(function_mappings[vocab]("10", pl.col("source_code")).alias("source_code"))
                            .with_columns(function_mappings[vocab]("9", pl.col("target_code")).alias("target_code"))          
                            .drop("flags"))

                        all_data.append(df)

    # Combine all the data into one DataFrame (if there are multiple vocabularies)
    lazy_df = pl.concat(all_data).lazy()
    
    return lazy_df

def map_stochastically(
    gem_mapping: pl.DataFrame | pl.LazyFrame,
    prevalence_df: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Creates a mapping of concept codes from source to target based on prevalences.

    Args:
        gem_mapping (pl.LazyFrame):
            The GEM conversion file in lazy format, containing mappings from source code to target code.

        prevalence (pl.LazyFrame):
            The prevalence of each concept code in lazy format.

    Returns:
        pl.LazyFrame:
            Returns a dictionary mapping source codes to target codes based on mappings defined in GEM file. 
            One-to-many mappings are resolved using provided data prevalence, where the highest prevalence code is selected. 

    Examples:
        >>> import polars as pl
        >>> gem_mapping = pl.DataFrame({
        ...     "source_code": ["A", "A", "B", "B", "C"],
        ...     "target_code": ["X", "Y", "Z", "W", "U"]
        ... }).lazy()
        >>> prevalence = pl.DataFrame({
        ...     "concept_code": ["X", "Y", "Z", "W", "U"],
        ...     "prevalence": [0.7, 0.2, 0.1, 0.3, 1]
        ... }).lazy()
        >>> result = map_stochastically(gem_mapping, prevalence)
        >>> result.collect().sort(by="source_code")
        >>> print(result.head())

        >>> prevalence = pl.DataFrame({
        ...     "concept_code": ["X", "Y", "U"],
        ...     "prevalence": [0.7, 0.2, 1]
        ... }).lazy()
        >>> result = map_stochastically(gem_mapping, prevalence)
        >>> result.collect().sort(by="source_code")
        >>> print(result.head())
    """

    prevalence_df = prevalence_df.rename({"concept_code": "target_code"})

    # Merge prevalence to map each target_code to its prevalence
    source_to_target_mappings = gem_mapping.join(
        prevalence_df,
        on="target_code",
        how="left"
    )

    source_to_target_mappings = source_to_target_mappings.group_by("source_code").agg(
        pl.col("target_code").alias("target_codes"), 
        pl.col("prevalence").alias("prevalences")
    )

    def sample_target_codes(row: dict) -> str:
        """
        Selects a target code from a list of codes based on their associated prevalences.

        This function operates on a dictionary containing:
        - "target_codes": a list of target codes (strings).
        - "prevalences": a list of corresponding prevalences (floats), which may contain None or NaN values.

        The selection logic is as follows:
        1. If there is only one target code, it is returned directly.
        2. If all prevalences are zero (or missing/NaN), the first target code is returned
        after sorting the codes lexicographically.
        3. Otherwise, the target code with the highest prevalence is returned.

        Args:
            row (dict): A dictionary with the following keys:
                - "target_codes" (list of str): List of target codes.
                - "prevalences" (list of float): List of prevalences for each code.

        Returns:
            str: The selected target code.
        """
        target_codes = row["target_codes"]
        prevalences = row["prevalences"]

        # Case 1: If only one target code, return it deterministically
        if len(target_codes) == 1:
            return target_codes[0]

        prevalences = [
            float(p) if p is not None and not (isinstance(p, float) and p != p) else 0.0
            for p in prevalences
        ]
        
        # Case 2: If all prevalences are zero after imputation, return the first target code (or a fallback)
        if sum(prevalences) == 0:
            return sorted(target_codes)[0]

        # Case 3: Return the target code with the highest prevalence
        max_index = prevalences.index(max(prevalences))
        return target_codes[max_index]
    
    result = source_to_target_mappings.with_columns(
        pl.struct(pl.all())
        .map_elements(sample_target_codes, return_dtype=pl.String).alias("selected_target_code")
    ).drop('prevalences').drop('target_codes')

    return result.lazy()
