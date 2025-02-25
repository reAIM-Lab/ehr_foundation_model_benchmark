import pathlib
import polars as pl


def main(args):
    concept = pl.scan_csv(
        pathlib.Path(args.athena_path) / "CONCEPT.CSV", separator="\t", infer_schema_length=0
    )
    ancestor = pl.scan_csv(
        pathlib.Path(args.athena_path) / "CONCEPT_ANCESTOR.CSV", separator="\t", infer_schema_length=0
    )
    relationship = pl.scan_csv(
        pathlib.Path(args.athena_path) / "CONCEPT_RELATIONSHIP.CSV", separator="\t", infer_schema_length=0
    )
    codes_to_remove = pl.scan_csv(args.disease_codes_to_remove, separator=",")
    code_col = pl.col("vocabulary_id") + "/" + pl.col("concept_code")

    icd_codes_to_remove = codes_to_remove.filter(
        pl.col("Source") != "SNOMED"
    )
    icd_codes_to_remove = icd_codes_to_remove.with_columns(
        pl.lit("ICD10CM").alias("Source")
    ).join(
        concept,
        left_on=["Source", "Code"],
        right_on=["vocabulary_id", "concept_code"],
    ).select(
        pl.col("concept_id").alias("non_standard_concept_id")
    )
    # Get the corresponding concept_id from the concept table
    snomed_codes_to_remove = codes_to_remove.filter(
        pl.col("Source") == "SNOMED"
    ).with_columns(pl.col("Code").cast(int)).join(
        concept,
        left_on=["Source", "Code"],
        right_on=["vocabulary_id", "concept_id"],
    ).select(
        pl.col("concept_id").alias("non_standard_concept_id")
    )
    all_codes_to_remove = pl.concat([icd_codes_to_remove, snomed_codes_to_remove])
    # Get the corresponding standard concept codes
    all_codes_to_remove = all_codes_to_remove.join(
        relationship,
        left_on=["non_standard_concept_id"],
        right_on=["concept_id_1"],
    ).filter(
        pl.col("relationship_id") == "Maps to"
    ).select(
        pl.col("non_standard_concept_id"),
        pl.col("concept_id_2").alias("standard_concept_id")
    )
    # Get all the descendant concepts
    all_codes_to_remove = all_codes_to_remove.join(
        ancestor,
        left_on=["standard_concept_id"],
        right_on=["ancestor_concept_id"],
    ).select(
        pl.col("descendant_concept_id")
    ).join(
        concept,
        left_on=["descendant_concept_id"],
        right_on=["concept_id"],
    ).select(
        pl.col("concept_id"),
        pl.col("vocabulary_id"),
        pl.col("concept_code"),
        code_col
    )
    all_codes_to_remove.sink_parquet(args.codes_to_remove)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for creating the codes to remove")
    parser.add_argument(
        "--athena_path",
        dest="athena_path",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--disease_codes_to_remove",
        dest="disease_codes_to_remove",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--codes_to_remove",
        dest="codes_to_remove",
        action="store",
        required=True,
    )
    main(parser.parse_args())
