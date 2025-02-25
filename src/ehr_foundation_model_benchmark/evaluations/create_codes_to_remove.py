import re
import pathlib
import polars as pl


def main(args):
    concept = pl.scan_csv(
        pathlib.Path(args.athena_path) / "CONCEPT.csv", separator="\t", infer_schema_length=0
    )
    ancestor = pl.scan_csv(
        pathlib.Path(args.athena_path) / "CONCEPT_ANCESTOR.csv", separator="\t", infer_schema_length=0
    )
    relationship = pl.scan_csv(
        pathlib.Path(args.athena_path) / "CONCEPT_RELATIONSHIP.csv", separator="\t", infer_schema_length=0
    )
    codes_to_remove = pl.scan_csv(args.disease_codes_to_remove, separator=",")
    code_col = (pl.col("vocabulary_id") + "/" + pl.col("concept_code")).alias("code")

    icd_codes_to_remove = codes_to_remove.filter(
        pl.col("Source") != "SNOMED"
    )

    icd_prefixes = icd_codes_to_remove.collect()["Code"].to_list()
    pattern = "^(?:" + "|".join(re.escape(prefix) for prefix in icd_prefixes) + ")"
    icd_codes_to_remove = concept.filter(
        pl.col("concept_code").str.contains(pattern) & pl.col("vocabulary_id").str.starts_with("ICD10")
    ).select(
        pl.col("concept_id"),
        code_col
    )

    # Get the corresponding concept_id from the concept table
    snomed_codes_to_remove = codes_to_remove.filter(
        pl.col("Source") == "SNOMED"
    ).join(
        concept,
        left_on=["Source", "Code"],
        right_on=["vocabulary_id", "concept_id"],
    ).select(
        pl.col("Code").alias("non_standard_concept_id")
    )

    # Get the corresponding standard concept codes
    snomed_codes_to_remove = snomed_codes_to_remove.join(
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
    snomed_codes_to_remove = snomed_codes_to_remove.join(
        ancestor,
        left_on=["standard_concept_id"],
        right_on=["ancestor_concept_id"],
    )
    # Map the descendant through concept_relationship to get the source codes
    snomed_codes_to_remove = snomed_codes_to_remove.join(
        relationship,
        left_on=["descendant_concept_id"],
        right_on=["concept_id_2"],
    ).filter(
        pl.col("relationship_id") == "Maps to"
    )

    # Map the source code for the source concept_id
    snomed_codes_to_remove = snomed_codes_to_remove.join(
        concept,
        left_on=["concept_id_1"],
        right_on=["concept_id"],
    ).select(
        pl.col("concept_id_1").alias("concept_id"),
        code_col
    )

    pl.concat([snomed_codes_to_remove, icd_codes_to_remove]).collect().write_parquet(args.codes_to_remove)


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
