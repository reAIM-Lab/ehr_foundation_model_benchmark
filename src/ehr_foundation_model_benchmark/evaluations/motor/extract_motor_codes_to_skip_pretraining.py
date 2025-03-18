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

    icd_codes_to_skip = codes_to_remove.filter(
        pl.col("Source") != "SNOMED"
    )

    icd_prefixes = icd_codes_to_skip.collect()["Code"].to_list()
    pattern = "^(?:" + "|".join(re.escape(prefix) for prefix in icd_prefixes) + ")"
    icd_codes_to_skip = concept.filter(
        pl.col("concept_code").str.contains(pattern) & pl.col("vocabulary_id").str.starts_with("ICD10")
    ).select(
        pl.col("concept_id"),
        code_col
    )

    # Get the corresponding concept_id from the concept table
    snomed_codes_to_skip = codes_to_remove.filter(
        pl.col("Source") == "SNOMED"
    ).join(
        concept,
        left_on=["Source", "Code"],
        right_on=["vocabulary_id", "concept_id"],
    ).select(
        pl.col("Code").alias("non_standard_concept_id")
    )

    # Get the corresponding standard concept codes
    snomed_codes_to_skip = snomed_codes_to_skip.join(
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
    snomed_codes_to_skip = snomed_codes_to_skip.join(
        ancestor,
        left_on=["standard_concept_id"],
        right_on=["ancestor_concept_id"],
    )
    # Map the descendant through concept_relationship to get the source codes
    snomed_codes_to_skip = snomed_codes_to_skip.join(
        relationship,
        left_on=["descendant_concept_id"],
        right_on=["concept_id_2"],
    ).filter(
        pl.col("relationship_id") == "Maps to"
    )

    # Map the source code for the source concept_id
    snomed_codes_to_skip = snomed_codes_to_skip.join(
        concept,
        left_on=["concept_id_1"],
        right_on=["concept_id"],
    ).select(
        pl.col("concept_id_1").alias("concept_id"),
        code_col
    )
    motor_codes_to_skip = pl.concat([snomed_codes_to_skip, icd_codes_to_skip]).collect()
    motor_codes_to_skip.write_parquet(args.motor_codes_to_skip)

    metadata_codes = pl.read_parquet(pathlib.Path(args.meds_reader) / "metadata" / "codes.parquet")
    motor_codes_to_skip_from_pretraining = motor_codes_to_skip["code"].to_list()
    n_somed_codes = [c for c in motor_codes_to_skip_from_pretraining if c.startswith("SNOMED")]
    n_snomed_codes_to_skip = metadata_codes.filter(pl.col("code").is_in(n_somed_codes))
    n_icd_codes = [c for c in motor_codes_to_skip_from_pretraining if c.startswith("ICD")]
    n_icd_codes_to_skip = metadata_codes.filter(pl.col("code").is_in(n_icd_codes))
    n_non_snomed_icd_codes =  [
        c for c in motor_codes_to_skip_from_pretraining
        if not c.startswith("ICD") and  not c.startswith("SNOMED")
    ]
    n_non_snomed_icd_codes_to_skip = metadata_codes.filter(pl.col("code").is_in(n_non_snomed_icd_codes))
    print(f"Number of SNOMED codes found in total: {len(n_somed_codes)}")
    print(f"Number of SNOMED codes to skip from motor pretraining: {len(n_snomed_codes_to_skip)}")
    print(f"Number of ICD codes found in total: {len(n_icd_codes)}")
    print(f"Number of ICD codes to skip from motor pretraining: {len(n_icd_codes_to_skip)}")
    print(f"Number of other codes found in total: {len(n_non_snomed_icd_codes)}")
    print(f"Number of other codes to skip from motor pretraining: {len(n_non_snomed_icd_codes_to_skip)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for creating the codes to remove")
    parser.add_argument(
        "--meds_reader",
        dest="meds_reader",
        action="store",
        required=True,
    )
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
        "--motor_codes_to_skip",
        dest="motor_codes_to_skip",
        action="store",
        required=True,
    )
    main(parser.parse_args())
