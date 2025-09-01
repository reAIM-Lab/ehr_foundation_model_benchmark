import femr.ontology
import pathlib
import meds_reader
import pickle
import femr.splits
from femr.models.tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
import femr.models.tasks_old
import femr.models.processor_old
import pandas as pd
import polars as pl


def main(args):
    pretraining_data_path = pathlib.Path(args.pretraining_data)
    meds_reader_path = pathlib.Path(args.meds_reader)
    subject_splits_path = meds_reader_path / "metadata/subject_splits.parquet"
    code_metadata_path = meds_reader_path / "metadata/codes.parquet"
    codes_to_skip = []
    if args.motor_codes_to_skip:
        codes_to_skip = pl.read_parquet(args.motor_codes_to_skip)
        codes_to_skip = codes_to_skip["code"].to_list()


    with meds_reader.SubjectDatabase(str(meds_reader_path), num_threads=32) as database:
        subject_ids = [_ for _ in database]
        ontology_path = pretraining_data_path / 'ontology.pkl'
        if not ontology_path.exists():
            print("Creating ontology")
            ontology = femr.ontology.Ontology(args.athena_path, code_metadata_path=str(code_metadata_path))
            print("Pruning the ontology")
            ontology.prune_to_dataset(
                database,
                prune_all_descriptions=True,
                remove_ontologies={'SPL', 'HemOnc', 'LOINC'}
            )

            with open(ontology_path, 'wb') as f:
                pickle.dump(ontology, f)
        else:
            with open(ontology_path, 'rb') as f:
                ontology = pickle.load(f)

        subject_splits = pd.read_parquet(subject_splits_path)
        subject_splits = subject_splits[subject_splits.subject_id.isin(subject_ids)]
        train_tuning_split = subject_splits[~subject_splits.split.isin(["held_out"])].subject_id.tolist()
        test_split = subject_splits[subject_splits.split.isin(["held_out"])].subject_id.tolist()
        main_split = femr.splits.SubjectSplit(train_tuning_split, test_split)
        main_split.save_to_csv(str(pretraining_data_path / 'main_split.csv'))

        train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 17, frac_test=0.05)

        main_database = database.filter(main_split.train_subject_ids)
        train_database = main_database.filter(train_split.train_subject_ids)
        val_database = main_database.filter(train_split.test_subject_ids)

        tokenizer_path = pretraining_data_path / 'tokenizer'
        if not tokenizer_path.exists():
            print("Train tokenizer")
            tokenizer = HierarchicalTokenizer.train(
                main_database,
                vocab_size=1024 * 16,
                ontology=ontology
            )
            # Save the tokenizer to the same directory as the model
            tokenizer.save_pretrained(tokenizer_path)
        else:
            tokenizer = HierarchicalTokenizer.from_pretrained(tokenizer_path, ontology=ontology)

        task_path = pretraining_data_path / 'motor_task.pkl'

        if not task_path.exists():
            # Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit
            print("Train MOTOR task")

            motor_task = femr.models.tasks_old.MOTORTask.fit_pretraining_task_info(
                main_database, tokenizer,
                num_tasks=8 * 1024,
                num_bins=8,
                final_layer_size=512,
                codes_to_skip=codes_to_skip
            )

            with open(task_path, 'wb') as f:
                pickle.dump(motor_task, f)

        else:
            with open(task_path, 'rb') as f:
                motor_task = pickle.load(f)

        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

        # example_subject_id = list(train_database)[0]
        # example_subject = train_database[example_subject_id]

        # # We can do this one subject at a time
        # print("Convert a single subject")
        # example_batch = processor.collate([processor.convert_subject(example_subject, tensor_type='pt')])

        train_batches_path = pretraining_data_path / 'train_batches'

        if not train_batches_path.exists():
            print("Convert batches")
            # But generally we want to convert entire datasets
            train_batches = processor.convert_dataset(train_database, tokens_per_batch=args.tokens_per_batch, num_proc=64)

            print("Convert batches to pytorch")
            # Convert our batches to pytorch tensors
            train_batches.set_format("pt")
            train_batches.save_to_disk(train_batches_path)

        val_batches_path = pretraining_data_path / 'val_batches'

        if not val_batches_path.exists():
            print("Convert val batches")
            val_batches = processor.convert_dataset(val_database, tokens_per_batch=args.tokens_per_batch, num_proc=64)
            # Convert our batches to pytorch tensors
            val_batches.set_format("pt")
            val_batches.save_to_disk(val_batches_path)


def create_omop_meds_tutorial_argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for preparing Motor")
    parser.add_argument(
        "--pretraining_data",
        dest="pretraining_data",
        action="store",
        required=True,
    )
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
        "--motor_codes_to_skip",
        dest="motor_codes_to_skip",
        action="store",
        required=False,
    )
    parser.add_argument(
        "--num_threads",
        dest="num_threads",
        action="store",
        required=False,
        type=int,
        default=16,
    )
    parser.add_argument(
        "--tokens_per_batch",
        dest="tokens_per_batch",
        action="store",
        required=False,
        type=int,
        # this is decided based on the 99% percentile of the number of tokens
        default=8192,
    )
    return parser


if __name__ == "__main__":
    main(create_omop_meds_tutorial_argparser().parse_args())