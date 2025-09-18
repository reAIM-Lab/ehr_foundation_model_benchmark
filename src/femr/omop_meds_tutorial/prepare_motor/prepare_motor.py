import femr.ontology
import pathlib
import meds_reader
import pickle
import femr.splits
from femr.models.tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
import femr.models.tasks
import femr.models.processor
import pandas as pd
import polars as pl
import logging
from femr.models.tasks.motor import MOTOR_Task as MOTOR_Task
from femr.models.tasks.tpp import TPP_Task as TPP_Task
from femr.models.tasks.mtpp import MTPP_Task as MTPP_Task

# logging.basicConfig(
#     level=logging.INFO,
#     filename='motor.log',
#     filemode='w',
# )

def main(args):
    pretraining_data_path = pathlib.Path(args.pretraining_data)
    meds_reader_path = pathlib.Path(args.meds_reader)
    subject_splits_path = meds_reader_path / "metadata/subject_splits.parquet"
    code_metadata_path = meds_reader_path / "metadata/codes.parquet"
    codes_to_skip = []
    if args.motor_codes_to_skip:
        codes_to_skip = pl.read_parquet(args.motor_codes_to_skip)
        codes_to_skip = codes_to_skip["code"].to_list()


    with meds_reader.SubjectDatabase(str(meds_reader_path), num_threads=args.num_threads) as database:
        subject_ids = [_ for _ in database]
        print(f"subject_ids length: {len(subject_ids)}")
        ontology_path = pretraining_data_path / 'ontology.pkl'
        print(f"ontology_path: {ontology_path}")
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
            print("Loading ontology")
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
            print("Loading tokenizer")
            tokenizer = HierarchicalTokenizer.from_pretrained(tokenizer_path, ontology=ontology)

        task_path = pretraining_data_path / f'{args.loss_type}_task.pkl'

        if not task_path.exists():
            # Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit
            print(f"Train {args.loss_type} task")

            if args.loss_type == "motor":
                task = MOTOR_Task.fit_pretraining_task_info(
                    main_database, tokenizer,
                    num_tasks=8 * 1024,
                    num_bins=args.num_bins,
                    final_layer_size=512,
                    codes_to_skip=codes_to_skip
                )
            elif args.loss_type == "tpp":
                task = TPP_Task.fit_pretraining_task_info(
                    main_database, tokenizer,
                    num_tasks=8 * 1024,
                    num_bins=args.num_bins,
                    final_layer_size=512,
                    codes_to_skip=codes_to_skip
                )
            if args.loss_type == "mtpp":
                task = MTPP_Task.fit_pretraining_task_info(
                    main_database, tokenizer,
                    num_tasks=8 * 1024,
                    num_bins=args.num_bins,
                    final_layer_size=512,
                    codes_to_skip=codes_to_skip
                )

            with open(task_path, 'wb') as f:
                pickle.dump(task, f)

        else:
            with open(task_path, 'rb') as f:
                task = pickle.load(f)

        # print(f"Motor task length: {len(motor_task.pretraining_task_codes)}")
        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task)

        # example_subject_id = list(train_database)[0]
        # example_subject = train_database[example_subject_id]

        # We can do this one subject at a time
        # print("Convert a single subject")
        # example_batch = processor.collate([processor.convert_subject(example_subject, tensor_type='pt')])
        # print(f"example subject_id {example_subject_id} , example_subject: {example_subject}")
        # print(f"example_batch keys: {example_batch['batch'].keys()}")
        # for key in example_batch['batch'].keys():
        #     print(f"example_batch[{key}], shape: {example_batch['batch'][key].shape}, values: {example_batch['batch'][key]}")

        # train_batches_path = pretraining_data_path / 'train_batches'

        # if not train_batches_path.exists():
        #     print("Convert batches")
        #     # But generally we want to convert entire datasets
        #     train_batches = processor.convert_dataset(train_database, tokens_per_batch=args.tokens_per_batch, num_proc=64)

        #     print("Convert batches to pytorch")
        #     # Convert our batches to pytorch tensors
        #     train_batches.set_format("pt")
        #     train_batches.save_to_disk(train_batches_path)

        # val_batches_path = pretraining_data_path / 'val_batches'

        # if not val_batches_path.exists():
        #     print("Convert val batches")
        #     val_batches = processor.convert_dataset(val_database, tokens_per_batch=args.tokens_per_batch, num_proc=64)
        #     # Convert our batches to pytorch tensors
        #     val_batches.set_format("pt")
        #     val_batches.save_to_disk(val_batches_path)


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
    parser.add_argument(
        "--num_bins",
        dest="num_bins",
        action="store",
        required=False,
        type=int,
        default=8,
    )
    parser.add_argument(
        "--loss_type",
        dest="loss_type",
        action="store",
        required=True,
        choices=["motor","tpp","mtpp"],
        type=str,
        default=8,
    )
    
    

    return parser


if __name__ == "__main__":
    main(create_omop_meds_tutorial_argparser().parse_args())

'''
mimic
gsb
python prepare_motor.py \
  --pretraining_data /user/zj2398/cache/motor_mimic_8k \
  --athena_path " " \
  --num_bins 8 \
  --loss_type motor \
  --num_threads 100 \
  --meds_reader /user/zj2398/cache/mimic/meds_v0.6_reader 

#   > out.log 2>&1  

kuvira
python prepare_motor.py \
  --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8 \
  --athena_path " " \
  --num_bins 8 \
  --num_threads 64 \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader

cumc-kuvira
python prepare_motor.py \
  --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/cumc/motor_cumc_bin_8 \
  --athena_path " " \
  --num_bins 8 \
  --num_threads 16 \
  --meds_reader /data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader
'''