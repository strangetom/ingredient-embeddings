#!/usr/bin/env/python3

import argparse
from pathlib import Path

from transformer_embeddings import fine_tune_embeddings, generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine tune mdbr-leaf-mt embeddings on recipes."
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    finetune_parser = subparsers.add_parser("finetune", help="Finetune embeddings.")
    dataset_parser = subparsers.add_parser(
        "dataset", help="Generate finetuning dataset."
    )

    finetune_parser.add_argument(
        "--dataset",
        help="Path to dataset directory.",
        type=Path,
        dest="dataset",
    )
    finetune_parser.add_argument(
        "--model",
        help="Path to save fine tuned model to.",
        type=Path,
        dest="model",
    )
    finetune_parser.add_argument(
        "--onnx",
        help="Path to save onnx exported model to.",
        type=Path,
        dest="onnx",
    )

    dataset_parser.add_argument(
        "--source",
        help="Path to recipes-en-201706 database.",
        type=Path,
        dest="source",
    )
    dataset_parser.add_argument(
        "--output",
        help="Path to save dataset to.",
        type=Path,
        dest="output",
    )
    args = parser.parse_args()

    if args.command == "dataset":
        generate_dataset(args.source, Path(""), args.output)
    elif args.command == "finetune":
        fine_tune_embeddings(args.dataset, args.model, args.onnx)
