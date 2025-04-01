#!/usr/bin/env/python3

import argparse

from embeddings.generate_bigrams import generate_bigrams
from embeddings.generate_embeddings import generate_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train word embeddings and bigrams for recipe ingredients."
    )
    subparsers = parser.add_subparsers(dest="command", help="Training commands")

    train_parser = subparsers.add_parser("embeddings", help="Train word embeddings.")
    bigram_parser = subparsers.add_parser("bigrams", help="Train word bigrams.")

    train_parser.add_argument(
        "--source",
        help="Path to RecipeNLG dataset csv.",
        type=str,
        dest="source",
        default=None,
    )
    train_parser.add_argument(
        "--training-file",
        help="Path to text file of preprocessed recipes for model training.",
        type=str,
        dest="training",
        default=None,
    )
    train_parser.add_argument(
        "--model",
        help="Path to save embeddings model to.",
        type=str,
        dest="model",
        default=None,
    )
    train_parser.add_argument(
        "--bigrams",
        help="Path to bigrams CSV file.",
        type=str,
        dest="bigrams",
        default=None,
    )
    train_parser.add_argument(
        "--preprocess-only",
        help="Perform preprocessing steps without training embeddings model.",
        action="store_true",
        dest="preprocess",
    )

    bigram_parser.add_argument(
        "--source",
        help="Path to RecipeNLG dataset csv.",
        type=str,
        dest="source",
        default=None,
    )
    bigram_parser.add_argument(
        "--output",
        help="Path to save bigrams to.",
        type=str,
        dest="output",
        default="bigrams.csv",
    )
    args = parser.parse_args()

    if args.command == "embeddings":
        generate_embeddings(args)
    elif args.command == "bigrams":
        generate_bigrams(args)
