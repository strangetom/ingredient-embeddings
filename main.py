#!/usr/bin/env/python3

import argparse

from embeddings.generate_bigrams import generate_bigrams
from embeddings.generate_embeddings import generate_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train word embeddings for recipe ingredients."
    )
    parser.add_argument(
        "--train",
        help="Model to train: embeddings, bigrams.",
        choices=["embeddings", "bigrams"],
        required=True,
    )
    parser.add_argument(
        "--source",
        help="Path to RecipeNLG dataset csv.",
        type=str,
        dest="source",
        default=None,
    )
    parser.add_argument(
        "--training-file",
        help="Path to text file of preprocessed recipes for model training.",
        type=str,
        dest="training",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Path to save embeddings model to.",
        type=str,
        dest="model",
        default=None,
    )
    parser.add_argument(
        "--preprocess-only",
        help="Perform preprocessing steps without training embeddings model.",
        action="store_true",
        dest="preprocess",
    )
    args = parser.parse_args()

    if args.train == "embeddings":
        generate_embeddings(args)
    elif args.train == "bigrams":
        generate_bigrams(args)
