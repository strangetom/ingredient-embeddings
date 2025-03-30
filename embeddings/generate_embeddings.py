#!/usr/bin/env/python3

import argparse
import sys
import tempfile
from pathlib import Path

import floret

from embeddings.data import load_recipes, download_recipenlg_dataset, tokenize_recipes
from embeddings.generate_bigrams import train_bigram_model_nouns


def generate_embeddings(args: argparse.Namespace):
    if not args.source and not args.training:
        raise ValueError("Supply either the source file or training file.")

    if not args.training:
        # If source file doesn't exist, download it
        source_path = Path(args.source)
        if not source_path.is_file():
            download_recipenlg_dataset(args.source)

        recipes = load_recipes(args.source)
        tokenized_recipes = tokenize_recipes(recipes)

        train_bigram_model_nouns(tokenized_recipes, 0.0001)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for recipe in tokenized_recipes:
                ingredients = [
                    ingred
                    for ingredient in recipe.ingredients
                    for ingred in ingredient
                    if ingred
                ]
                instructions = [
                    instruct
                    for instruction in recipe.instructions
                    for instruct in instruction
                    if instruct
                ]
                f.write(" ".join(ingredients + instructions))
                f.write("\n")

            training_file = f.name
    else:
        training_file = args.training

    if args.preprocess:
        # If only preprocess, print location of training file and exit.
        print(f"Preprocessed sentences saved to {training_file}")
        sys.exit(0)

    model = floret.train_unsupervised(
        training_file,
        mode="floret",  # more size/memory efficient
        model="cbow",  # model type, skipgram or cbow
        ws=5,  # window size
        minn=2,  # smallest subtoken n-grams to generate
        maxn=5,  # largest subtoken n-grams to generate
        minCount=3,  # only include tokens that occur at least this many times
        dim=300,  # model dimensions
        epoch=30,  # training epochs
        lr=0.01,  # learning rate, between 0 and 1
        wordNgrams=3,  # length of word n-grams
        bucket=50000,
        hashCount=2,
    )
    if not args.model:
        dim = model.get_dimension()
        model_name = f"ingredient_embeddings.{dim}d.floret.bin"
    else:
        model_name = args.model

    model.save_model(model_name)
