#!/usr/bin/env/python3

import argparse
from collections import Counter
from pathlib import Path

import nltk

from embeddings.data import (
    TokenizedRecipe,
    load_recipes,
    download_recipenlg_dataset,
    tokenize_recipes,
)


def generate_bigrams(args: argparse.Namespace):
    if not args.source and not args.training:
        raise ValueError("Supply either the source file or training file.")

    if not args.training:
        # If source file doesn't exist, download it
        source_path = Path(args.source)
        if not source_path.is_file():
            download_recipenlg_dataset(args.source)

        recipes = load_recipes(args.source)
        tokenized_recipes = tokenize_recipes(recipes)
        train_bigram_model_nouns(tokenized_recipes, 0.00005)


def train_bigram_model_nouns(
    tokenized_recipes: list[TokenizedRecipe], freq_filter: int | float
) -> None:
    """Identify common bigrams in training data.

    Parameters
    ----------
    tokenized_recipes : list[TokenizedRecipe]
        List of TokenizedRecipes recipes to identify bigrams from.
    freq_filter : int | float
        Minimum frequency of bigrams to keep.
        If integer, this refers to the absolute count.
        If float, this refers the fraction of total bigrams.
    """
    print("Identifying bigrams...")
    bigram_dist = []
    for recipe in tokenized_recipes:
        for tokens, pos_tags in zip(recipe.instructions, recipe.instructions_pos):
            for (w1, p1), (w2, p2) in nltk.bigrams(zip(tokens, pos_tags)):
                if w1 and w2 and w1 != w2 and p1.startswith("N") and p2.startswith("N"):
                    bigram_dist.append((w1, w2))

    bigram_fdist = Counter(bigram_dist)

    if isinstance(freq_filter, float):
        freq_filter = int(len(bigram_dist) * freq_filter)

    filtered_bigram_fdist = {
        key: value for key, value in bigram_fdist.items() if value >= freq_filter
    }

    print("Bigrams saved to bigrams.csv")
    with open("bigrams.csv", "w") as f:
        # Write out bigrams to csv in order of most to least common
        for bigram, freq in sorted(
            filtered_bigram_fdist.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(",".join(bigram) + "\n")
