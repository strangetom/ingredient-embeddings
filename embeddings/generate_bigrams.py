#!/usr/bin/env/python3

import argparse
import json
from collections import Counter
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

import nltk

from embeddings.data import (
    TokenizedRecipe,
    load_recipes,
    download_recipenlg_dataset,
    tokenize_recipes,
    stem,
)


@lru_cache
def load_units_list() -> list[str]:
    """Load list of unit names from file.

    Returns
    -------
    list[str]
        List of unit names.
    """
    with as_file(files(__package__) / "units.json") as p:
        with open(p, "r") as f:
            units = json.load(f)

    return [stem(u) for u in units]


@lru_cache
def load_tools_list() -> list[str]:
    """Load list of tools names from file.

    Returns
    -------
    list[str]
        List of tools names.
    """
    with as_file(files(__package__) / "tools.json") as p:
        with open(p, "r") as f:
            tools = json.load(f)

    return [stem(t) for t in tools]


def generate_bigrams(args: argparse.Namespace):
    if not args.source:
        raise ValueError("Supply source file.")

    # If source file doesn't exist, download it
    source_path = Path(args.source)
    if not source_path.is_file():
        download_recipenlg_dataset(args.source)

    recipes = load_recipes(args.source)
    tokenized_recipes = tokenize_recipes(recipes)
    extract_bigrams(tokenized_recipes, 0.00001, args.output)


def extract_bigrams(
    tokenized_recipes: list[TokenizedRecipe], freq_filter: int | float, output: str
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
    units_tools = load_units_list() + load_tools_list()

    print("Identifying bigrams...")
    bigram_dist = []
    for recipe in tokenized_recipes:
        for tokens, pos_tags in zip(recipe.instructions, recipe.instructions_pos):
            for (w1, p1), (w2, p2) in nltk.bigrams(zip(tokens, pos_tags)):
                if (
                    w1
                    and w2
                    and w1 != w2
                    and w1 not in units_tools
                    and w2 not in units_tools
                    and p1.startswith(("N", "J"))
                    and p2.startswith("N")
                ):
                    bigram_dist.append((w1, w2))

    bigram_fdist = Counter(bigram_dist)

    if isinstance(freq_filter, float):
        freq_filter = int(len(bigram_dist) * freq_filter)

    filtered_bigram_fdist = {
        key: value for key, value in bigram_fdist.items() if value >= freq_filter
    }

    print("Bigrams saved to bigrams.csv")
    with open(output, "w") as f:
        # Write out bigrams to csv in order of most to least common
        for bigram, freq in sorted(
            filtered_bigram_fdist.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(",".join(bigram) + "\n")
