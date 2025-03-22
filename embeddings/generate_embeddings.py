#!/usr/bin/env/python3

import argparse
import concurrent.futures as cf
import csv
import json
import sys
import tempfile
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Iterable

import floret
from tqdm import tqdm

from embeddings.download import download_recipenlg_dataset
from embeddings.preprocess import preprocess_recipes


def chunked(iterable: Iterable, n: int) -> Iterable:
    """Break *iterable* into lists of length *n*:

    >>> list(chunked([1, 2, 3, 4, 5, 6], 3))
    [[1, 2, 3], [4, 5, 6]]

    By the default, the last yielded list will have fewer than *n* elements
    if the length of *iterable* is not divisible by *n*:

    >>> list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3))
    [[1, 2, 3], [4, 5, 6], [7, 8]]

    Parameters
    ----------
    iterable : Iterable
        Iterable to chunk.
    n : int
        Size of each chunk.

    Returns
    -------
    Iterable
        Chunks of iterable with size n (or less for the last chunk).
    """

    def take(n, iterable):
        "Return first n items of the iterable as a list."
        return list(islice(iterable, n))

    return iter(partial(take, n, iter(iterable)), [])


def generate_embeddings(args: argparse.Namespace):
    if not args.source and not args.training:
        raise ValueError("Supply either the source file or training file.")

    if not args.training:
        # If source file doesn't exist, download it
        source_path = Path(args.source)
        if not source_path.is_file():
            download_recipenlg_dataset(args.source)

        recipes = []
        print("Loading recipes...")
        with open(args.source, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                recipes.append(
                    " ".join(
                        json.loads(row["ingredients"]) + json.loads(row["directions"])
                    )
                )

        # Chunk data into 4 groups to process in parallel.
        n_chunks = 100
        # Define chunk size so all groups have about the same number of elements, except the
        # last group which will be slightly smaller.
        chunk_size = int((len(recipes) + n_chunks) / n_chunks)
        chunks = chunked(recipes, chunk_size)

        preprocessed_recipes = []
        print("Preprocessing recipes...")
        with cf.ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(preprocess_recipes, c) for c in chunks]
            for future in tqdm(cf.as_completed(futures), total=len(futures)):
                preprocessed_recipes.extend(future.result())

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for recipe in preprocessed_recipes:
                f.write(recipe)
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
        model="cbow",   # model type, skipgram or cbow
        ws=5,           # window size
        minn=2,         # smallest subtoken n-grams to generate
        maxn=5,         # largest subtoken n-grams to generate
        minCount=3,     # only include tokens that occur at least this many times
        dim=10,         # model dimensions
        epoch=30,       # training epochs
        lr=0.01,        # learning rate, between 0 and 1
        wordNgrams=3,   # length of word n-grams
        bucket=50000,
        hashCount=2,
    )
    if not args.model:
        dim = model.get_dimension()
        model_name = f"ingredient_embeddings.{dim}d.floret.bin"
    else:
        model_name = args.model

    model.save_model(model_name)
