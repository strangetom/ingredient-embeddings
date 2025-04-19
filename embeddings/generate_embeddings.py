#!/usr/bin/env/python3

import argparse
import concurrent.futures as cf
import gzip
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from embeddings.bigrams import BigramModel
from embeddings.data import (
    TokenizedRecipe,
    chunked,
    load_recipes,
    download_recipenlg_dataset,
    tokenize_recipes,
)
from embeddings.glove import VocabCount, Cooccur, Shuffle, GloVe


def join_bigrams_in_recipes(
    recipes: list[TokenizedRecipe], bm: BigramModel | None
) -> list[str]:
    """Use bigram model to join bigram tokens in recipe ingredients and instructions.

    Each recipe is returned as a single str, created by joining the ingredient then
    instructions tokens with a space, after joining bigrams with an underscore.

    Parameters
    ----------
    recipes : list[TokenizedRecipe]
        List of TokenizedRecipes to join bigrams for.
    b : BigramModel | None
        Bigram model, or None if not using bigrams

    Returns
    -------
    list[str]
        List of recipe strings.
    """
    joined_recipes = []
    for recipe in recipes:
        if bm:
            ingredients = [
                ingred
                for ingredient in recipe.ingredients
                for ingred in bm.join_bigrams(ingredient)
                if ingred
            ]
            instructions = [
                instruct
                for instruction in recipe.instructions
                for instruct in bm.join_bigrams(instruction)
                if instruct
            ]
        else:
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

        joined_recipes.append(" ".join(ingredients + instructions))
    return joined_recipes


def flatten_recipes(
    tokenize_recipes: list[TokenizedRecipe], bigrams_file: str | None
) -> list[str]:
    """Flatten recipes by joining bigram tokens with an underscore, then joining all
    tokens with a space into a single string.

    Parameters
    ----------
    tokenize_recipes : list[TokenizedRecipe]
        Description
    bigrams_file : str | None
        Path to bigrams file, or None to not apply bigrams.

    Returns
    -------
    list[str]
        List of flattened recipes.
    """
    # Chunk data into 100 groups to process in parallel.
    n_chunks = 100
    # Define chunk size so all groups have about the same number of elements, except the
    # last group which will be slightly smaller.
    chunk_size = int((len(tokenize_recipes) + n_chunks) / n_chunks)
    chunks = chunked(tokenize_recipes, chunk_size)

    bm = None
    if bigrams_file:
        bm = BigramModel("bigrams.csv")

    flattened_recipes = []
    print("Flattening recipes...")
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(join_bigrams_in_recipes, c, bm) for c in chunks]
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            flattened_recipes.extend(future.result())

    return flattened_recipes


def compress_file(path: str):
    """Compress file using gzip.

    Compressed file as ".gz" appended to end of file name.

    Parameters
    ----------
    path : str
        Path to file to compress.
    """
    with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
        dst.writelines(src)


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
        flattened_recipes = flatten_recipes(tokenized_recipes, args.bigrams)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for recipe in flattened_recipes:
                f.write(recipe)
                f.write("\n")

            training_file = f.name
    else:
        training_file = args.training

    if args.preprocess:
        # If only preprocess, print location of training file and exit.
        print(f"Preprocessed sentences saved to {training_file}")
        sys.exit(0)

    vocab = VocabCount.run(training_file, verbose=2, min_count=5)
    cooccur = Cooccur.run(
        training_file,
        verbose=2,
        symmetric=1,
        window_size=15,
        vocab_file=vocab,
        memory=32,
    )
    shuff = Shuffle.run(cooccur, verbose=2, memory=32)
    embeddings = GloVe.run(
        input_file=shuff,
        vocab_file=vocab,
        verbose=2,
        write_header=1,
        iter=100,
        binary=2,
        vector_size=25,
        save_file=args.model,
    )
    compress_file(embeddings + ".txt")
