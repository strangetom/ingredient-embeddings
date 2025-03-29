#!/usr/bin/env/python3

import argparse
import concurrent.futures as cf
import csv
import json
import string
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Iterable

import floret
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from embeddings.download import download_recipenlg_dataset
from embeddings.preprocess import preprocess_recipe, stem, tokenize

STOP_WORDS = stopwords.words("english")
ALLOWED_POS_TAGS = {
    "NN",
    "NNS",
    "NNP",
    "NNPS",
    "JJ",
    "JJR",
    "JJS",
    "RB",
    "RBR",
    "RBS",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "FW",
}


@dataclass
class Recipe:
    id_: int
    ingredients: str
    instructions: str

    def __post_init__(self):
        self.ingredients = preprocess_recipe(self.ingredients).lower()
        self.instructions = preprocess_recipe(self.instructions).lower()

    def ingredient_tokens(self) -> list[str]:
        return self._tokens(self.ingredients)

    def instruction_tokens(self) -> list[str]:
        return self._tokens(self.instructions)

    def _tokens(self, text: str) -> list[str]:
        return [
            stem(token)
            for token, pos in nltk.pos_tag(tokenize(text))
            if pos in ALLOWED_POS_TAGS
            and token not in string.punctuation
            and not token.isdigit()
            and not token.isnumeric()
            and not token.isspace()
            and token not in STOP_WORDS
            and len(token) > 1  # avoid leftover units e.g. 'c''
        ]


@dataclass
class TokenizedRecipe:
    id_: int
    ingredients: list[str]
    instructions: list[str]


def load_recipes(csv_file: str) -> list[Recipe]:
    """Load recipes from CSV file.

    Parameters
    ----------
    csv_file : str
        Path to CSV file to load recipes from.

    Returns
    -------
    list[Recipe]
        List of Recipe objects loaded from CSV.
    """
    recipes = []
    print("Loading recipes...")
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "cookbooks.com" in row["link"]:
                # Recipes from cookbooks seem to be all user submitted and of
                # extremely variable quality (including entries that are not
                # recipes at all), so exclude them all
                continue

            recipe = Recipe(
                id_=int(row[""]),
                ingredients=" ".join(json.loads(row["ingredients"])),
                instructions=" ".join(json.loads(row["directions"])),
            )
            recipes.append(recipe)

    return recipes


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


def get_recipes_tokens(recipes: list[Recipe]) -> list[TokenizedRecipe]:
    """Get tokens for recipe ingredients and instructions and return TokenizedRecipe.

    Parameters
    ----------
    recipes : list[Recipe]
        List of Recipes to get tokens for.

    Returns
    -------
    list[TokenizedRecipe]
    """
    return [
        TokenizedRecipe(
            id_=recipe.id_,
            ingredients=recipe.ingredient_tokens(),
            instructions=recipe.instruction_tokens(),
        )
        for recipe in recipes
    ]


def tokenize_recipes(recipes: list[Recipe]) -> list[TokenizedRecipe]:
    """Preprocess recipes to obtain their ingredient and instruction tokens.

    This is done in parallel because calling pos_tag repeatedly is slow.

    Parameters
    ----------
    recipes : list[Recipe]
        List of recipes.

    Returns
    -------
    list[TokenizedRecipe]
        List of tokenized recipes.
    """
    # Chunk data into 100 groups to process in parallel.
    n_chunks = 100
    # Define chunk size so all groups have about the same number of elements, except the
    # last group which will be slightly smaller.
    chunk_size = int((len(recipes) + n_chunks) / n_chunks)
    chunks = chunked(recipes, chunk_size)

    tokenized_recipes = []
    print("Preprocessing recipes...")
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_recipes_tokens, c) for c in chunks]
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            tokenized_recipes.extend(future.result())

    return tokenized_recipes


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

        print("Preparing for bigram model training...")
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        all_instructions = [
            token for recipe in tokenized_recipes for token in recipe.instructions
        ]
        print("Training bigram model...")
        finder = nltk.collocations.BigramCollocationFinder.from_words(all_instructions)
        with open("bigrams.csv", "w") as f:
            for bigram in finder.above_score(bigram_measures.raw_freq, 1000 / finder.N):
                f.write(",".join(bigram) + "\n")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for recipe in tokenized_recipes:
                f.write(" ".join(recipe.ingredients + recipe.instructions))
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
        dim=300,         # model dimensions
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
