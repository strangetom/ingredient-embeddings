#!/usr/bin/env/python3

import concurrent.futures as cf
import csv
import json
import string
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from embeddings.preprocess import preprocess_recipe, stem, tokenize

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/saldenisov/recipenlg"

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


def download_recipenlg_dataset(save_path: str = "data/recipenlg.zip"):
    """Download RecipeNLG dataset from kaggle and extract csv from the downlaoded zip
    file.

    If the specified folder doesn't exist, create it.

    Parameters
    ----------
    save_path : str, optional
        Path to save downloaded zip file to.
    """
    if not Path(save_path).parent.is_dir():
        Path(save_path).parent.mkdir()

    print(f"Downloading {DATASET_URL} to {save_path}")
    urllib.request.urlretrieve(DATASET_URL, save_path)

    print(f"Extracting {save_path} to {save_path.replace('.zip', '.csv')}")
    with zipfile.ZipFile(save_path, "r") as zip:
        with open(save_path.replace(".zip", ".csv"), "wb") as csv:
            csv.write(zip.read("dataset/full_dataset.csv"))

    print("Done")


@dataclass
class Recipe:
    id_: int
    ingredients: list[str]
    instructions: list[str]

    def __post_init__(self):
        self.ingredients = [
            preprocess_recipe(ingred).lower() for ingred in self.ingredients if ingred
        ]
        self.instructions = [
            preprocess_recipe(instruct).lower()
            for instruct in self.instructions
            if instruct
        ]

    def ingredient_tokens(self) -> list[list[tuple[str, str]]]:
        """Return tokens for ingredients.

        Returns
        -------
        list[list[tuple[str, str]]]
            List of tokens for each ingredient sentence.
        """
        tokens = [self._tokens(ingreds) for ingreds in self.ingredients]
        return [tok for tok in tokens if tok]

    def instruction_tokens(self) -> list[list[tuple[str, str]]]:
        """Return tokens for instructions.

        Returns
        -------
        list[list[tuple[str, str]]]
            List of tokens for each instruction step.
        """
        tokens = [self._tokens(instruct) for instruct in self.instructions]
        return [tok for tok in tokens if tok]

    def _tokens(self, text: str) -> list[tuple[str, str]]:
        """Tokenize input text, only keeping tokens that meeting criteria.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        list[tuple[str, str]]
            List of (token, pos) tuples.
        """
        tokens = []
        for token, pos in nltk.pos_tag(tokenize(text)):
            if (
                pos in ALLOWED_POS_TAGS
                and token not in string.punctuation
                and not token.isdigit()
                and not token.isnumeric()
                and not token.isspace()
                and token not in STOP_WORDS
                and len(token) > 1  # avoid leftover units e.g. 'c'
            ):
                tokens.append((stem(token), pos))
            else:
                tokens.append((None, None))

        return tokens


@dataclass
class TokenizedRecipe:
    id_: int
    ingredients: list[list[str]]
    ingredients_pos: list[list[str]]
    instructions: list[list[str]]
    instructions_pos: list[list[str]]


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
                ingredients=json.loads(row["ingredients"]),
                instructions=json.loads(row["directions"]),
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
    tokenized_recipes = []
    for recipe in recipes:
        ingredient_tokens, ingredient_pos = [], []
        for sentence in recipe.ingredient_tokens():
            tokens, pos = zip(*sentence)
            ingredient_tokens.append(list(tokens))
            ingredient_pos.append(list(pos))
        instruction_tokens, instruction_pos = [], []
        for sentence in recipe.instruction_tokens():
            tokens, pos = zip(*sentence)
            instruction_tokens.append(list(tokens))
            instruction_pos.append(list(pos))

        tokenized_recipes.append(
            TokenizedRecipe(
                id_=recipe.id_,
                ingredients=ingredient_tokens,
                ingredients_pos=ingredient_pos,
                instructions=instruction_tokens,
                instructions_pos=instruction_pos,
            )
        )
    return tokenized_recipes


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
