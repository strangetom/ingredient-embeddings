#!/usr/bin/env/python3

import concurrent.futures as cf
import csv
import json
import string
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import islice
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterable

import nltk
import numpy as np
from tqdm import tqdm

from embeddings.preprocess import preprocess_recipe, stem, tokenize

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/saldenisov/recipenlg"

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


@lru_cache
def load_stopwords_list() -> list[str]:
    """Load list of stopwords names from file.

    This is a list of high frequency grammatical words derived from
    nltk.corpus.stopwords .The original list from NLTK has been edited to remove words
    that the tokenizer cannot output.
    See also https://dx.doi.org/10.18653/v1/W18-2502

    This function is cached so it can be called multiple times without the overhead
    of loading the list from file everytime.

    Returns
    -------
    list[str]
        List of stop words.
    """
    with as_file(files(__package__) / "stopwords.json") as p:
        with open(p, "r") as f:
            stopwords = json.load(f)

    return [stem(w) for w in stopwords]


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
        stopwords = load_stopwords_list()

        tokens = []
        for token, pos in nltk.pos_tag(tokenize(text)):
            if (
                # Allow tokens ending in % even if their POS tag is not in allowed list.
                (pos in ALLOWED_POS_TAGS or token.endswith("%"))
                and not token.isnumeric()
                and not token.isdigit()
                and not token.isdecimal()
                and not token.isspace()
                and token not in string.punctuation
                and token not in stopwords
                and len(token) > 1
                and "=" not in token
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


def load_bad_recipes_list(file: str) -> set[int]:
    """Read list of bad recipe IDs from file.

    Each line of file should follow the pattern:
        12345 # comment

    Parameters
    ----------
    file : str
        File containing bad recipe IDs.

    Returns
    -------
    set[int]
        Set of recipe IDs.
    """
    bad_recipes = set()
    with open(file, "r") as f:
        for line in f.read().splitlines():
            id_, comment = line.split("#", 1)
            bad_recipes.add(int(id_))

    return bad_recipes


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
    BAD_RECIPES = load_bad_recipes_list("bad_recipes.txt")

    recipes = []
    print("Loading recipes...")
    with open(csv_file, "r") as f:
        row_count = sum(1 for _ in csv.reader(f))
        f.seek(0)  # Rewind to start of file after counting rows.
        for row in tqdm(csv.DictReader(f), total=row_count):
            if "cookbooks.com" in row["link"]:
                # Recipes from cookbooks seem to be all user submitted and of
                # extremely variable quality (including entries that are not
                # recipes at all), so exclude them all
                continue

            if int(row[""]) in BAD_RECIPES:
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
            if not sentence:
                continue
            tokens, pos = zip(*sentence)
            ingredient_tokens.append(list(tokens))
            ingredient_pos.append(list(pos))
        instruction_tokens, instruction_pos = [], []
        for sentence in recipe.instruction_tokens():
            if not sentence:
                continue
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


@lru_cache
def load_embeddings(path: str) -> tuple[dict[str, np.ndarray], str]:
    """Load GloVe embeddings from text file, return dict of embeddings as well as header

    Parameters
    ----------
    path : str
        Path to embeddings text file.

    Returns
    -------
    tuple[dict[str, np.ndarray], str]
        Returns embeddings and header from file
    """
    embeddings = {}
    with open(path, "r") as f:
        # Read first line as header
        header = f.readline().rstrip()

        # Read remaining lines and load vectors
        for line in f:
            parts = line.rstrip().split()
            token = parts[0]
            vector = np.array([float(v) for v in parts[1:]], dtype=np.float32)
            embeddings[token] = vector

        return embeddings, header
