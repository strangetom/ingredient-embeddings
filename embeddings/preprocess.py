#!/usr/bin/env/python3

import concurrent.futures as cf
import math
import re
import string
from dataclasses import dataclass
from itertools import islice
from functools import partial
from html import unescape
from typing import Iterable

import nltk
from tqdm import tqdm

from ._constants import ALLOWED_POS_TAGS, STOPWORDS
from ._utils import tokenize, stem

HTML_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
URL_HTTP = re.compile(r"(https?://\S+)", re.UNICODE)
URL_WWW = re.compile(r"(www\.\S+)", re.UNICODE)
URL_TLD = re.compile(r"\s(\S+\.com\S+)", re.UNICODE)
# There's a negative lookahead on the NUMERIC regex to allow numbers that end with %.
NUMERIC = re.compile(r"(([0-9\-\.\/])+)(?![%0-9\-\.\/])", re.UNICODE)
CURRENCY = re.compile(r"([#£$]\S+)\b", re.UNICODE)
LQUOTE = re.compile(r"\b[\"\']", re.UNICODE)
RQUOTE = re.compile(r"[\"\']\b", re.UNICODE)
SYMBOLS = re.compile(r"[™®@]", re.UNICODE)
AMPERSAND = re.compile(r"(?<=[a-z])(&)(?![a-z])", re.UNICODE)
MULTIPLE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


def remove_html_tags(recipe: str) -> str:
    """Remove HTML tags and their contents from recipe.

    Parameters
    ----------
    recipe : str
        Recipe, as string.

    Returns
    -------
    str
        Recipe, with HTML tags removed.
    """
    return HTML_TAGS.sub(" ", recipe)


def remove_urls(recipe: str) -> str:
    """Remove URLs from recipe.

    Assumes remove_html_tags has already been run on recipe.

    Parameters
    ----------
    recipe : str
        Recipe, as string.

    Returns
    -------
    str
        Recipe, with URLs removed.
    """
    recipe = URL_HTTP.sub(" ", recipe)
    recipe = URL_WWW.sub(" ", recipe)
    recipe = URL_TLD.sub(" ", recipe)
    return recipe


def remove_numeric(recipe: str) -> str:
    """Remove numeric words from recipe.

    This includes numbers (e.g. 1, 215), decimals (e.g. 0.2) and ranges (e.g. 1-2).

    Parameters
    ----------
    recipe : str
        Recipe, as string.

    Returns
    -------
    str
        Recipe with numeric words removed.
    """
    return NUMERIC.sub(" ", recipe)


def remove_currency(recipe: str) -> str:
    """Remove currency words from recipe.

    e.g. £12.25, $100 etc.

    Parameters
    ----------
    recipe : str
        Recipe, as string

    Returns
    -------
    str
        Recipe with currency tokens removed
    """
    return CURRENCY.sub(" ", recipe)


def remove_quotes(recipe: str) -> str:
    """Remove quotes from start or end of word.

    Parameters
    ----------
    recipe : str
        Recipe, as string

    Returns
    -------
    str
        Recipe with quote symbols removed
    """
    recipe = LQUOTE.sub(" ", recipe)
    recipe = RQUOTE.sub(" ", recipe)
    return recipe


def remove_symbols(recipe: str) -> str:
    """Remove symbols such as ™ from recipe.

    Parameters
    ----------
    recipe : str
        Recipe, as string.

    Returns
    -------
    str
        Recipe with symbols removed.
    """
    return SYMBOLS.sub("", recipe)


def split_ampersand_from_word(recipe: str) -> str:
    """Split ampersand from end of word by inserting space.

    The regex has a positive lookbehind for a lower case character and a negative
    lookahead for a lower case character. This is so we capture cases like "salt&" but
    not "m&m".

    Parameters
    ----------
    recipe : str
        Recipe, as string.

    Returns
    -------
    str
        Recipe with with space inserted by ampersands.
    """
    return AMPERSAND.sub(" &", recipe)


def remove_multiple_whitespace(recipe: str) -> str:
    """Remove repeating consecutive whitespace characters and replace in single space.

    Parameters
    ----------
    recipe : str
        Recipe, as string

    Returns
    -------
    str
        Recipe with repeating whitespace removed.
    """
    return MULTIPLE_WHITESPACE.sub(" ", recipe)


def remove_bad_words(recipe: str) -> str:
    """Remove bad words from recipe.

    * Words containing underscores - these are typically errors in the recipe text
      where javascript or html entities have been included.
    * Words that only contain punctuation marks

    Parameters
    ----------
    recipe : str
        Recipe, as string.

    Returns
    -------
    str
        Recipe with bad words removed.
    """
    words = []
    for word in recipe.split(" "):
        if "_" in word:
            continue
        if all(char in string.punctuation for char in word):
            continue

        words.append(word)

    return " ".join(words)


CLEAN_FUNCS = [
    unescape,
    remove_html_tags,
    remove_urls,
    remove_currency,
    remove_numeric,
    remove_symbols,
    remove_quotes,
    split_ampersand_from_word,
    remove_multiple_whitespace,
    remove_bad_words,
]


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
                # Allow tokens ending in % even if their POS tag is not in allowed list.
                (pos in ALLOWED_POS_TAGS or token.endswith("%"))
                and not token.isnumeric()
                and not token.isdigit()
                and not token.isdecimal()
                and not token.isspace()
                and token not in string.punctuation
                and token not in STOPWORDS
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


def preprocess_recipe(recipe: str) -> str:
    """Preprocess recipe for embeddings training.

    Parameters
    ----------
    recipe : str
        Recipe ingredients followed by instruction steps, as a single string.

    Returns
    -------
    list[str]
        Preprocessed recipe.
    """
    for func in CLEAN_FUNCS:
        recipe = func(recipe)

    return recipe


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
    chunk_size = math.ceil(len(recipes) / n_chunks)
    chunks = chunked(recipes, chunk_size)

    tokenized_recipes = []
    print("Preprocessing recipes...")
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_recipes_tokens, c) for c in chunks]
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            tokenized_recipes.extend(future.result())

    return tokenized_recipes
