#!/usr/bin/env/python3

import re
import string
from itertools import chain
from functools import lru_cache
from html import unescape

import nltk.stem.snowball as nsp

STEMMER = nsp.EnglishStemmer()

# Define regular expressions used by tokenizer.
# Matches one or more whitespace characters
WHITESPACE_TOKENISER = re.compile(r"\S+")
# Matches and captures one of the following: ( ) [ ] { } , " / : ; ? ! ~
PUNCTUATION_TOKENISER = re.compile(r"([\(\)\[\]\{\}\,/:;\?\!\*\~])")
# Matches and captures full stop at end of string
# (?<!\.\w) is a negative lookbehind that prevents matches if the last full stop
# is preceded by a a full stop then a word character.
FULL_STOP_TOKENISER = re.compile(r"(?<!\.\w)(\.)$")

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
MULTPLE_WHITESPACE = re.compile(r"(\s)+ ", re.UNICODE)


def tokenize(sentence: str) -> list[str]:
    """Tokenise an ingredient sentence.

    The sentence is split on whitespace characters into a list of tokens.
    If any of these tokens contains of the punctuation marks captured by
    PUNCTUATION_TOKENISER, these are then split and isolated as a separate
    token.

    The returned list of tokens has any empty tokens removed.

    Parameters
    ----------
    sentence : str
        Ingredient sentence to tokenize

    Returns
    -------
    list[str]
        List of tokens from sentence.

    Examples
    --------
    >>> tokenize("2 cups (500 ml) milk")
    ["2", "cups", "(", "500", "ml", ")", "milk"]

    >>> tokenize("1-2 mashed bananas: as ripe as possible")
    ["1-2", "mashed", "bananas", ":", "as", "ripe", "as", "possible"]

    >>> tokenize("1.5 kg bananas, mashed")
    ["1.5", "kg", "bananas", ",", "mashed"]

    >>> tokenize("Freshly grated Parmesan cheese, for garnish.")
    ["Freshly", "grated", "Parmesan", "cheese", ",", "for", "garnish", "."]

    >>> tokenize("2 onions, finely chopped*")
    ["2", "onions", ",", "finely", "chopped", "*"]

    >>> tokenize("2 cups beef and/or chicken stock")
    ["2", "cups", "beef", "and/or", "chicken", "stock"]
    """
    tokens = [
        PUNCTUATION_TOKENISER.split(tok)
        for tok in WHITESPACE_TOKENISER.findall(sentence)
    ]
    flattened = [tok for tok in chain.from_iterable(tokens) if tok]

    # Second pass to separate full stops from end of tokens
    tokens = [FULL_STOP_TOKENISER.split(tok) for tok in flattened]

    return [tok for tok in chain.from_iterable(tokens) if tok]


@lru_cache(maxsize=512)
def stem(token: str) -> str:
    """Stem function with cache to improve performance.

    The stem of a word output by the PorterStemmer is always the same, so we can
    cache the result the first time and return that for subsequent future calls
    without the need to do all the processing again.

    Parameters
    ----------
    token : str
        Token to stem

    Returns
    -------
    str
        Stem of token
    """
    return STEMMER.stem(token)


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
    return MULTPLE_WHITESPACE.sub(" ", recipe)


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
