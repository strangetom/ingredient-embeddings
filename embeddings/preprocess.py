#!/usr/bin/env/python3

import re
import string
from itertools import chain
from functools import lru_cache

import nltk.stem.porter as nsp
from nltk import pos_tag
from nltk.corpus import stopwords


STEMMER = nsp.PorterStemmer()

# Define regular expressions used by tokenizer.
# Matches one or more whitespace characters
WHITESPACE_TOKENISER = re.compile(r"\S+")
# Matches and captures one of the following: ( ) [ ] { } , " / : ; ? ! ~
PUNCTUATION_TOKENISER = re.compile(r"([\(\)\[\]\{\}\,/:;\?\!\*\~])")
# Matches and captures full stop at end of string
# (?<!\.\w) is a negative lookbehind that prevents matches if the last full stop
# is preceded by a a full stop then a word character.
FULL_STOP_TOKENISER = re.compile(r"(?<!\.\w)(\.)$")

STOP_WORDS = stopwords.words("english")

ALLOWED_POS_TAGS = {
    "NN",
    "NNS",
    "NNP",
    "NNPS",
    "JJ",
    "JJR",
    "JJS",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
}


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
def stem(token: str, to_lowecase: bool) -> str:
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
    return STEMMER.stem(token, to_lowercase=to_lowecase)


def preprocess_recipe(recipe: str) -> list[str]:
    """Preprocess recipe for embeddings training.

    Each recipe is the combination of the ingredients and instructions.
    The preprocessing returns the tokens of the recipe with stop words, punctuation,
    numbers and whitespace removed.

    Only include nouns, verbs, adjectives per
    http://link.springer.com/10.1007/978-3-030-72379-8_23

    Parameters
    ----------
    recipe : str
        Recipe ingredients followed by instruction steps, as a single string.

    Returns
    -------
    list[str]
        Preprocessed recipe.
    """
    tokens = tokenize(recipe)

    return [
        stem(str(token), to_lowecase=True)
        for token, pos in pos_tag(tokens)
        if pos in ALLOWED_POS_TAGS 
        and token not in string.punctuation
        and not token.isdigit()
        and not token.isnumeric()
        and not token.isspace()
        and token not in STOP_WORDS
        and len(token) > 1  # avoid leftover units e.g. 'c''
    ]


def preprocess_recipes(recipes: list[str]) -> list[str]:
    return [" ".join(preprocess_recipe(recipe)) for recipe in recipes]
