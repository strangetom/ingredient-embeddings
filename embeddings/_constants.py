#!/usr/bin/env/python3

import json
from importlib.resources import as_file, files

from ._utils import stem


def load_stopwords_list() -> set[str]:
    """Load list of stopwords names from file.

    This is a list of high frequency grammatical words derived from
    nltk.corpus.stopwords .The original list from NLTK has been edited to remove words
    that the tokenizer cannot output.
    See also https://dx.doi.org/10.18653/v1/W18-2502

    This function is cached so it can be called multiple times without the overhead
    of loading the list from file every time.

    Returns
    -------
    set[str]
        Set of stop words.
    """
    with as_file(files(__package__) / "stopwords.json") as p:
        with open(p, "r") as f:
            stopwords = json.load(f)

    return {stem(w) for w in stopwords}


STOPWORDS = load_stopwords_list()


def load_units_list() -> set[str]:
    """Load list of unit names from file.

    Returns
    -------
    set[str]
        Set of unit names.
    """
    with as_file(files(__package__) / "units.json") as p:
        with open(p, "r") as f:
            units = json.load(f)

    return {stem(u) for u in units}


UNITS = load_units_list()


def load_tools_list() -> set[str]:
    """Load list of tools names from file.

    Returns
    -------
    set[str]
        Set of tools names.
    """
    with as_file(files(__package__) / "tools.json") as p:
        with open(p, "r") as f:
            tools = json.load(f)

    return {stem(t) for t in tools}


TOOLS = load_tools_list()

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
