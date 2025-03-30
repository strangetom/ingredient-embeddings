#!/usr/bin/env/python3

from collections import Counter

import nltk

from embeddings.data import TokenizedRecipe


def train_bigram_model_nltk(tokenized_recipes: list[TokenizedRecipe]) -> None:
    """Extract bigrrams from tokenized recipes.

    Parameters
    ----------
    tokenized_recipes : list[TokenizedRecipe]
        List of tokenized recipes
    """
    print("Preparing for bigram model training...")
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    print("Training bigram model...")
    finder = nltk.collocations.BigramCollocationFinder.from_documents(
        recipe.instructions for recipe in tokenized_recipes
    )
    with open("bigrams.csv", "w") as f:
        # Keep bigrams that occur in 5% of recipes
        for bigram in finder.above_score(bigram_measures.raw_freq, 0.0001):
            f.write(",".join(bigram) + "\n")


def train_bigram_model_nouns(
    tokenized_recipes: list[TokenizedRecipe], freq_filter: int | float
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
    bigram_dist = []
    for recipe in tokenized_recipes:
        for tokens, pos_tags in zip(recipe.instructions, recipe.instructions_pos):
            for (w1, p1), (w2, p2) in nltk.bigrams(zip(tokens, pos_tags)):
                if w1 and w2 and w1 != w2 and p1.startswith("N") and p2.startswith("N"):
                    bigram_dist.append((w1, w2))

    bigram_fdist = Counter(bigram_dist)

    if isinstance(freq_filter, float):
        freq_filter = int(len(bigram_dist) * freq_filter)

    filtered_bigram_fdist = {
        key: value for key, value in bigram_fdist.items() if value >= freq_filter
    }

    with open("bigrams.csv", "w") as f:
        # Write out bigrams to csv in order of most to least common
        for bigram, freq in sorted(
            filtered_bigram_fdist.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(",".join(bigram) + "\n")
