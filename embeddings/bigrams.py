#!/usr/bin/env/python3

import csv


class BigramModel:
    def __init__(self, bigram_csv: str):
        self.bigram_csv = bigram_csv
        self.bigrams = self._load_csv(bigram_csv)

    def __repr__(self) -> str:
        return f"BigramModel(bigram_csv='{self.bigram_csv}')"

    def __str__(self) -> str:
        return f"BigramModel(n_bigrams={len(self.bigrams)})"

    def _load_csv(self, csv_file: str) -> list[tuple[str, str]]:
        """Load CSV file of bigrams.

        The CSV should have two columns and no headers.

        Parameters
        ----------
        csv_file : str
            Path to CSV file to load.

        Returns
        -------
        list[tuple[str, str]]
            List of bigram tuples.
        """
        bigrams = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                bigrams.append(tuple(row))

        return bigrams

    def join_bigrams(self, tokens: list[str]) -> list[str]:
        """Join bigrams in tokens list with underscore.

        Provided tokens should already been stemmed and had stop words, numeric tokens,
        punctuation and single character tokens removed.
        Provided tokens should only be nouns, verbs, adjectives, adverbs or foreign
        words.

        Parameters
        ----------
        tokens : list[str]
            List of tokens.

        Returns
        -------
        list[str]
            List of tokens, with bigrams joined by underscore

        Examples
        --------
        >>> b = BigramModel("...")
        >>> b.join_bigrams(["cup", "confectioners", "sugar"])
        ["cup", "confectioners_sugar"]
        """
        joined_tokens = []
        consumed = None
        for i, token in enumerate(tokens):
            if i == consumed:
                consumed = None
                continue

            if i < len(tokens) - 1:
                candidate_bigram = (token, tokens[i + 1])
                if candidate_bigram in self.bigrams:
                    joined_tokens.append("_".join(candidate_bigram))
                    consumed = i + 1
                else:
                    joined_tokens.append(token)
            else:
                joined_tokens.append(token)

        return joined_tokens
