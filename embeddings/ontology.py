#!/usr/bin/env/python3

import csv
import string
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

from nltk.corpus import stopwords
import owlready2

from embeddings.bigrams import BigramModel
from embeddings.data import load_embeddings
from embeddings.preprocess import stem, tokenize

# Suppress owlready2 warnings about unsupported datatypes
warnings.filterwarnings("ignore", category=UserWarning, module="owlready2")

DATASET_URL = (
    "https://raw.githubusercontent.com/FoodOntology/foodon/refs/heads/master/foodon.owl"
)

STOP_WORDS = stopwords.words("english")


class FoodOn:
    def __init__(
        self,
        embeddings_file_path: str,
        bigrams_file_path: None | str = None,
        owl_file_path: None | str = None,
    ):
        self.embeddings_file_path = embeddings_file_path
        self.bigrams_file_path = bigrams_file_path

        if owl_file_path:
            self.owl_file_path = owl_file_path
        else:
            self.owl_file_path = self.download()

        self.ingredient_groups = self.group_ingredients()
        self.token_similarity = self.similar_tokens()

    def download(self, save_path: str = "data/foodon.owl") -> str:
        """Download FoodOn ontology in OWL format.

        If the specified folder doesn't exist, create it.

        Parameters
        ----------
        save_path : str, optional
            Path to save downloaded owl file to.

        Returns
        -------
        str
            Path to saved owl file.
        """
        if not Path(save_path).parent.is_dir():
            Path(save_path).parent.mkdir()

        print(f"Downloading {DATASET_URL} to {save_path}")
        urllib.request.urlretrieve(DATASET_URL, save_path)
        print("Done")
        return save_path

    def _get_primary_label(self, cls: owlready2.entity.ThingClass) -> str:
        """Return primary label for class.

        Parameters
        ----------
        cls : owlready2.entity.ThingClass
            Class to get primary label for.

        Returns
        -------
        str
            Primary label
        """
        try:
            if cls.label:
                return cls.label.first().strip()
            else:
                return str(cls.name)
        except ValueError:
            return str(cls.name)

    def get_leaf_paths(self, lst: list[owlready2.entity.ThingClass]) -> list[list[str]]:
        """Recurse through node hiearchy, return list of labels for path to leaf node.

        Parameters
        ----------
        lst : list[owlready2.entity.ThingClass]
            List of labels

        Returns
        -------
        list[list[str]]
            List of paths
        """
        if not lst:
            return [[]]
        return [
            [self._get_primary_label(cls), *path]
            for cls in lst
            for path in self.get_leaf_paths(list(cls.subclasses()))
        ]

    def _extract_food_material_leaf_node_paths(self) -> list[list[str]]:
        """Extract paths to leaf nodes under food_material category from FoodOn OWL file.

        Parameters
        ----------
        owl_file_path : str
            Path to the FoodOn OWL file

        Returns
        -------
        dict
            Dictionary with leaf node names as keys and synonyms as values
        """
        onto = owlready2.get_ontology(f"file://{self.owl_file_path}").load()
        food_material_class = onto.search_one(label="food material")
        return self.get_leaf_paths([food_material_class])

    def group_ingredients(self) -> dict[str, list[str]]:
        """Group ingredients from FoodOn ontology by "food product".

        Returns
        -------
        dict[str, list[str]]
            Dict of food product: [ingredients]
        """
        groups = defaultdict(list)

        for path in self._extract_food_material_leaf_node_paths():
            for p in reversed(path):
                if p.endswith("food product"):
                    groups[p].append(path[-1])
                    break

        return groups

    def similar_tokens(self) -> dict[str, set[str]]:
        """From ingredients grouped by food product, calculate all tokens that belong to
        each group.

        Returns
        -------
        dict[str, set[str]]
            Dict of similar tokens for each token.
        """
        if self.bigrams_file_path:
            bm = BigramModel(self.bigrams_file_path)
        else:
            bm = None

        similar = defaultdict(set)
        for group in self.ingredient_groups.values():
            group_tokens = set()
            for ingredient in set(group):
                tokens = self._tokenise(ingredient)
                if bm:
                    group_tokens |= set(bm.join_bigrams(tokens)) | set(tokens)
                else:
                    group_tokens |= set(tokens)

            for token in group_tokens:
                similar[token] = similar[token] | group_tokens - {token}

        return similar

    def _tokenise(self, ingredient: str) -> list[str]:
        """Tokenize ingredient.

        This output must be a list to preserve ordering for applying bigrams.

        Parameters
        ----------
        ingredient : str
            Ingredient to tokenize.

        Returns
        -------
        list[str]
            List of tokens.
        """
        embeddings, _ = load_embeddings(self.embeddings_file_path)
        embedding_tokens = set(embeddings.keys())

        return [
            stem(token)
            for token in tokenize(ingredient)
            if not token.isnumeric()
            and not token.isdigit()
            and not token.isdecimal()
            and not token.isspace()
            and token not in string.punctuation
            and token not in STOP_WORDS
            and stem(token) in embedding_tokens
        ]


def save_leaf_nodes_to_csv(leaf_nodes_paths: list[list[str]], output_path: str):
    """Save leaf node paths to csv file.

    Parameters
    ----------
    leaf_nodes_paths : list[list[str]]
        Leaf node paths
    output_path : str
        CSV file
    """
    max_depth = max(len(p) for p in leaf_nodes_paths)
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        for path in leaf_nodes_paths:
            path += [""] * (max_depth - len(path))
            writer.writerow(path)

    print(f"Results saved to {output_path}")
