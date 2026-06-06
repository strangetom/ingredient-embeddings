#!/usr/bin/env/python3

import csv
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ._dataclasses import Embeddings
from .preprocess import Recipe


def load_embeddings(path: str) -> Embeddings:
    """Load GloVe embeddings from text file, return dict of embeddings as well as header

    Parameters
    ----------
    path : str
        Path to embeddings text file.

    Returns
    -------
    Embeddings
        Embeddings object.
    """
    embeddings = {}
    with open(path, "r") as f:
        # Read first line as header
        _ = f.readline()

        # Read remaining lines and load vectors
        for line in f:
            parts = line.rstrip().split()
            token = parts[0]
            vector = np.array([float(v) for v in parts[1:]], dtype=np.float32)
            embeddings[token] = vector

        return Embeddings(embeddings, path)


DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/saldenisov/recipenlg"


def download_recipenlg_dataset(save_path: str = "data/recipenlg.zip"):
    """Download RecipeNLG dataset from kaggle and extract csv from the downloaded zip
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
        for row in tqdm(csv.DictReader(f), unit="recipes", total=row_count):
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
