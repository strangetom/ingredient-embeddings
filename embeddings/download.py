#!/usr/bin/env/python3

import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/saldenisov/recipenlg"


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
