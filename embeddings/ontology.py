#!/usr/bin/env/python3

import csv
import warnings
from pathlib import Path
import urllib.request

import owlready2

# Suppress owlready2 warnings about unsupported datatypes
warnings.filterwarnings("ignore", category=UserWarning, module="owlready2")

DATASET_URL = (
    "https://raw.githubusercontent.com/FoodOntology/foodon/refs/heads/master/foodon.owl"
)


def download_foodon_owl(save_path: str = "data/foodon.owl"):
    """Download FoodOn ontology in OWL format.

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
    print("Done")


def get_primary_label(cls: owlready2.entity.ThingClass) -> str:
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


def get_leaf_paths(lst: list) -> list[list[str]]:
    """Recurse through node hiearchy, return list of labels for path to leaf node.

    Parameters
    ----------
    lst : list
        List of labels

    Returns
    -------
    list[list[str]]
        List of paths
    """
    if not lst:
        return [[]]
    return [
        [get_primary_label(cls), *path]
        for cls in lst
        for path in get_leaf_paths(list(cls.subclasses()))
    ]


def extract_food_material_leaf_node_paths(owl_file_path):
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
    onto = owlready2.get_ontology(f"file://{owl_file_path}").load()
    food_material_class = onto.search_one(label="food material")
    return get_leaf_paths([food_material_class])


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
