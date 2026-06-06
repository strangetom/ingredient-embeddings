#!/usr/bin/env/python3

import numpy as np

from ._dataclasses import Embeddings


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
