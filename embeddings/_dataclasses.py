#!/usr/bin/env/python3

import gzip

import numpy as np


class Embeddings:
    def __init__(self, embeddings_dict: dict[str, np.ndarray], file_path: str):
        self.embeddings = embeddings_dict
        self.file_path = file_path

    def __getitem__(self, token: str) -> np.ndarray:
        if token not in self.embeddings:
            raise KeyError(token)

        return self.embeddings[token]

    def __contains__(self, token: str) -> bool:
        return token in self.embeddings

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def items(self):
        return self.embeddings.items()

    def delete(self, token: str):
        if token in self.embeddings:
            del self.embeddings[token]

    def write(self, compress: bool = True):
        """Write embeddings to text file at given path, with given header.

        The embeddings are written to the file specified by the file_path attribute.

        Parameters
        ----------
        compress : bool, optional
            If True, write gzipped file.
        """
        vocab_count = len(self.embeddings)

        first_key = next(iter(self.embeddings))
        dimension = self.embeddings[first_key].shape[0]

        header = f"{vocab_count} {dimension}"
        with open(self.file_path, "w") as f:
            f.write(f"{header}\n")
            for token, vector in self.embeddings.items():
                vec = " ".join(str(v) for v in vector)
                line = token + " " + vec + "\n"
                f.write(line)

        if compress:
            self.compress_file()

    def compress_file(self):
        """Compress file using gzip.

        Compressed file as ".gz" appended to end of file name.
        """
        with (
            open(self.file_path, "rb") as src,
            gzip.open(self.file_path + ".gz", "wb") as dst,
        ):
            dst.writelines(src)
