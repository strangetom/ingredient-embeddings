#!/usr/bin/env/python3


from embeddings._dataclasses import Embeddings


class BoundaryTokenRemover:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def remove_boundary_tokens(self) -> Embeddings:
        """Remove boundary tokens from embeddings.

        Boundary tokens are <start_ing>, <end_ing>, <start_inst>, <end_inst>.

        Returns
        -------
        Embeddings
            Embeddings object.
        """
        self.embeddings.delete("<start_ing>")
        self.embeddings.delete("<end_ing>")
        self.embeddings.delete("<start_inst>")
        self.embeddings.delete("<end_inst>")

        return self.embeddings
