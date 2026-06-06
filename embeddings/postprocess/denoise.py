#!/usr/bin/env/python3

import numpy as np
from sklearn.decomposition import TruncatedSVD

from embeddings._dataclasses import Embeddings


class Denoiser:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def _calculate_isotropy(self, vectors: np.ndarray) -> np.floating:
        """Calculate the isotropy of the vectors.

        Isotropy is a measure of how uniformly spaced the vectors are. A higher value
        indicate more uniformly spaced, a lower value indicates a stronger bias in the
        vectors.

        Parameters
        ----------
        vectors : np.ndarray
            Embeddings vectors.

        Returns
        -------
        np.floating
            Isotropy measure.
        """
        # Center the vectors
        vectors = vectors - np.mean(vectors, axis=0)
        # Compute covariance matrix eigenvalues
        cov = np.cov(vectors, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)

        # Participation ratio formula
        numerator = np.sum(eigenvalues) ** 2
        denominator = np.sum(eigenvalues**2)
        return numerator / (len(eigenvalues) * denominator)

    def denoise(self, n: int) -> Embeddings:
        """Denoise embeddings by removing n principal components.

        References
        ----------
        Kawin Ethayarajh. 2018. Unsupervised Random Walk Sentence Embeddings: A Strong
        but Simple Baseline. In Proceedings of the Third Workshop on Representation
        Learning for NLP, pages 91–100, Melbourne, Australia. Association for
        Computational Linguistics. https://aclanthology.org/W18-3012/

        Parameters
        ----------
        n : int
            Number of principal components to remove.

        Returns
        -------
        dict[str, np.ndarray
            Denoised embeddings.
        """

        def _projection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a.dot(b.T) * b

        tokens = list(self.embeddings.keys())
        vectors = list(self.embeddings.values())

        if n == 0:
            return self.embeddings

        svd = TruncatedSVD(n_components=n, random_state=0).fit(vectors)
        # Remove the weighted projections on the common discourse vectors
        singular_value_sum = (svd.singular_values_**2).sum()
        for i in range(n):
            lambda_i = (svd.singular_values_[i] ** 2) / singular_value_sum
            pc = svd.components_[i]
            vectors = [v - lambda_i * _projection(v, pc) for v in vectors]

        return Embeddings(
            {token: vector for token, vector in zip(tokens, vectors)},
            self.embeddings.file_path,
        )

    def find_best_denoising(self) -> int:
        """Find the number of principal components to remove that maximises isotropy.

        Returns
        -------
        int
            Number of principal components.
        """
        isotropy_scores = {}
        for i in range(0, 10):
            denoised_vectors = self.denoise(i)
            vectors_array = np.array(list(denoised_vectors.values()))
            isotropy_scores[i] = self._calculate_isotropy(vectors_array)

        n_components, max_score = max(isotropy_scores.items(), key=lambda x: x[1])
        print(
            (
                f"Denoising embeddings by removing {n_components} principal components, "
                f"increases isotropy from {isotropy_scores[0]:.4f} to {max_score:.4f}."
            )
        )
        return n_components
