#!/usr/bin/env/python3

import numpy as np

from embeddings._dataclasses import Embeddings

from .ontology import FoodOn


class Retrofitter:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def retrofit(
        self,
        bigram_path: str | None,
        ontology_path: str,
        alpha: float = 0.5,
        beta: float = 0.5,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-3,
    ) -> Embeddings:
        """Retrofit embeddings using FoodOn ontology to provide an external source of
        semantic linking.

        References
        ----------
        Manaal Faruqui, Jesse Dodge, Sujay Kumar Jauhar, Chris Dyer, Eduard Hovy, and
        Noah A. Smith. 2015. Retrofitting Word Vectors to Semantic Lexicons. In
        Proceedings of the 2015 Conference of the North American Chapter of the
        Association for Computational Linguistics: Human Language Technologies, pages
        1606–1615, Denver, Colorado. Association for Computational Linguistics.

        Parameters
        ----------
        bigram_path : str
            Path to bigrams csv file.
        ontology_path : str
            Path to ontology owl file
        alpha : float, optional
            Description
        beta : float, optional
            Description
        max_iterations : int, optional
            Maximum number of iterations to run retrofitting for.
        convergence_threshold : float, optional
            Criteria for stopping retrofitting if average change is less than this
            threshold.
        """
        print("Retrofitting embeddings using ontology.")
        ontology = FoodOn(self.embeddings, bigram_path, ontology_path)
        word_neighbours = ontology.similar_tokens()
        retrofitted = {word: vec.copy() for word, vec in self.embeddings.items()}

        for iter_ in range(max_iterations):
            total_change = 0.0
            words_updated = 0

            for word in self.embeddings.keys():
                neighbour_vecs = [retrofitted[word] for word in word_neighbours[word]]

                if not neighbour_vecs:
                    continue

                original_vec = self.embeddings[word]
                neighbour_average = np.mean(neighbour_vecs, axis=0)
                new_embedding = (
                    alpha * original_vec
                    + beta * len(neighbour_vecs) * neighbour_average
                ) / (alpha + beta * len(neighbour_vecs))

                # Calculate change magnitude from where the retrofitted embedding was
                change = np.linalg.norm(new_embedding - retrofitted[word])
                total_change += change
                words_updated += 1

                retrofitted[word] = new_embedding

            avg_change = total_change / max(words_updated, 1)
            print(f"Iteration {iter_ + 1}: avg change = {avg_change:.6f}")
            if avg_change < convergence_threshold:
                print(f"Converged after {iter_ + 1} iterations")
                break

        return Embeddings(retrofitted, self.embeddings.file_path)
