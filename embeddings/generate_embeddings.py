#!/usr/bin/env/python3

import argparse
import concurrent.futures as cf
import gzip
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

from embeddings.bigrams import BigramModel
from embeddings.data import (
    TokenizedRecipe,
    chunked,
    load_embeddings,
    load_recipes,
    download_recipenlg_dataset,
    tokenize_recipes,
)
from embeddings.glove import VocabCount, Cooccur, Shuffle, GloVe
from embeddings.ontology import FoodOn


def join_bigrams_in_recipes(
    recipes: list[TokenizedRecipe], bm: BigramModel | None
) -> list[str]:
    """Use bigram model to join bigram tokens in recipe ingredients and instructions.

    Each recipe is returned as a single str, created by joining the ingredient then
    instructions tokens with a space, after joining bigrams with an underscore.

    Parameters
    ----------
    recipes : list[TokenizedRecipe]
        List of TokenizedRecipes to join bigrams for.
    bm : BigramModel | None
        Bigram model, or None if not using bigrams

    Returns
    -------
    list[str]
        List of recipe strings.
    """
    joined_recipes = []
    for recipe in recipes:
        if bm:
            ingredients = [
                ingred
                for ingredient in recipe.ingredients
                for ingred in bm.join_bigrams(ingredient)
                if ingred
            ]
            instructions = [
                instruct
                for instruction in recipe.instructions
                for instruct in bm.join_bigrams(instruction)
                if instruct
            ]
            joined_recipes.append(" ".join(ingredients + instructions))

        ingredients = [
            ingred
            for ingredient in recipe.ingredients
            for ingred in ingredient
            if ingred
        ]
        instructions = [
            instruct
            for instruction in recipe.instructions
            for instruct in instruction
            if instruct
        ]

        joined_recipes.append(" ".join(ingredients + instructions))
    return joined_recipes


def flatten_recipes(
    tokenize_recipes: list[TokenizedRecipe], bigrams_file: str | None
) -> list[str]:
    """Flatten recipes by joining bigram tokens with an underscore, then joining all
    tokens with a space into a single string.

    Parameters
    ----------
    tokenize_recipes : list[TokenizedRecipe]
        Description
    bigrams_file : str | None
        Path to bigrams file, or None to not apply bigrams.

    Returns
    -------
    list[str]
        List of flattened recipes.
    """
    # Chunk data into 100 groups to process in parallel.
    n_chunks = 100
    # Define chunk size so all groups have about the same number of elements, except the
    # last group which will be slightly smaller.
    chunk_size = int((len(tokenize_recipes) + n_chunks) / n_chunks)
    chunks = chunked(tokenize_recipes, chunk_size)

    bm = None
    if bigrams_file:
        bm = BigramModel(bigrams_file)

    flattened_recipes = []
    print("Flattening recipes...")
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(join_bigrams_in_recipes, c, bm) for c in chunks]
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            flattened_recipes.extend(future.result())

    return flattened_recipes


def compress_file(path: str):
    """Compress file using gzip.

    Compressed file as ".gz" appended to end of file name.

    Parameters
    ----------
    path : str
        Path to file to compress.
    """
    with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
        dst.writelines(src)


def denoise(path: str, n: int) -> None:
    """Denoise embeddings by removing n principal components.

    References
    ----------
    Kawin Ethayarajh. 2018. Unsupervised Random Walk Sentence Embeddings: A Strong but
    Simple Baseline. In Proceedings of the Third Workshop on Representation Learning for
    NLP, pages 91–100, Melbourne, Australia. Association for Computational
    Linguistics. https://aclanthology.org/W18-3012/

    Parameters
    ----------
    path : str
        Path to embeddings text file.
    n : int
        Number of principal components to remove.
    """
    if n == 0:
        return

    def _projection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a.dot(b.T) * b

    print(f"Denoising embeddings by removing {n} principal components.")
    embeddings, header = load_embeddings(path)
    tokens = list(embeddings.keys())
    vectors = list(embeddings.values())

    svd = TruncatedSVD(n_components=n, random_state=0).fit(vectors)
    # Remove the weighted projections on the common discourse vectors
    singular_value_sum = (svd.singular_values_**2).sum()
    for i in range(n):
        lambda_i = (svd.singular_values_[i] ** 2) / singular_value_sum
        pc = svd.components_[i]
        vectors = [v - lambda_i * _projection(v, pc) for v in vectors]

    with open(path, "w") as f:
        f.write(f"{header}\n")
        for token, vector in zip(tokens, vectors):
            vec = " ".join(str(v) for v in vector)
            line = token + " " + vec + "\n"
            f.write(line)


def retrofit_embeddings(
    embedding_path: str,
    bigram_path: str | None,
    ontology_path: str,
    alpha: float = 0.5,
    beta: float = 0.5,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-3,
) -> None:
    """Retrofit embeddings using FoodOn ontology to provide an external source of
    sementic linking.

    References
    ----------
    Manaal Faruqui, Jesse Dodge, Sujay Kumar Jauhar, Chris Dyer, Eduard Hovy, and Noah
    A. Smith. 2015. Retrofitting Word Vectors to Semantic Lexicons. In Proceedings of
    the 2015 Conference of the North American Chapter of the Association for
    Computational Linguistics: Human Language Technologies, pages 1606–1615, Denver,
    Colorado. Association for Computational Linguistics.

    Parameters
    ----------
    embedding_path : str
        Path to embeddings text file.
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
    ontology = FoodOn(embedding_path, bigram_path, ontology_path)
    word_neighbours = ontology.similar_tokens()
    embeddings, header = load_embeddings(embedding_path)
    retrofitted = {word: vec.copy() for word, vec in embeddings.items()}

    for iter_ in range(max_iterations):
        total_change = 0.0
        words_updated = 0

        for word in embeddings.keys():
            neighbour_vecs = [retrofitted[word] for word in word_neighbours[word]]

            if not neighbour_vecs:
                continue

            original_vec = embeddings[word]
            neighbour_average = np.mean(neighbour_vecs, axis=0)
            new_embedding = (
                alpha * original_vec + beta * len(neighbour_vecs) * neighbour_average
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

    with open(embedding_path, "w") as f:
        f.write(f"{header}\n")
        for token, vector in retrofitted.items():
            vec = " ".join(str(v) for v in vector)
            line = token + " " + vec + "\n"
            f.write(line)


def generate_embeddings(args: argparse.Namespace):
    if not args.source and not args.training:
        raise ValueError("Supply either the source file or training file.")

    if not args.training:
        # If source file doesn't exist, download it
        source_path = Path(args.source)
        if not source_path.is_file():
            download_recipenlg_dataset(args.source)

        recipes = load_recipes(args.source)
        tokenized_recipes = tokenize_recipes(recipes)
        flattened_recipes = flatten_recipes(tokenized_recipes, args.bigrams)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for recipe in flattened_recipes:
                f.write(recipe)
                f.write("\n")

            training_file = f.name
    else:
        training_file = args.training

    print(f"Preprocessed sentences saved to {training_file}")
    if args.preprocess:
        # If only preprocessing, exit now
        sys.exit(0)

    vocab = VocabCount.run(training_file, verbose=2, min_count=15)
    cooccur = Cooccur.run(
        training_file,
        verbose=2,
        symmetric=1,
        window_size=10,
        vocab_file=vocab,
        memory=32,
    )
    shuff = Shuffle.run(cooccur, verbose=2, memory=32)
    embeddings = GloVe.run(
        input_file=shuff,
        vocab_file=vocab,
        verbose=2,
        write_header=1,
        iter=100,
        binary=2,
        vector_size=args.dim,
        save_file=args.model,
    )
    denoise(embeddings + ".txt", n=5)
    # retrofit_embeddings(embeddings + ".txt", args.bigrams, "data/foodon.owl")
    compress_file(embeddings + ".txt")
