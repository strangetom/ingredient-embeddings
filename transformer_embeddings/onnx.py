#!/usr/bin/env/python3

import csv
import gzip
import os
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from tqdm import tqdm


@dataclass
class TokenVectors:
    """Tokens and their embedding vectors.

    tokens : list[str]
        List of tokens.
    vectors : np.ndarray
        NumPy array of vectors
        The dimension of the array is (seq_len, vec_size)
    """

    tokens: list[str]
    vectors: np.ndarray


class IngredientEmbeddings:
    def __init__(self, onnx_path: os.PathLike, tokenizer_path: os.PathLike):
        # Load tokenizer from file.
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Configure inference session for a CPUExecutionProvider, using default options.
        opts = ort.SessionOptions()
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def get_token_vectors(self, text: str) -> TokenVectors:
        """Return the embedding vector for each token in the input text.

        Parameters
        ----------
        text : str
            Text to return embedding vectors for.

        Returns
        -------
        TokenVectors
            TokenVectors object contains the tokens and their vectors.
        """
        # Encode the text string into token IDs and masks
        encoded = self.tokenizer.encode(text)

        # Reshape inputs to batch format: [batch_size=1, sequence_length]
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # Format the exact tensor dictionary the ONNX model expects
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
        }

        # Run CPU inference
        outputs = self.session.run(None, onnx_inputs)

        # The feature-extraction task returns token embeddings as the first output
        # Shape: [1, sequence_length, hidden_dimension]
        token_embeddings = outputs[0][0]

        # Return both the human-readable string tokens and their matching arrays
        return TokenVectors(tokens=encoded.tokens, vectors=token_embeddings)

    def get_vector(self, text: str) -> TokenVectors:
        """Return single embedding vector for the input text.

        Parameters
        ----------
        text : str
            Text to return single embedding vector for.

        Returns
        -------
        TokenVectors
            TokenVectors object contains the text and it's vector.
        """
        encoded = self.tokenizer.encode(text)

        # Reshape inputs to batch format: [batch_size=1, sequence_length]
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # Format the exact tensor dictionary the ONNX model expects
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
        }

        # Run CPU inference
        outputs = self.session.run(None, onnx_inputs)

        # The feature-extraction task returns token embeddings as the first output
        # Shape: [1, sequence_length, hidden_dimension]
        token_embeddings = outputs[0]

        # Return the mean pooled vector for the full input text.
        return TokenVectors(
            tokens=[text],
            vectors=self._mean_pooling(token_embeddings, onnx_inputs["attention_mask"]),
        )

    def _mean_pooling(
        self, last_hidden_state: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Combine embedding vectors for tokens in text into a single embedding vector
        using mean pooling.

        Parameters
        ----------
        last_hidden_state : np.ndarray
            Embedding vectors for tokens.
            Dimensions: (1, seq_len, vec_size)
        attention_mask : np.ndarray
            Attention mask for tokens, indicating where the non-padding tokens are.

        Returns
        -------
        np.ndarray
            Embedding vector.
        """
        # Expand attention_mask from (1, seq_len) to (1, seq_len, vec_size)
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(
            input_mask_expanded, last_hidden_state.shape
        )

        # Zero out padding token vectors
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)

        # Count non-padded tokens safely
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask


def cosine_similarity(vec1, vec2) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    embeddings = IngredientEmbeddings(
        "onnx_output/model_quantized.onnx", "onnx_output/tokenizer.json"
    )

    load_start_time = time.monotonic()
    fdc_ingredients = []
    with gzip.open(
        "../ingredient-parser/ingredient_parser/en/data/fdc_ingredients.csv.gz", "rt"
    ) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=11371):
            fdc_ingredients.append(
                {
                    "fdc_id": int(row["fdc_id"]),
                    "description": row["description"],
                    "vector": embeddings.get_vector(
                        row["description"]
                    ).vectors.flatten(),
                }
            )
    print(
        f"Loaded FDC ingredients in {time.monotonic() - load_start_time:.3f} seconds."
    )

    test_ingredient = "eggs"
    test_ingredient_vector = embeddings.get_vector(test_ingredient).vectors.flatten()

    match_start_time = time.monotonic()
    scores = []
    for fdc in fdc_ingredients:
        score = cosine_similarity(test_ingredient_vector, fdc["vector"])
        scores.append((score, fdc["fdc_id"], fdc["description"]))

    print(f"Best matches for '{test_ingredient}':")
    for score, fdc_id, description in sorted(scores, key=lambda x: x[0], reverse=True)[
        :10
    ]:
        print(f"{score:.4f} :: {description}")
    print(
        f"Matched FDC ingredients in {time.monotonic() - match_start_time:.3f} seconds."
    )
