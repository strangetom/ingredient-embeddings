#!/usr/bin/env/python3

import json
import os
import sqlite3
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

from .finetune import FineTuneMdbrLeafMT
from .preprocess import preprocess_recipe


def fine_tune_embeddings(
    dataset: os.PathLike, model_output: os.PathLike, onnx_output: os.PathLike
):
    """Fine tune embeddings

    Parameters
    ----------
    dataset : os.PathLike
        Directory containing dataset.
    model_output : os.PathLike
        Directory to save fine tuned model to.
    onnx_output : os.PathLike
        Directory to save onnx exported model to.
    """
    training_data = Dataset.load_from_disk(dataset)

    finetuner = FineTuneMdbrLeafMT()
    finetuner.fine_tune(training_data, output_model=model_output)

    corpus = [str(s) for s in list(training_data.data["text"])]
    finetuner.prune_vocabulary(corpus, model_output, model_output)

    finetuner.export_onnx(model_output, onnx_output)
    finetuner.quantize_onnx(onnx_output, "model.onnx")


def generate_dataset(source: Path, fdc_dataset: Path, output: Path):
    """Generate dataset for fine tuning embeddings model from source database.

    WARNING: This takes ages.

    Parameters
    ----------
    source : Path
        Source database.
    fdc_dataset : Path
        FDC ingredient database.
    output : Path
        Output directory for generated dataset.
    """
    corpus = []
    with sqlite3.connect(source) as conn:
        c = conn.cursor()

        # Get table names
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [name for (name,) in c.fetchall()]

        # Get data for each
        for table in tables:
            print(f"Processing recipes from '{table}'.")
            c.execute(f"SELECT ingredients, instructions FROM {table}")
            for ingredients, instructions in tqdm(c.fetchall()):
                corpus.extend([preprocess_recipe(i) for i in json.loads(ingredients)])
                corpus.extend([preprocess_recipe(i) for i in json.loads(instructions)])

    print(f"Corpus contains {len(corpus):,} sentences.")

    finetuner = FineTuneMdbrLeafMT()
    finetuner.generate_training_dataset(corpus, output)
    print(f"Dataset saved to '{output}'.")
