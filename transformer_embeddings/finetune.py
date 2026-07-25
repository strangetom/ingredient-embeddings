#!/usr/bin/env/python3

import gc
import os
from typing import Generator

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import sentence_transformers.sentence_transformer.losses as losses
from optimum.onnxruntime import ORTQuantizer, ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

TrainingSentence = dict[str, str | list[float]]


def training_data_generator(
    corpus: list[str], teacher_model: SentenceTransformer, batch_size: int = 64
) -> Generator[TrainingSentence]:
    """Generator that yields TrainingSentence dicts.

    Encoding of the corpus is batched to limit the amount of GPU memory used.

    Parameters
    ----------
    corpus : list[str]
        Corpus of training sentences.
    teacher_model : SentenceTransformer
        Teacher model used to obtain embedding vectors for training sentences.
    batch_size : int, optional
        Number of sentences to encode at the same time.

    Yields
    ------
    Generator[TrainingSentence]
        TrainingSentence dicts.
    """
    total_len = len(corpus)

    for i in range(0, total_len, batch_size):
        batch_text = corpus[i : i + batch_size]

        # Generate target vectors just for this small slice
        batch_embeddings = teacher_model.encode(batch_text, convert_to_numpy=True)

        # Unzip the batch elements instantly in local RAM right at the yield point
        # This keeps the Apache Arrow engine perfectly happy
        for text, vector in zip(batch_text, batch_embeddings):
            yield {"text": text, "label": vector.tolist()}


class FineTuneMdbrLeafMT:
    def __init__(
        self,
        teacher_model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        student_model_name: str = "MongoDB/mdbr-leaf-mt",
    ):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name

    def _garbage_collect(self, model: SentenceTransformer) -> None:
        """Clear GPU memory for given model.

        Parameters
        ----------
        model : SentenceTransformer
            Model to clear from GPU memory.
        """
        del model

        # Force Python's garbage collector to destroy the unreferenced objects
        gc.collect()

        # Force PyTorch to release the freed VRAM back to the GPU OS memory pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_training_dataset(
        self, corpus: list[str], output_directory: str = "data/bert_training_data"
    ) -> Dataset:
        """Generate training dataset and save to output directory

        Parameters
        ----------
        corpus : list[str]
            Corpus of training sentences.
        output_directory : str, optional
            Directory to save dataset to.

        Returns
        -------
        Dataset
            Training dataset.
        """
        teacher_model = SentenceTransformer(self.teacher_model_name)

        train_dataset = Dataset.from_generator(
            training_data_generator,
            gen_kwargs={
                "corpus": corpus,
                "teacher_model": teacher_model,
                "batch_size": 512,
            },
        )
        train_dataset.save_to_disk(output_directory, "1GB")

        self._garbage_collect(teacher_model)
        return train_dataset

    def fine_tune(
        self,
        training_dataset: Dataset,
        checkpoint_directory: str = "leaf_training_output",
        output_model: str = "ingredient-leaf-mt",
        use_checkpoint: bool = False,
    ):
        """Fine tune student model using teacher model and provided training dataset.

        Parameters
        ----------
        training_dataset : Dataset
            Training dataset used to fine tune student model.
        output_directory : str, optional
            Output directory for checkpoints.
        output_model : str, optional
            Directory for fine-tuned model.
        use_checkpoint : bool, optional
            If True and a checkpoint exists, continue training from checkpoint.
        """
        student_model = SentenceTransformer(self.student_model_name)
        train_loss = losses.MSELoss(model=student_model)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=checkpoint_directory,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=1000,
            save_strategy="steps",
            save_steps=500,  # save a checkpoint every steps
            save_total_limit=2,  # max of 2 checkpoints
        )

        print("Training...")
        trainer = SentenceTransformerTrainer(
            model=student_model,
            args=training_args,
            train_dataset=training_dataset,
            loss=train_loss,
        )

        # Check if checkpoint exists.
        has_checkpoint = os.path.isdir(checkpoint_directory) and any(
            "checkpoint" in d for d in os.listdir(checkpoint_directory)
        )
        if use_checkpoint and has_checkpoint:
            print(
                f"Interruption detected! Resuming training from last checkpoint in {checkpoint_directory}"
            )
            trainer.train(resume_from_checkpoint=True)
        else:
            print("Starting fresh fine-tuning run.")
            trainer.train()

        student_model.save_pretrained(output_model)
        print("Knowledge distillation complete using an in-memory dataset!")

    def export_onnx(self, model_directory: str, onnx_directory: str):
        """Export model to ONNX format.

        Parameters
        ----------
        model_directory : str
            Directory containing fine-tuned model.
        onnx_directory : str
            Directory to save ONNX model and tokenizer to.
        """
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_directory,
            export=True,
        )
        # Save graph and configuration structures
        model.save_pretrained(onnx_directory)
        # Save standard text token handling assets
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        tokenizer.save_pretrained(onnx_directory)

    def quantize_onnx(self, onnx_directory: str, onnx_model_name: str):
        """Quantize ONNX model to optimize for CPU deployment.

        Parameters
        ----------
        onnx_directory : str
            Directory containing ONNX model.
        onnx_model_name : str
            Name of ONNX model within onnx_directory.
        """
        quantizer = ORTQuantizer.from_pretrained(
            onnx_directory, file_name=onnx_model_name
        )

        # Use standard Dynamic Quantization (ideal for CPU deployment)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

        # Export the highly compressed 45MB model file
        quantizer.quantize(
            save_dir=onnx_directory,
            quantization_config=qconfig,
            file_suffix="quantized",
        )
