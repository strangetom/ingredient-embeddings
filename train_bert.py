#!/usr/bin/env/python3

from datasets import Dataset

from transformer_embeddings import FineTuneMdbrLeafMT

OUTPUT_MODEL = "ingredient-leaf-mt"
OUTPUT_ONNX = "onnx_output"

if __name__ == "__main__":
    training_data = Dataset.load_from_disk("data/bert_training_data")
    finetuner = FineTuneMdbrLeafMT()
    finetuner.fine_tune(training_data, output_model=OUTPUT_MODEL)
    finetuner.export_onnx(OUTPUT_MODEL, OUTPUT_ONNX)
    finetuner.quantize_onnx(OUTPUT_ONNX, "model.onnx")
