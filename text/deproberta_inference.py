import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)

# ── Configure these paths for your environment ─────────────────────────────────
@dataclass
class InferenceConfig:
    """A single place to store all configuration for the inference pipeline."""
    model_path: Path = Path("CONFIGURE_ME/models/deproberta_finetuned")
    input_dir:  Path = Path("CONFIGURE_ME/summaries")       # JSON files from llm_extract_summaries.py
    output_dir: Path = Path("CONFIGURE_ME/deproberta_predictions")

    batch_size: int = 8
    text_key: str = "summary"
    max_token_length: int = 512
# ──────────────────────────────────────────────────────────────────────────────


class TextClassifier:
    """A class to handle model loading and run predictions efficiently."""

    def __init__(self, model_path: Path):
        """Loads the model and tokenizer onto the correct device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading model from: {model_path}")
        logging.info(f"Using device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.label_map = self.model.config.id2label
            logging.info("Model and tokenizer loaded successfully.")

        except OSError as e:
            logging.error(f"Could not load model from {model_path}. Check the path. Error: {e}")
            raise

    def predict_batch(self, texts: List[str], max_length: int) -> List[Dict[str, float]]:
        """Performs inference on a batch of texts."""
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        results = []
        for prob_set in probabilities:
            prediction = {self.label_map[i]: float(prob) for i, prob in enumerate(prob_set)}
            results.append(prediction)
        return results

def run_inference(config: InferenceConfig):
    """Orchestrates the entire inference pipeline."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    classifier = TextClassifier(config.model_path)

    # --- Stage 1: Read all data from disk. ---
    logging.info(f"Scanning for input files in: {config.input_dir}")
    files_to_process: List[Tuple[Path, str]] = []
    for path in config.input_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data.get(config.text_key, "").strip()
                if text:
                    files_to_process.append((path, text))
                else:
                    logging.warning(f"No text found in file, skipping: {path.name}")
        except Exception as e:
            logging.error(f"Could not read file {path.name}, skipping. Error: {e}")

    if not files_to_process:
        logging.warning("No valid files found to process. Exiting.")
        return
    logging.info(f"Found {len(files_to_process)} files to classify.")

    # --- Stage 2: Process all data in batches. ---
    all_predictions = []
    for i in tqdm(range(0, len(files_to_process), config.batch_size), desc="Classifying Texts"):
        batch_slice = files_to_process[i : i + config.batch_size]
        batch_texts = [text for path, text in batch_slice]

        batch_preds = classifier.predict_batch(batch_texts, config.max_token_length)
        all_predictions.extend(batch_preds)

    # --- Stage 3: Write all results to disk. ---
    logging.info(f"Writing {len(all_predictions)} output files...")
    for i, (input_path, _) in enumerate(files_to_process):
        try:
            patient_id = input_path.stem.split("_")[0]
            output_data = {
                "patient_id": patient_id,
                "depression_prediction": all_predictions[i]
            }
            output_path = config.output_dir / input_path.name
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to write output for {input_path.name}: {e}")

    logging.info(f"Inference complete. Outputs saved to: {config.output_dir}")


if __name__ == "__main__":
    config = InferenceConfig()
    run_inference(config)
