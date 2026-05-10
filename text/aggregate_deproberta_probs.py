import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ── Configure these paths for your environment ─────────────────────────────────
@dataclass
class AggregationConfig:
    """Centralized configuration for the aggregation script."""
    input_dir:  Path = Path("CONFIGURE_ME/deproberta_predictions")  # from deproberta_inference.py
    output_csv: Path = Path("CONFIGURE_ME/features/deproberta_probs.csv")
# ──────────────────────────────────────────────────────────────────────────────


def aggregate_json_to_csv(config: AggregationConfig):
    """Scans a directory of JSON files, aggregates their contents into a single DataFrame."""
    records: List[Dict] = []

    json_files = list(config.input_dir.glob("*.json"))
    if not json_files:
        logging.warning(f"No JSON files found in the input directory: {config.input_dir}")
        return

    logging.info(f"Found {len(json_files)} JSON files to process.")

    for fpath in tqdm(json_files, desc="Aggregating JSON files"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                probs = data["depression_prediction"]
                patient_id = data["patient_id"]
                _norm = {" ".join(str(k).lower().split()): v for k, v in probs.items()}

                record = {
                    "patient_id": patient_id,
                    "prob_severe": next((_norm[a] for a in ("severe",) if a in _norm), 0.0),
                    "prob_moderate": next((_norm[a] for a in ("moderate",) if a in _norm), 0.0),
                    "prob_not_depression": next((_norm[a] for a in ("not depressed", "not depression") if a in _norm), 0.0),
                }
                records.append(record)

        except FileNotFoundError:
            logging.error(f"File not found during processing: {fpath}")
        except json.JSONDecodeError:
            logging.error(f"Could not parse JSON from file: {fpath}")
        except KeyError as e:
            logging.error(f"Missing expected key {e} in file: {fpath}")

    if not records:
        logging.error("No valid records could be processed. The output CSV will not be created.")
        return

    logging.info(f"Successfully processed {len(records)} records. Creating DataFrame...")
    df = pd.DataFrame(records)

    df["patient_id"] = pd.to_numeric(df["patient_id"], errors='coerce')
    df.dropna(subset=["patient_id"], inplace=True)
    df["patient_id"] = df["patient_id"].astype(int)
    df = df.sort_values(by="patient_id").reset_index(drop=True)

    try:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.output_csv, index=False)
        logging.info(f"Successfully saved aggregated data to: {config.output_csv}")
    except Exception as e:
        logging.error(f"Failed to save the final CSV file: {e}")


if __name__ == "__main__":
    aggregation_config = AggregationConfig()
    aggregate_json_to_csv(aggregation_config)
