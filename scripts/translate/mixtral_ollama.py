#!/usr/bin/env python3
"""
Bidirectional Arabic Translation: MSA ↔ Syrian Dialect (Shami)

This script performs translation between Modern Standard Arabic (MSA) and
Syrian dialect (Shami) using Ollama with Mixtral model.

Usage:
    python scripts/translate/mixtral_ollama.py --input data/shami_v2.csv
    python scripts/translate/mixtral_ollama.py --config configs/mixtral_ollama.yaml
"""

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import yaml
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("translation.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """Configuration for the translation pipeline."""

    # Ollama settings
    model_name: str = "mixtral:8x22b"
    ollama_url: str = "http://localhost:11434/api/generate"

    # GPU settings
    gpu_ids: str = ""  # Comma-separated GPU IDs (e.g., "0,1,2") - empty means use all
    num_gpu: int = -1  # Number of layers to offload to GPU (-1 = all, 0 = CPU only)

    # Inference parameters
    temperature: float = 0.3
    max_tokens: int = 256
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Processing settings
    batch_size: int = 8
    input_file: str = "data/shami_v2.csv"
    output_dir: str = "outputs"

    # Prompt templates
    shami_to_msa_template: str = (
        "Translate from Syrian Arabic (Shami) to Modern Standard Arabic (MSA).\n\n"
        "RULES:\n"
        "1) Output the MSA translation ONLY. No other text.\n"
        "2) Output Arabic ONLY. No English.\n"
        "3) Do NOT add explanations, notes, alternatives, or examples.\n"
        "4) Write WITHOUT diacritics/tashkeel/harakat.\n"
        "5) Keep names, numbers, dates, URLs, and emojis unchanged.\n\n"
        "Syrian Arabic: {text}\n"
        "MSA Translation:"
    )
    msa_to_shami_template: str = (
        "Translate from Modern Standard Arabic (MSA) to Syrian Arabic (Shami dialect).\n\n"
        "RULES:\n"
        "1) Output the Shami translation ONLY. No other text.\n"
        "2) Output Arabic ONLY. No English.\n"
        "3) Do NOT add explanations, notes, alternatives, or examples.\n"
        "4) Write WITHOUT diacritics/tashkeel/harakat.\n"
        "5) Keep names, numbers, dates, URLs, and emojis unchanged.\n\n"
        "MSA: {text}\n"
        "Syrian Arabic Translation:"
    )

    # CSV Column Names
    id_column: str = "sentID.BTEC"
    shami_column: str = "Shami-sent"
    msa_column: str = "MSA-sent"

    @classmethod
    def from_yaml(cls, path: str) -> "TranslationConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)


@dataclass
class TranslationRecord:
    """A single translation record."""

    sent_id: str
    source_text: str
    prediction: str = ""
    ground_truth: str = ""


class ArabicTranslator:
    """Bidirectional Arabic translator using Ollama."""

    def __init__(self, config: TranslationConfig):
        """Initialize the translator with given configuration."""
        self.config = config
        self._setup_gpus()

    def _setup_gpus(self) -> None:
        """Configure GPU visibility via environment variable."""
        if self.config.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu_ids
            logger.info(f"Using GPUs: {self.config.gpu_ids}")
        else:
            logger.info("Using all available GPUs")

    def _call_ollama(self, prompt: str) -> str:
        """Make a single call to Ollama API."""
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repetition_penalty,
            "num_predict": self.config.max_tokens,
        }
        
        # Add GPU layer setting if specified
        if self.config.num_gpu >= 0:
            options["num_gpu"] = self.config.num_gpu

        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        try:
            response = requests.post(self.config.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return "[ERROR]"

    def translate_batch(self, texts: list[str], prompt_template: str) -> list[str]:
        """
        Translate a batch of texts using the specified prompt template.

        Args:
            texts: List of source texts to translate
            prompt_template: Template with {text} placeholder

        Returns:
            List of translated texts
        """
        translations = []
        for text in texts:
            prompt = prompt_template.format(text=text)
            translation = self._call_ollama(prompt)
            translations.append(translation)
        return translations

    def _process_in_batches(
        self,
        records: list[TranslationRecord],
        prompt_template: str,
        description: str,
    ) -> list[TranslationRecord]:
        """Process records in batches with progress tracking."""
        batch_size = self.config.batch_size

        with tqdm(total=len(records), desc=description) as pbar:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                texts = [r.source_text for r in batch]

                translations = self.translate_batch(texts, prompt_template)

                for record, translation in zip(batch, translations):
                    record.prediction = translation

                pbar.update(len(batch))

        return records

    def process_shami_to_msa(self, data: list[dict]) -> list[TranslationRecord]:
        """Generate predictions for Shami → MSA translation."""
        logger.info("Processing Shami → MSA translations")

        records = [
            TranslationRecord(
                sent_id=row[self.config.id_column],
                source_text=row[self.config.shami_column],
                ground_truth=row[self.config.msa_column],
            )
            for row in data
        ]

        return self._process_in_batches(
            records,
            self.config.shami_to_msa_template,
            "Shami → MSA",
        )

    def process_msa_to_shami(self, data: list[dict]) -> list[TranslationRecord]:
        """Generate predictions for MSA → Shami translation."""
        logger.info("Processing MSA → Shami translations")

        records = [
            TranslationRecord(
                sent_id=row[self.config.id_column],
                source_text=row[self.config.msa_column],
                ground_truth=row[self.config.shami_column],
            )
            for row in data
        ]

        return self._process_in_batches(
            records,
            self.config.msa_to_shami_template,
            "MSA → Shami",
        )


def load_input_data(file_path: str) -> list[dict]:
    """Load input CSV file."""
    logger.info(f"Loading input file: {file_path}")
    try:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        logger.info(f"Loaded {len(data)} records")
        return data
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise


def save_results(
    records: list[TranslationRecord],
    output_path: str,
    direction: str,
    config: TranslationConfig,
) -> None:
    """Save translation results to CSV."""
    logger.info(f"Saving results to: {output_path}")

    if direction == "shami-to-msa":
        fieldnames = [
            config.id_column,
            config.shami_column,
            "MSA-prediction",
            "MSA-Ground-Truth",
        ]
        row_builder = lambda r: {
            config.id_column: r.sent_id,
            config.shami_column: r.source_text,
            "MSA-prediction": r.prediction,
            "MSA-Ground-Truth": r.ground_truth,
        }
    else:
        fieldnames = [
            config.id_column,
            config.msa_column,
            "Shami-prediction",
            "Shami-Ground-Truth",
        ]
        row_builder = lambda r: {
            config.id_column: r.sent_id,
            config.msa_column: r.source_text,
            "Shami-prediction": r.prediction,
            "Shami-Ground-Truth": r.ground_truth,
        }

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(row_builder(record))
    logger.info(f"Successfully saved {len(records)} records")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bidirectional Arabic Translation (MSA ↔ Shami) using Ollama",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--model", type=str, default="mixtral:8x22b", help="Ollama model name")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate", help="Ollama API URL")
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU IDs to use (e.g., '0,1,2'). Empty = all GPUs")
    parser.add_argument("--num-gpu", type=int, default=-1, help="Number of layers to offload to GPU (-1=all, 0=CPU only)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--input", type=str, default="data/shami_v2.csv", help="Input CSV file path")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--direction",
        type=str,
        choices=["both", "shami-to-msa", "msa-to-shami"],
        default="both",
        help="Translation direction(s)",
    )
    parser.add_argument("--generate-config", type=str, help="Generate sample config file and exit")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TranslationConfig:
    """Build configuration from arguments or config file."""
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        return TranslationConfig.from_yaml(args.config)

    return TranslationConfig(
        model_name=args.model,
        ollama_url=args.ollama_url,
        gpu_ids=args.gpu_ids,
        num_gpu=args.num_gpu,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        input_file=args.input,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    if args.generate_config:
        config = TranslationConfig()
        config.to_yaml(args.generate_config)
        logger.info(f"Generated sample config: {args.generate_config}")
        return

    config = build_config(args)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_input_data(config.input_file)

    translator = ArabicTranslator(config)

    direction = args.direction if not args.config else "both"

    if direction in ("both", "shami-to-msa"):
        shami_to_msa_results = translator.process_shami_to_msa(data)
        save_results(
            shami_to_msa_results,
            str(output_dir / "shami-to-msa.csv"),
            "shami-to-msa",
            config,
        )

    if direction in ("both", "msa-to-shami"):
        msa_to_shami_results = translator.process_msa_to_shami(data)
        save_results(
            msa_to_shami_results,
            str(output_dir / "msa-to-shami.csv"),
            "msa-to-shami",
            config,
        )

    logger.info("Translation complete!")


if __name__ == "__main__":
    main()