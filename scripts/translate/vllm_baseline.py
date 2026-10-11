#!/usr/bin/env python3
"""
Bidirectional Arabic Translation: MSA ↔ Syrian Dialect (Shami)

This script performs translation between Modern Standard Arabic (MSA) and
Syrian dialect (Shami) using vLLM for efficient batch inference.

Usage:
    python scripts/translate/vllm_baseline.py --input data/shami_v2.csv --model "Qwen/Qwen3-0.6B"
    python scripts/translate/vllm_baseline.py --config configs/vllm.yaml
"""

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams


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

    # Model settings
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"

    # Inference parameters
    temperature: float = 0.3
    max_tokens: int = 256
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Processing settings
    batch_size: int = 32
    input_file: str = "data/shami_v2.csv"
    output_dir: str = "outputs"

    # Prompt templates
    shami_to_msa_template: str = (
        "Translate the following Syrian Arabic (Shami) text to Modern Standard Arabic (MSA).\n"
        "Only provide the translation, nothing else.\n\n"
        "Syrian Arabic: {text}\n"
        "MSA Translation:"
    )
    msa_to_shami_template: str = (
        "Translate the following Modern Standard Arabic (MSA) text to Syrian Arabic (Shami dialect).\n"
        "Only provide the translation, nothing else.\n\n"
        "MSA: {text}\n"
        "Syrian Arabic Translation:"
    )

    # Column names (for flexibility with different CSV formats)
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
    """Bidirectional Arabic translator using vLLM."""

    def __init__(self, config: TranslationConfig):
        """Initialize the translator with given configuration."""
        self.config = config
        self.model: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None

    def load_model(self) -> None:
        """Load the vLLM model with configured parameters."""
        logger.info(f"Loading model: {self.config.model_name}")
        try:
            self.model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                dtype=self.config.dtype,
                trust_remote_code=True,
            )
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def translate_batch(
        self, texts: list[str], prompt_template: str
    ) -> list[str]:
        """
        Translate a batch of texts using the specified prompt template.

        Args:
            texts: List of source texts to translate
            prompt_template: Template with {text} placeholder

        Returns:
            List of translated texts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Format prompts
        prompts = [prompt_template.format(text=text) for text in texts]

        # Generate translations
        try:
            outputs = self.model.generate(prompts, self.sampling_params)
            translations = [
                output.outputs[0].text.strip() for output in outputs
            ]
            return translations
        except Exception as e:
            logger.error(f"Translation batch failed: {e}")
            return ["[ERROR]"] * len(texts)

    def _process_in_batches(
        self,
        records: list[TranslationRecord],
        prompt_template: str,
        description: str,
    ) -> list[TranslationRecord]:
        """Process records in batches with progress tracking."""
        batch_size = self.config.batch_size
        total_batches = (len(records) + batch_size - 1) // batch_size

        with tqdm(total=len(records), desc=description) as pbar:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                texts = [r.source_text for r in batch]

                translations = self.translate_batch(texts, prompt_template)

                for record, translation in zip(batch, translations):
                    record.prediction = translation

                pbar.update(len(batch))

        return records

    def process_shami_to_msa(
        self, data: list[dict]
    ) -> list[TranslationRecord]:
        """
        Generate predictions for Shami → MSA translation.

        Args:
            data: List of dictionaries from input CSV

        Returns:
            List of TranslationRecord with predictions
        """
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

    def process_msa_to_shami(
        self, data: list[dict]
    ) -> list[TranslationRecord]:
        """
        Generate predictions for MSA → Shami translation.

        Args:
            data: List of dictionaries from input CSV

        Returns:
            List of TranslationRecord with predictions
        """
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
    """
    Load input CSV file.

    Args:
        file_path: Path to input CSV file

    Returns:
        List of dictionaries representing rows
    """
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
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        raise


def save_results(
    records: list[TranslationRecord],
    output_path: str,
    direction: str,
    config: TranslationConfig,
) -> None:
    """
    Save translation results to CSV.

    Args:
        records: List of TranslationRecord objects
        output_path: Path for output CSV file
        direction: Either "shami-to-msa" or "msa-to-shami"
        config: Translation configuration
    """
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
    else:  # msa-to-shami
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

    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(row_builder(record))
        logger.info(f"Successfully saved {len(records)} records")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bidirectional Arabic Translation (MSA ↔ Shami)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for distributed inference",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)",
    )

    # Inference parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )

    # Processing settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/shami_v2.csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for result files",
    )

    # Direction selection
    parser.add_argument(
        "--direction",
        type=str,
        choices=["both", "shami-to-msa", "msa-to-shami"],
        default="both",
        help="Translation direction(s) to perform",
    )

    # Utility options
    parser.add_argument(
        "--generate-config",
        type=str,
        help="Generate a sample config file and exit",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TranslationConfig:
    """Build configuration from command-line arguments or config file."""
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        return TranslationConfig.from_yaml(args.config)

    return TranslationConfig(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        input_file=args.input,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Handle config generation
    if args.generate_config:
        config = TranslationConfig()
        config.to_yaml(args.generate_config)
        logger.info(f"Generated sample config: {args.generate_config}")
        return

    # Build configuration
    config = build_config(args)

    # Ensure output directory exists
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    data = load_input_data(config.input_file)

    # Initialize translator and load model
    translator = ArabicTranslator(config)
    translator.load_model()

    # Process translations based on direction
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
