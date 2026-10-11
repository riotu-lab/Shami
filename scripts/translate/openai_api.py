#!/usr/bin/env python3
"""
Bidirectional Arabic Translation: MSA ↔ Syrian Dialect (Shami)

This script performs translation between Modern Standard Arabic (MSA) and
Syrian dialect (Shami) using the OpenAI API.

Usage:
    python scripts/translate/openai_api.py --input data/shami_v2.csv
    python scripts/translate/openai_api.py --config configs/openai_api.yaml
"""

import argparse
import csv
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI
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

    # OpenAI API settings
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-2025-04-14"

    # Inference parameters
    temperature: float = 0.3
    max_tokens: int = 256

    # Processing settings
    input_file: str = "data/shami_v2.csv"
    output_dir: str = "outputs"
    retry_attempts: int = 3
    retry_delay: float = 1.0

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

    # CSV Column names
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
        """Save configuration to YAML file (API keys are left out; they come from the environment)."""
        data = {k: v for k, v in self.__dict__.items() if k != "openai_api_key"}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


@dataclass
class TranslationRecord:
    """A single translation record."""

    sent_id: str
    source_text: str
    prediction: str = ""
    ground_truth: str = ""


# =============================================================================
# API Client Classes
# =============================================================================

class BaseAPIClient(ABC):
    """Abstract base class for API clients."""

    def __init__(self, config: TranslationConfig):
        self.config = config
        self.client: Optional[OpenAI] = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the API client."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name being used."""
        pass

    def translate(self, text: str, prompt_template: str) -> str:
        """
        Translate a single text using the API.

        Args:
            text: Source text to translate
            prompt_template: Template with {text} placeholder

        Returns:
            Translated text
        """
        if self.client is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        prompt = prompt_template.format(text=text)

        for attempt in range(self.config.retry_attempts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.get_model_name(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return completion.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"All retry attempts failed for text: {text[:50]}...")
                    return "[ERROR]"


class OpenAIClient(BaseAPIClient):
    """OpenAI API client."""

    def initialize(self) -> None:
        """Initialize the OpenAI client."""
        logger.info(f"Initializing OpenAI client for model: {self.config.openai_model}")
        
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key is required. Set 'openai_api_key' in config or use --openai-api-key")
        
        self.client = OpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )
        logger.info("OpenAI client initialized successfully")

    def get_model_name(self) -> str:
        return self.config.openai_model


def create_api_client(config: TranslationConfig) -> BaseAPIClient:
    """
    Create the OpenAI API client.

    Args:
        config: Translation configuration

    Returns:
        Initialized API client
    """
    client = OpenAIClient(config)
    client.initialize()
    return client


# =============================================================================
# Translator Class
# =============================================================================

class ArabicTranslator:
    """Bidirectional Arabic translator."""

    def __init__(self, config: TranslationConfig):
        """Initialize the translator with given configuration."""
        self.config = config
        self.client: Optional[BaseAPIClient] = None

    def initialize(self) -> None:
        """Initialize the API client."""
        self.client = create_api_client(self.config)

    def _process_records(
        self,
        records: list[TranslationRecord],
        prompt_template: str,
        description: str,
        output_path: str,
        direction: str,
    ) -> list[TranslationRecord]:
        """Process records with progress tracking and incremental saving."""
        for i, record in enumerate(tqdm(records, desc=description)):
            record.prediction = self.client.translate(record.source_text, prompt_template)
            
            # Save after each translation
            self._save_progress(records[:i+1], output_path, direction)
        
        return records

    def _save_progress(
        self,
        records: list[TranslationRecord],
        output_path: str,
        direction: str,
    ) -> None:
        """Save current progress to CSV."""
        if direction == "shami-to-msa":
            fieldnames = [
                self.config.id_column,
                self.config.shami_column,
                "MSA-prediction",
                "MSA-Ground-Truth",
            ]
            row_builder = lambda r: {
                self.config.id_column: r.sent_id,
                self.config.shami_column: r.source_text,
                "MSA-prediction": r.prediction,
                "MSA-Ground-Truth": r.ground_truth,
            }
        else:
            fieldnames = [
                self.config.id_column,
                self.config.msa_column,
                "Shami-prediction",
                "Shami-Ground-Truth",
            ]
            row_builder = lambda r: {
                self.config.id_column: r.sent_id,
                self.config.msa_column: r.source_text,
                "Shami-prediction": r.prediction,
                "Shami-Ground-Truth": r.ground_truth,
            }

        try:
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow(row_builder(record))
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")

    def process_shami_to_msa(self, data: list[dict], output_path: str) -> list[TranslationRecord]:
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

        return self._process_records(
            records,
            self.config.shami_to_msa_template,
            "Shami → MSA",
            output_path,
            "shami-to-msa",
        )

    def process_msa_to_shami(self, data: list[dict], output_path: str) -> list[TranslationRecord]:
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

        return self._process_records(
            records,
            self.config.msa_to_shami_template,
            "MSA → Shami",
            output_path,
            "msa-to-shami",
        )


# =============================================================================
# Utility Functions
# =============================================================================

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
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bidirectional Arabic Translation (MSA ↔ Shami) using the OpenAI API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # OpenAI settings
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key")
    parser.add_argument("--openai-model", type=str, default="gpt-4.1-2025-04-14", help="OpenAI model name")

    # Common settings
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--input", type=str, default="data/shami_v2.csv", help="Input CSV file path")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument(
        "--direction",
        type=str,
        choices=["both", "shami-to-msa", "msa-to-shami"],
        default="both",
        help="Translation direction(s)",
    )
    parser.add_argument("--generate-config", type=str, help="Generate a sample config file and exit")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TranslationConfig:
    """Build configuration from arguments or config file."""
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        return TranslationConfig.from_yaml(args.config)

    config = TranslationConfig(
        openai_model=args.openai_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        input_file=args.input,
        output_dir=args.output_dir,
    )

    if args.openai_api_key:
        config.openai_api_key = args.openai_api_key

    return config


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
    translator.initialize()

    direction = args.direction if not args.config else "both"

    if direction in ("both", "shami-to-msa"):
        shami_to_msa_path = str(output_dir / "openai-shami-to-msa.csv")
        translator.process_shami_to_msa(data, shami_to_msa_path)
        logger.info(f"Shami → MSA results saved to: {shami_to_msa_path}")

    if direction in ("both", "msa-to-shami"):
        msa_to_shami_path = str(output_dir / "openai-msa-to-shami.csv")
        translator.process_msa_to_shami(data, msa_to_shami_path)
        logger.info(f"MSA → Shami results saved to: {msa_to_shami_path}")

    logger.info("Translation complete!")


if __name__ == "__main__":
    main()