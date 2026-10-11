#!/usr/bin/env python3
"""
Bidirectional Arabic Translation using vLLM with JAIS Model

Supports:
- Local JAIS model loading
- YAML/JSON configuration
- Simple text output (non-JSON) for better compatibility with JAIS
- Configurable column names
"""

import argparse
import logging
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import json
import os

# Fix for CUDA all-reduce errors with tensor parallelism
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from vllm import LLM, SamplingParams


@dataclass
class TranslationConfig:
    # Model settings
    model_name: str = "jais-70b-local"
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"

    # Inference settings
    temperature: float = 0.0
    max_tokens: int = 32
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.3
    enforce_eager: bool = True  # Disable CUDA graphs (fixes all-reduce errors)

    # Processing settings
    batch_size: int = 64

    # File paths
    input_file: str = "data/shami_v2.csv"
    output_dir: str = "outputs"

    # Column names
    id_column: str = "sentID.BTEC"
    shami_column: str = "Shami-sent"
    msa_column: str = "MSA-sent"

    # Prompt templates
    shami_to_msa_template: str = """You are a translation engine. Convert Syrian Arabic (Shami) to Modern Standard Arabic (MSA).
HARD CONSTRAINTS (must follow):
- Output ONLY the MSA translation. Nothing else.
- Output Arabic script ONLY. No English letters/words at all.
- No diacritics/harakat/tashkeel and no decorative characters (no َ ً ُ ٌ ِ ٍ ْ ّ and no ـ).
- Do NOT add explanations, notes, labels, alternatives, or examples.
- Do NOT add or remove sentences. Keep the same meaning and same content.
- Keep names, numbers, dates, symbols, and emojis unchanged.
STYLE:
- Use clear, natural MSA (not dialect).
- Prefer standard vocabulary over regional words.
- Preserve tone (polite/angry/joking) without adding new content.
Shami: {text}
MSA:"""

    msa_to_shami_template: str = """You are a translation engine. Convert Modern Standard Arabic (MSA) into Syrian Arabic (Shami).
HARD CONSTRAINTS (must follow):
- Output ONLY the Shami translation. Nothing else.
- Output Arabic script ONLY. No English letters/words at all.
- No diacritics/harakat/tashkeel and no decorative characters (no َ ً ُ ٌ ِ ٍ ْ ّ and no ـ).
- Do NOT add explanations, notes, labels, alternatives, or examples.
- Do NOT add or remove sentences. Keep the same meaning and same content.
- Keep names, numbers, dates, symbols, and emojis unchanged.
STYLE:
- Natural Syrian phrasing (Shami), not formal MSA.
- Translate meaning, not word-for-word.
MSA: {text}
Shami:"""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TranslationConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # Handle hyphenated keys by converting to underscore
        converted = {}
        key_mapping = {
            "model-name": "model_name",
            "tensor-parallel-size": "tensor_parallel_size",
            "gpu-memory-utilization": "gpu_memory_utilization",
            "max-tokens": "max_tokens",
            "top-p": "top_p",
            "top-k": "top_k",
            "repetition-penalty": "repetition_penalty",
            "enforce-eager": "enforce_eager",
            "batch-size": "batch_size",
            "input-file": "input_file",
            "output-dir": "output_dir",
            "id-column": "id_column",
            "shami-column": "shami_column",
            "msa-column": "msa_column",
            "shami-to-msa-template": "shami_to_msa_template",
            "msa-to-shami-template": "msa_to_shami_template",
        }
        
        for k, v in config_dict.items():
            if v is not None:
                new_key = key_mapping.get(k, k.replace("-", "_"))
                converted[new_key] = v
        
        return cls(**converted)

    @classmethod
    def from_json(cls, json_path: str) -> "TranslationConfig":
        """Load configuration from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def from_file(cls, file_path: str) -> "TranslationConfig":
        """Load configuration from YAML or JSON file based on extension."""
        path = Path(file_path)
        if path.suffix.lower() in [".yaml", ".yml"]:
            return cls.from_yaml(file_path)
        elif path.suffix.lower() == ".json":
            return cls.from_json(file_path)
        else:
            # Try YAML first, then JSON
            try:
                return cls.from_yaml(file_path)
            except Exception:
                return cls.from_json(file_path)


class ArabicTranslator:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def load_model(self):
        self.logger.info(f"Loading model: {self.config.model_name}")

        self.llm = LLM(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=self.config.dtype,
            trust_remote_code=True,
            max_num_seqs=self.config.batch_size,
            enforce_eager=self.config.enforce_eager,  # Disable CUDA graphs to avoid all-reduce errors
        )

        # Stop sequences to prevent extra text after translation
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            stop=[
                "\n\n",
                "\nShami:",
                "\nMSA:",
                "\nNote:",
                "\nTranslation:",
                "\nExplanation:",
                "Note:",
                "Explanation:",
                "###",
            ],
        )

        self.logger.info("Model loaded successfully")

    def _build_prompt(self, user_text: str) -> str:
        """
        Build prompt for JAIS model.
        JAIS uses a specific chat format with ### markers.
        """
        assert self.llm is not None
        tokenizer = self.llm.get_tokenizer()

        # Try to use chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": user_text},
            ]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                self.logger.debug(f"Chat template failed: {e}, using raw prompt")

        # Fallback: JAIS-style prompt format
        # JAIS typically uses: ### Instruction:\n{instruction}\n\n### Response:
        return f"### Instruction:\n{user_text}\n\n### Response:"

    def _clean_output(self, text: str) -> str:
        """Clean model output to extract just the translation."""
        output = text.strip()
        
        # Remove any leading/trailing markers
        for marker in ["###", "Response:", "Assistant:"]:
            if output.startswith(marker):
                output = output[len(marker):].strip()
        
        # Take only first line if multiple lines (to avoid explanations)
        lines = output.split("\n")
        if lines:
            output = lines[0].strip()
        
        return output

    def translate_batch(
        self,
        texts: List[str],
        prompt_template: str,
    ) -> List[Dict[str, str]]:
        """
        Translate a batch of texts.
        Returns list of dicts with:
        - translation: cleaned translation text
        - raw: full raw output string
        """
        if self.llm is None or self.sampling_params is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompts: List[str] = []
        for t in texts:
            user_prompt = prompt_template.format(text=t).strip()
            prompts.append(self._build_prompt(user_prompt))

        outputs = self.llm.generate(prompts, self.sampling_params)

        results: List[Dict[str, str]] = []
        for out in outputs:
            raw = out.outputs[0].text.strip()
            translation = self._clean_output(raw)
            
            results.append({
                "translation": translation,
                "raw": raw,
            })

        return results

    def load_input_data(self) -> pd.DataFrame:
        self.logger.info(f"Loading input file: {self.config.input_file}")
        df = pd.read_csv(self.config.input_file)

        required_cols = [
            self.config.id_column,
            self.config.shami_column,
            self.config.msa_column,
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.logger.info(f"Loaded {len(df)} sentences")
        return df

    def process_shami_to_msa(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Processing Shami → MSA translation")

        shami_texts = df[self.config.shami_column].tolist()
        batch_size = self.config.batch_size

        pred_texts: List[str] = []
        pred_raws: List[str] = []

        for i in tqdm(range(0, len(shami_texts), batch_size), desc="Shami→MSA"):
            batch = shami_texts[i:i + batch_size]
            batch_res = self.translate_batch(batch, self.config.shami_to_msa_template)

            pred_texts.extend([r["translation"] for r in batch_res])
            pred_raws.extend([r["raw"] for r in batch_res])

        return pd.DataFrame({
            self.config.id_column: df[self.config.id_column],
            self.config.shami_column: df[self.config.shami_column],
            "MSA-prediction": pred_texts,
            "MSA-prediction-raw": pred_raws,
            "MSA-Ground-Truth": df[self.config.msa_column],
        })

    def process_msa_to_shami(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Processing MSA → Shami translation")

        msa_texts = df[self.config.msa_column].tolist()
        batch_size = self.config.batch_size

        pred_texts: List[str] = []
        pred_raws: List[str] = []

        for i in tqdm(range(0, len(msa_texts), batch_size), desc="MSA→Shami"):
            batch = msa_texts[i:i + batch_size]
            batch_res = self.translate_batch(batch, self.config.msa_to_shami_template)

            pred_texts.extend([r["translation"] for r in batch_res])
            pred_raws.extend([r["raw"] for r in batch_res])

        return pd.DataFrame({
            self.config.id_column: df[self.config.id_column],
            self.config.msa_column: df[self.config.msa_column],
            "Shami-prediction": pred_texts,
            "Shami-prediction-raw": pred_raws,
            "Shami-Ground-Truth": df[self.config.shami_column],
        })

    def save_results(self, shami_to_msa_df: pd.DataFrame, msa_to_shami_df: pd.DataFrame):
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        p1 = out_dir / "shami-to-msa.csv"
        p2 = out_dir / "msa-to-shami.csv"

        self.logger.info(f"Saving Shami→MSA results to {p1}")
        shami_to_msa_df.to_csv(p1, index=False, encoding="utf-8")

        self.logger.info(f"Saving MSA→Shami results to {p2}")
        msa_to_shami_df.to_csv(p2, index=False, encoding="utf-8")

        self.logger.info("Results saved successfully")

    def run_translation_pipeline(self):
        self.load_model()
        input_df = self.load_input_data()

        shami_to_msa_df = self.process_shami_to_msa(input_df)
        msa_to_shami_df = self.process_msa_to_shami(input_df)

        self.save_results(shami_to_msa_df, msa_to_shami_df)
        self.logger.info("Translation pipeline completed successfully")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bidirectional Arabic Translation using vLLM with JAIS Model"
    )

    parser.add_argument(
        "--config", type=str, help="Path to YAML/JSON configuration file"
    )

    # Model settings
    parser.add_argument(
        "--model-name", type=str, default="jais-70b-local",
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=4,
        help="Tensor parallelism size (number of GPUs)"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        help="Data type: auto, float16, bfloat16"
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", default=True,
        help="Disable CUDA graphs (fixes all-reduce errors with tensor parallelism)"
    )
    parser.add_argument(
        "--no-enforce-eager", dest="enforce_eager", action="store_false",
        help="Enable CUDA graphs (may cause errors with tensor parallelism)"
    )

    # Inference parameters
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.3,
        help="Repetition penalty (1.0 = no penalty)"
    )

    # Processing settings
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--input-file", type=str, default="data/shami_v2.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Output directory for results"
    )

    # Column names
    parser.add_argument(
        "--id-column", type=str, default="sentID.BTEC",
        help="Column name for sentence IDs"
    )
    parser.add_argument(
        "--shami-column", type=str, default="Shami-sent",
        help="Column name for Shami sentences"
    )
    parser.add_argument(
        "--msa-column", type=str, default="MSA-sent",
        help="Column name for MSA sentences"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.config:
        config = TranslationConfig.from_file(args.config)
        # Override with command line args if provided
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                attr_name = key.replace("-", "_")
                if hasattr(config, attr_name):
                    # Only override if explicitly set (not default)
                    pass  # Config file takes precedence
    else:
        # Build config from command line args
        config_dict = {}
        for k, v in vars(args).items():
            if v is not None and k != "config":
                config_dict[k.replace("-", "_")] = v
        config = TranslationConfig(**config_dict)

    translator = ArabicTranslator(config)
    translator.run_translation_pipeline()


if __name__ == "__main__":
    main()