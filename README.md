# Shami ↔ MSA Translation Benchmark

Bidirectional translation between Syrian Arabic (Shami) and Modern Standard Arabic (MSA), comparing a dedicated MT model, open LLMs, and GPT-4.1 on a 1,000-sentence parallel test set.

## Results

Averages over 1,000 sentences per direction. The AI-judge score (1–5) comes from GPT-4.1 (`gpt-4.1-2025-04-14`) rating each prediction against the reference.

Shami-MT is a pair of T5 models, [Shami-MT](https://huggingface.co/Omartificial-Intelligence-Space/Shami-MT) (MSA → Shami) and [SHAMI-MT-2MSA](https://huggingface.co/Omartificial-Intelligence-Space/SHAMI-MT-2MSA) (Shami → MSA), trained with the [AraT5](https://github.com/UBC-NLP/araT5) training code.

| Model | Direction | BLEU-1 | BLEU-4 | ROUGE-L | AI judge |
|---|---|---:|---:|---:|---:|
| Shami-MT | MSA → Shami | **0.401** | **0.107** | **0.420** | 4.02 |
| GPT-4.1 | MSA → Shami | 0.359 | 0.092 | 0.378 | **4.03** |
| Jais | MSA → Shami | 0.268 | 0.063 | 0.290 | 3.77 |
| Mixtral 8x22B | MSA → Shami | 0.143 | 0.028 | 0.155 | 2.56 |
| Fanar-1-9B | MSA → Shami | 0.131 | 0.023 | 0.142 | 2.99 |
| Jais | Shami → MSA | **0.329** | 0.097 | 0.363 | 3.95 |
| Shami-MT | Shami → MSA | 0.321 | 0.097 | **0.369** | 3.93 |
| GPT-4.1 | Shami → MSA | 0.272 | 0.080 | 0.320 | **4.13** |
| Mixtral 8x22B | Shami → MSA | 0.155 | 0.034 | 0.201 | 2.52 |
| Fanar-1-9B | Shami → MSA | 0.121 | 0.022 | 0.140 | 2.66 |

Per-sentence predictions are in [results/predictions/](results/predictions/) and per-sentence scores (BLEU-1/4, ROUGE-1/2/L, AI judge) in [results/evaluation/](results/evaluation/).

## Data

[data/shami_v2.csv](data/shami_v2.csv) is the test set used for all experiments: 1,000 sentences (500 Aleppo, 500 Damascus) with their MSA equivalents, drawn from the [MADAR corpus](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/). Columns: `sentID.BTEC`, `lang`, `Shami-sent`, `MSA-sent`.

[data/shami_v1.csv](data/shami_v1.csv) is the same set before correction. In v2, GPT-4.1 checked each MSA reference against its Shami sentence and rewrote the ones whose meaning didn't match.

The raw MADAR corpus files are not included in this repository; request them from CAMeL Lab.

## Project layout

```
configs/              YAML configs (model, decoding params, prompts, column names)
data/                 Shami–MSA test set
scripts/
  data/               Reference correction: correct_data.py → update_shami_csv.py
  translate/          One script per model / provider
  postprocess/        clean_data.py (strip English, diacritics) and check_outputs.py
  evaluate.py         BLEU, ROUGE and GPT-4.1 judge scoring
results/
  predictions/        Final model outputs, <model>-<direction>.csv
  evaluation/         Scored outputs, <model>-<direction>-eval.csv
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # add your OpenAI key
set -a && source .env && set +a   # export it to the shell
```

`OPENAI_API_KEY` is used by `openai_api.py`, `evaluate.py` and `correct_data.py`.

## Usage

Run everything from the repository root. Translation scripts write `shami-to-msa.csv` and `msa-to-shami.csv` (or similar) to `outputs/`, which is git-ignored. Copy the runs you want to keep into `results/predictions/`.

```bash
# Translate
python scripts/translate/shami_mt.py                                                      # Shami-MT (T5)
python scripts/translate/fanar_jais_vllm.py --config configs/fanar_jais.yaml              # Fanar / Jais via vLLM
python scripts/translate/mixtral_ollama.py --config configs/mixtral_ollama.yaml           # Mixtral via Ollama
python scripts/translate/openai_api.py --config configs/openai_api.yaml                   # GPT-4.1
python scripts/translate/vllm_baseline.py --config configs/vllm.yaml                      # any Hugging Face model via vLLM

# Clean and inspect outputs
python scripts/postprocess/clean_data.py outputs/ outputs/cleaned/
python scripts/postprocess/check_outputs.py outputs/cleaned/

# Score (edit the file paths at the bottom of evaluate.py first)
python scripts/evaluate.py
```

## Citation

```bibtex
@misc{sibaee2025shamimtsyrianarabicdialect,
      title={SHAMI-MT: A Syrian Arabic Dialect to Modern Standard Arabic Bidirectional Machine Translation System}, 
      author={Serry Sibaee and Omer Nacar and Yasser Al-Habashi and Adel Ammar and Wadii Boulila},
      year={2025},
      eprint={2508.02268},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.02268}, 
}
```
