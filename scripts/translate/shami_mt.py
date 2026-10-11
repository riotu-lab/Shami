import pandas as pd
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch
import gc
import os

# Configuration
INPUT_CSV = "data/shami_v2.csv"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 1

# Read CSV
print("Reading CSV...")
df = pd.read_csv(INPUT_CSV, encoding='utf-8')

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")


def translate_batch(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== PART 1: MSA to Shami =====
print("\n[1/2] Loading MSA-to-Shami model...")
tokenizer_msa_to_shami = T5Tokenizer.from_pretrained("Omartificial-Intelligence-Space/Shami-MT")
model_msa_to_shami = AutoModelForSeq2SeqLM.from_pretrained("Omartificial-Intelligence-Space/Shami-MT")
model_msa_to_shami.to(device).eval()

print("Translating MSA to Shami dialect...")
msa_to_shami_predictions = []
for i in range(0, len(df), BATCH_SIZE):
    batch = df['MSA-sent'].iloc[i:i+BATCH_SIZE].tolist()
    msa_to_shami_predictions.extend(translate_batch(batch, tokenizer_msa_to_shami, model_msa_to_shami, device))
    if (i + BATCH_SIZE) % 100 == 0:
        print(f"  Processed {min(i+BATCH_SIZE, len(df))}/{len(df)} samples")

del model_msa_to_shami, tokenizer_msa_to_shami
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

# ===== PART 2: Shami to MSA =====
print("\n[2/2] Loading Shami-to-MSA model...")
tokenizer_shami_to_msa = T5Tokenizer.from_pretrained("Omartificial-Intelligence-Space/SHAMI-MT-2MSA")
model_shami_to_msa = AutoModelForSeq2SeqLM.from_pretrained("Omartificial-Intelligence-Space/SHAMI-MT-2MSA")
model_shami_to_msa.to(device).eval()

print("Translating Shami dialect to MSA...")
shami_to_msa_predictions = []
for i in range(0, len(df), BATCH_SIZE):
    batch = df['Shami-sent'].iloc[i:i+BATCH_SIZE].tolist()
    shami_to_msa_predictions.extend(translate_batch(batch, tokenizer_shami_to_msa, model_shami_to_msa, device))
    if (i + BATCH_SIZE) % 100 == 0:
        print(f"  Processed {min(i+BATCH_SIZE, len(df))}/{len(df)} samples")

del model_shami_to_msa, tokenizer_shami_to_msa
gc.collect()

# Create output DataFrames
df_msa_to_shami = pd.DataFrame({
    'sentID.BTEC': df['sentID.BTEC'],
    'lang': df['lang'],
    'MSA-sent': df['MSA-sent'],
    'shami-prediction': msa_to_shami_predictions,
    'GT-Shami': df['Shami-sent']
})

df_shami_to_msa = pd.DataFrame({
    'sentID.BTEC': df['sentID.BTEC'],
    'lang': df['lang'],
    'Shami-sent': df['Shami-sent'],
    'MSA-prediction': shami_to_msa_predictions,
    'GT-MSA': df['MSA-sent']
})

# Save to CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_msa_to_shami.to_csv(os.path.join(OUTPUT_DIR, 'shami-model-msa-to-shami.csv'), index=False, encoding='utf-8')
df_shami_to_msa.to_csv(os.path.join(OUTPUT_DIR, 'shami-model-shami-to-msa.csv'), index=False, encoding='utf-8')

print("\n✓ Done!")