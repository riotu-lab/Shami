import pandas as pd
from openai import OpenAI
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Initialize OpenAI client
client = OpenAI()  # reads OPENAI_API_KEY from the environment

def calculate_bleu(prediction, ground_truth):
    """Calculate BLEU-1 and BLEU-4 scores for Arabic text"""
    if not prediction or not ground_truth:
        return 0.0, 0.0
    
    reference = [ground_truth.split()]
    hypothesis = prediction.split()
    
    if not hypothesis or not reference[0]:
        return 0.0, 0.0
    
    smoothing = SmoothingFunction().method1
    
    # BLEU-1 (unigram precision)
    bleu1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), 
                          smoothing_function=smoothing)
    
    # BLEU-4 (4-gram precision)
    bleu4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoothing) if len(hypothesis) >= 4 else 0.0
    
    return round(bleu1, 4), round(bleu4, 4)

def calculate_rouge(prediction, ground_truth):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores with custom tokenization for Arabic"""
    if not prediction or not ground_truth:
        return 0.0, 0.0, 0.0
    
    # Tokenize by whitespace (works for Arabic)
    pred_tokens = prediction.strip().split()
    ref_tokens = ground_truth.strip().split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    
    # ROUGE-1: Unigram overlap
    pred_unigrams = set(pred_tokens)
    ref_unigrams = set(ref_tokens)
    overlap_unigrams = pred_unigrams.intersection(ref_unigrams)
    
    if len(ref_unigrams) == 0:
        rouge1 = 0.0
    else:
        precision = len(overlap_unigrams) / len(pred_unigrams) if len(pred_unigrams) > 0 else 0
        recall = len(overlap_unigrams) / len(ref_unigrams)
        rouge1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROUGE-2: Bigram overlap
    pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
    ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
    overlap_bigrams = pred_bigrams.intersection(ref_bigrams)
    
    if len(ref_bigrams) == 0:
        rouge2 = 0.0
    else:
        precision = len(overlap_bigrams) / len(pred_bigrams) if len(pred_bigrams) > 0 else 0
        recall = len(overlap_bigrams) / len(ref_bigrams)
        rouge2 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROUGE-L: Longest Common Subsequence
    def lcs_length(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]
    
    lcs_len = lcs_length(pred_tokens, ref_tokens)
    
    if len(ref_tokens) == 0:
        rougeL = 0.0
    else:
        precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = lcs_len / len(ref_tokens)
        rougeL = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return round(rouge1, 4), round(rouge2, 4), round(rougeL, 4)

def evaluate_with_openai(source, prediction, ground_truth, source_lang, target_lang):
    """Evaluate translation quality using OpenAI API (1-5 score)"""
    prompt = f"""Evaluate the translation quality on a scale of 1-5:

Source ({source_lang}): {source}
Prediction ({target_lang}): {prediction}
Ground Truth ({target_lang}): {ground_truth}

Rate how accurate the prediction is compared to the ground truth:
5 = Perfect or nearly perfect translation
4 = Good translation with minor differences
3 = Acceptable translation with some errors
2 = Poor translation with significant errors
1 = Very poor or incorrect translation

Respond with ONLY a number (1-5)."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        score = int(response.choices[0].message.content.strip())
        return score if 1 <= score <= 5 else 3
    except Exception as e:
        print(f"\nOpenAI API Error: {e}")
        return None

def evaluate_shami_to_msa(input_file="shami_to_msa_predictions.csv", 
                          output_file="shami_to_msa_evaluated.csv",
                          use_openai=True):
    """Evaluate Shami to MSA translations with BLEU, ROUGE, and optionally OpenAI"""
    print(f"📊 Processing {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize score columns
    df['BLEU-1'] = 0.0
    df['BLEU-4'] = 0.0
    df['ROUGE-1'] = 0.0
    df['ROUGE-2'] = 0.0
    df['ROUGE-L'] = 0.0
    if use_openai:
        df['AI-judge-score'] = None
    
    for idx, row in df.iterrows():
        print(f"Evaluating {idx + 1}/{len(df)}...", end="\r")
        
        # Get prediction and ground truth as strings
        pred = str(row['MSA-prediction']).strip()
        gt = str(row['MSA-Ground-Truth']).strip()
        
        # Calculate BLEU and ROUGE (fast, no API)
        bleu1, bleu4 = calculate_bleu(pred, gt)
        rouge1, rouge2, rougeL = calculate_rouge(pred, gt)
        
        df.at[idx, 'BLEU-1'] = bleu1
        df.at[idx, 'BLEU-4'] = bleu4
        df.at[idx, 'ROUGE-1'] = rouge1
        df.at[idx, 'ROUGE-2'] = rouge2
        df.at[idx, 'ROUGE-L'] = rougeL
        
        # Optional: OpenAI evaluation (slower, costs money)
        if use_openai:
            ai_score = evaluate_with_openai(
                source=str(row['Shami-sent']),
                prediction=pred,
                ground_truth=gt,
                source_lang="Shami Dialect",
                target_lang="MSA"
            )
            df.at[idx, 'AI-judge-score'] = ai_score
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")
    print(f"📈 Average Scores:")
    print(f"   BLEU-1: {df['BLEU-1'].mean():.4f}")
    print(f"   BLEU-4: {df['BLEU-4'].mean():.4f}")
    print(f"   ROUGE-1: {df['ROUGE-1'].mean():.4f}")
    print(f"   ROUGE-2: {df['ROUGE-2'].mean():.4f}")
    print(f"   ROUGE-L: {df['ROUGE-L'].mean():.4f}")
    if use_openai:
        print(f"   AI Judge: {df['AI-judge-score'].mean():.2f}/5")
    return df

def evaluate_msa_to_shami(input_file="msa_to_shami_predictions.csv", 
                          output_file="msa_to_shami_evaluated.csv",
                          use_openai=True):
    """Evaluate MSA to Shami translations with BLEU, ROUGE, and optionally OpenAI"""
    print(f"📊 Processing {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize score columns
    df['BLEU-1'] = 0.0
    df['BLEU-4'] = 0.0
    df['ROUGE-1'] = 0.0
    df['ROUGE-2'] = 0.0
    df['ROUGE-L'] = 0.0
    if use_openai:
        df['AI-judge-score'] = None
    
    for idx, row in df.iterrows():
        print(f"Evaluating {idx + 1}/{len(df)}...", end="\r")
        
        # Get prediction and ground truth as strings
        pred = str(row['Shami-prediction']).strip()
        gt = str(row['Shami-Ground-Truth']).strip()
        
        # Calculate BLEU and ROUGE (fast, no API)
        bleu1, bleu4 = calculate_bleu(pred, gt)
        rouge1, rouge2, rougeL = calculate_rouge(pred, gt)
        
        df.at[idx, 'BLEU-1'] = bleu1
        df.at[idx, 'BLEU-4'] = bleu4
        df.at[idx, 'ROUGE-1'] = rouge1
        df.at[idx, 'ROUGE-2'] = rouge2
        df.at[idx, 'ROUGE-L'] = rougeL
        
        # Optional: OpenAI evaluation (slower, costs money)
        if use_openai:
            ai_score = evaluate_with_openai(
                source=str(row['MSA-sent']),
                prediction=pred,
                ground_truth=gt,
                source_lang="MSA",
                target_lang="Shami Dialect"
            )
            df.at[idx, 'AI-judge-score'] = ai_score
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")
    print(f"📈 Average Scores:")
    print(f"   BLEU-1: {df['BLEU-1'].mean():.4f}")
    print(f"   BLEU-4: {df['BLEU-4'].mean():.4f}")
    print(f"   ROUGE-1: {df['ROUGE-1'].mean():.4f}")
    print(f"   ROUGE-2: {df['ROUGE-2'].mean():.4f}")
    print(f"   ROUGE-L: {df['ROUGE-L'].mean():.4f}")
    if use_openai:
        print(f"   AI Judge: {df['AI-judge-score'].mean():.2f}/5")
    return df

if __name__ == "__main__":
    # Option 1: Full evaluation (BLEU + ROUGE + OpenAI)
    
    evaluate_shami_to_msa(input_file="results/predictions/mixtral2-shami-to-msa.csv", 
                          output_file="results/evaluation/mixtral2-shami-to-msa-eval.csv",use_openai=True)
    evaluate_msa_to_shami(input_file="results/predictions/mixtral2-msa-to-shami.csv", 
                          output_file="results/evaluation/mixtral2-msa-to-shami-eval.csv" ,use_openai=True)
    
    # Option 2: Fast evaluation (BLEU + ROUGE only, no API needed)
    # evaluate_shami_to_msa(use_openai=False)
    # evaluate_msa_to_shami(use_openai=False)