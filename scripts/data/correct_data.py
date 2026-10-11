import pandas as pd
import json
import os
from openai import OpenAI
import csv
# Initialize the OpenAI client
# Reads OPENAI_API_KEY from the environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def verify_msa(shami_sent, msa_sent):
    """Sends the Shami and MSA sentences to OpenAI and returns the evaluation."""
    
    # --- UPDATED PROMPT ---
    system_prompt = """
    You are an expert Arabic linguist specializing in Levantine (Shami) Arabic and Modern Standard Arabic (MSA). 
    The user will provide a Shami sentence and a proposed MSA equivalent.
    Check if the Shami sentence and the MSA sentence have the same overall meaning.
    
    IMPORTANT RULES:
    1. Focus on the OVERALL INTENT and CONTEXT, not literal word-for-word translation.
    2. Be lenient. If the MSA sentence conveys the core message accurately, output "yes". Do not reject it for minor phrasing variations (like adding "near" to "here").
    3. Be aware of Levantine conversational idioms. For example, in directions, "ساوي" or "سوي" means "straight ahead" (استمر في السير), not "do something". If the MSA captures the correct idiomatic meaning, accept it as "yes".
    
    You must respond ONLY with a JSON object containing exactly two keys:
    1. "is_msa": "yes" if the meaning matches, or "no" if the meaning is fundamentally incorrect or completely unrelated.
    2. "new_msa_sent": If "yes", leave this entirely empty (an empty string ""). If "no", provide the corrected MSA sentence.
    """
    
    user_prompt = f"Shami-sent: {shami_sent}\nMSA-sent: {msa_sent}"

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14", # Or gpt-3.5-turbo
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("is_msa", "error"), result.get("new_msa_sent", "error")
        
    except Exception as e:
        print(f"API Error for sentence '{shami_sent}': {e}")
        return "error", "error"

def process_csv_live_save(input_filepath, output_filepath):
    print(f"Reading data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    
    if not {'Shami-sent', 'MSA-sent'}.issubset(df.columns):
        print("Error: The CSV must contain 'Shami-sent' and 'MSA-sent' columns.")
        return

    total_rows = len(df)
    
    # Prepare the header for the new CSV
    header = list(df.columns) + ['is-MSA', 'new-MSA-sent']
    
    # Create the output file and write the header FIRST
    with open(output_filepath, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Iterate row by row
    for index, row in df.iterrows():
        shami = row['Shami-sent']
        msa = row['MSA-sent']
        
        print(f"Processing and saving row {index + 1}/{total_rows}...")
        
        # Call the API
        is_msa, new_msa = verify_msa(shami, msa)
        
        # Convert the current row to a list and add the new API results
        new_row = row.tolist() + [is_msa, new_msa]
        
        # Open the file in APPEND mode ('a') and save the single row immediately
        with open(output_filepath, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

    print(f"\nFinished! All results are safely saved to {output_filepath}")

# --- Run the script ---
if __name__ == "__main__":
    INPUT_FILE = "data/shami_v1.csv"
    OUTPUT_FILE = "outputs/shami_correctness.csv"  # Input for update_shami_csv.py
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    process_csv_live_save(INPUT_FILE, OUTPUT_FILE)