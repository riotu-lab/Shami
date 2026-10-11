#!/usr/bin/env python3
"""
CSV Arabic Text Checker
Checks specific columns (Shami-prediction or MSA-prediction) and reports:
- How many rows contain English letters
- How many rows contain diacritics (tashkeel)
- How many rows contain "MSA:" prefix
- How many rows contain "Shami:" prefix
"""

import csv
import re
import sys
import os
from glob import glob


def has_english(text):
    """Check if text contains English letters."""
    if not text:
        return False
    return bool(re.search(r'[a-zA-Z]', text))


def has_diacritics(text):
    """Check if text contains Arabic diacritical marks (tashkeel)."""
    if not text:
        return False
    return bool(re.search(r'[\u064B-\u065F\u0670]', text))


def has_msa_prefix(text):
    """Check if text contains 'MSA:' prefix."""
    if not text:
        return False
    return 'MSA:' in text


def has_shami_prefix(text):
    """Check if text contains 'Shami:' prefix."""
    if not text:
        return False
    return 'Shami:' in text


def check_csv(input_file, verbose=True):
    """
    Check a CSV file for English, diacritics, and prefixes in prediction columns.
    
    Args:
        input_file: Path to input CSV file
        verbose: Print progress messages
    
    Returns:
        Tuple of (rows_with_english, rows_with_diacritics, rows_with_msa, rows_with_shami)
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return 0, 0, 0, 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
    
    rows_with_english = 0
    rows_with_diacritics = 0
    rows_with_msa = 0
    rows_with_shami = 0
    col_index = None
    col_name = None
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, dialect)
        
        # Find the target column in header
        header = next(reader)
        for i, col in enumerate(header):
            if col in ['Shami-prediction', 'MSA-prediction']:
                col_index = i
                col_name = col
                break
        
        if col_index is None:
            print(f"  {os.path.basename(input_file)}: No prediction column found")
            return 0, 0, 0, 0
        
        # Check each row
        for row in reader:
            if col_index < len(row):
                cell = row[col_index]
                if has_english(cell):
                    rows_with_english += 1
                if has_diacritics(cell):
                    rows_with_diacritics += 1
                if has_msa_prefix(cell):
                    rows_with_msa += 1
                if has_shami_prefix(cell):
                    rows_with_shami += 1
    
    if verbose:
        print(f"  {os.path.basename(input_file)} [{col_name}]:")
        print(f"      English: {rows_with_english} rows")
        print(f"      Diacritics: {rows_with_diacritics} rows")
        print(f"      MSA: {rows_with_msa} rows")
        print(f"      Shami: {rows_with_shami} rows")
    
    return rows_with_english, rows_with_diacritics, rows_with_msa, rows_with_shami


def check_directory(input_dir):
    """
    Check all CSV files in a directory.
    
    Args:
        input_dir: Path to directory containing CSV files
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        sys.exit(1)
    
    csv_files = glob(os.path.join(input_dir, '*.csv'))
    csv_files += glob(os.path.join(input_dir, '*.CSV'))
    csv_files = list(set(csv_files))
    
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'")
        sys.exit(1)
    
    print(f"Checking {len(csv_files)} CSV files...\n")
    
    total_english = 0
    total_diacritics = 0
    total_msa = 0
    total_shami = 0
    
    for csv_file in sorted(csv_files):
        english, diacritics, msa, shami = check_csv(csv_file, verbose=True)
        total_english += english
        total_diacritics += diacritics
        total_msa += msa
        total_shami += shami
        print()
    
    print(f"{'='*50}")
    print(f"Total rows with English: {total_english}")
    print(f"Total rows with diacritics: {total_diacritics}")
    print(f"Total rows with MSA: {total_msa}")
    print(f"Total rows with Shami: {total_shami}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("CSV Arabic Text Checker")
        print("="*50)
        print("\nChecks 'Shami-prediction' or 'MSA-prediction' columns")
        print("\nUsage:")
        print("  Single file:  python scripts/postprocess/check_outputs.py <input.csv>")
        print("  Directory:    python scripts/postprocess/check_outputs.py <input_dir>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if os.path.isdir(input_path):
        check_directory(input_path)
    else:
        check_csv(input_path)


if __name__ == "__main__":
    main()