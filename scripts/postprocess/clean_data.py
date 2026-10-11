#!/usr/bin/env python3
"""
CSV Arabic Text Cleaner
Removes English letters, diacritics, and non-Arabic characters from CSV cells.
Keeps only Arabic letters and words.
Supports processing single files or entire directories.
"""

import csv
import re
import sys
import os
from glob import glob


def remove_diacritics(text):
    """Remove Arabic diacritical marks (tashkeel)."""
    # Arabic diacritics (tashkeel) Unicode range: \u064B-\u0652
    # Also includes other common diacritical marks
    diacritics_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return diacritics_pattern.sub('', text)


def keep_arabic_only(text):
    """Keep only Arabic letters and basic punctuation/spaces."""
    if not text:
        return text
    
    # Remove dialect labels (MSA: and Shami:)
    text = re.sub(r'\bMSA:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bShami:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove diacritics first
    text = remove_diacritics(text)
    
    # Arabic letters range: \u0621-\u063A (أ-غ) and \u0641-\u064A (ف-ي)
    # Also include Arabic extended characters and common Arabic punctuation
    # Keep: Arabic letters, spaces, Arabic-Indic numerals, basic punctuation
    
    # Pattern to match Arabic letters only
    arabic_pattern = re.compile(r'[^\u0621-\u063A\u0641-\u064A\u0660-\u0669\s\.\,\؟\!\:\،\؛]')
    
    # Remove everything that's not Arabic letters, spaces, or basic punctuation
    cleaned = arabic_pattern.sub('', text)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def clean_csv(input_file, output_file=None, verbose=True):
    """
    Clean a CSV file by removing English and diacritics from all cells.
    Skips the first row (header).
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: input_cleaned.csv)
        verbose: Print progress messages
    
    Returns:
        Tuple of (output_file, rows_processed, cells_cleaned)
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return None, 0, 0
    
    # Generate output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    # Detect the dialect/delimiter
    with open(input_file, 'r', encoding='utf-8') as f:
        sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # Default to standard CSV
    
    # Process the CSV
    rows_processed = 0
    cells_cleaned = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, dialect)
        rows = []
        
        for row_num, row in enumerate(reader):
            # Skip cleaning the header row (first row)
            if row_num == 0:
                rows.append(row)
                rows_processed += 1
                continue
            
            cleaned_row = []
            for cell in row:
                original = cell
                cleaned = keep_arabic_only(cell)
                cleaned_row.append(cleaned)
                if original != cleaned:
                    cells_cleaned += 1
            rows.append(cleaned_row)
            rows_processed += 1
    
    # Write the cleaned CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, dialect)
        writer.writerows(rows)
    
    if verbose:
        print(f"  ✓ {os.path.basename(input_file)}: {rows_processed} rows, {cells_cleaned} cells cleaned (header skipped)")
    
    return output_file, rows_processed, cells_cleaned


def clean_directory(input_dir, output_dir=None):
    """
    Clean all CSV files in a directory.
    
    Args:
        input_dir: Path to directory containing CSV files
        output_dir: Path to output directory (default: input_dir/cleaned)
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        sys.exit(1)
    
    # Find all CSV files
    csv_files = glob(os.path.join(input_dir, '*.csv'))
    csv_files += glob(os.path.join(input_dir, '*.CSV'))
    csv_files = list(set(csv_files))  # Remove duplicates
    
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'")
        sys.exit(1)
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'cleaned')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(csv_files)} CSV files...")
    print(f"Output directory: {output_dir}\n")
    
    total_rows = 0
    total_cells = 0
    files_processed = 0
    
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        output_file = os.path.join(output_dir, filename)
        
        result, rows, cells = clean_csv(csv_file, output_file, verbose=True)
        if result:
            total_rows += rows
            total_cells += cells
            files_processed += 1
    
    print(f"\n{'='*50}")
    print(f"✓ Completed!")
    print(f"  Files processed: {files_processed}")
    print(f"  Total rows: {total_rows}")
    print(f"  Total cells cleaned: {total_cells}")
    print(f"  Output saved to: {output_dir}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("CSV Arabic Text Cleaner")
        print("="*50)
        print("\nUsage:")
        print("  Single file:  python scripts/postprocess/clean_data.py <input.csv> [output.csv]")
        print("  Directory:    python scripts/postprocess/clean_data.py <input_dir> [output_dir]")
        print("\nExamples:")
        print("  python scripts/postprocess/clean_data.py data.csv")
        print("  python scripts/postprocess/clean_data.py data.csv cleaned_data.csv")
        print("  python scripts/postprocess/clean_data.py ./csv_files/")
        print("  python scripts/postprocess/clean_data.py ./csv_files/ ./cleaned_files/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if input is a directory or file
    if os.path.isdir(input_path):
        clean_directory(input_path, output_path)
    else:
        clean_csv(input_path, output_path)


if __name__ == "__main__":
    main()