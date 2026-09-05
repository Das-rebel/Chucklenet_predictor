#!/usr/bin/env python3
"""
Convert word-level laughter data to humor strength labels for Chucklenet Predictor training.

This script converts the word-level laughter data (0/1 labels) into humor strength scores
(0-100) that can be used to train the Chucklenet Predictor model.

Usage:
    python convert_laughter_data.py > training_data.jsonl
"""

import json
import sys
from pathlib import Path

def convert_file(input_path: str, output_path: str):
    """Convert a JSONL file of word-level laughter data to humor strength scores."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                words = data['words']
                labels = data['labels']
                
                # Calculate humor strength as proportion of laughter tokens
                laughter_count = sum(labels)
                total_words = len(labels)
                
                if total_words > 0:
                    humor_strength = laughter_count / total_words  # 0.0 to 1.0
                    # Convert to 0-100 scale
                    humor_strength_100 = humor_strength * 100
                    
                    # Create a new entry with humor_strength
                    new_entry = {
                        "text": " ".join(words),
                        "humor_strength": humor_strength_100
                    }
                    f_out.write(json.dumps(new_entry) + '\n')
                    
            except Exception as e:
                print(f"Error processing line {line}: {e}", file=sys.stderr)
                continue

def main():
    """Main function - reads from stdin and writes to stdout."""
    input_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/Subho/autonomous_laughter_prediction_essential/data/train_small.jsonl"
    output_file = "/Users/Subho/funny-strength-predictor/training_data.jsonl"
    
    print(f"Converting {input_file} to {output_file}...")
    convert_file(input_file, output_file)
    print(f"Conversion complete. Training data written to {output_file}")

if __name__ == "__main__":
    main()