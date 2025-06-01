#!/usr/bin/env python3
"""
Batch JSON to Text Converter

This script converts all Whisper JSON transcripts to plain text files for quick review.
It processes all JSON files in the specified directory and creates corresponding .txt files.

Usage:
  python batch_json_to_text.py --input-dir /path/to/json/transcripts --output-dir /path/to/text/output
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("json_to_text_conversion.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("json_to_text")

def convert_json_to_text(json_path: str, output_path: str) -> bool:
    """Convert a Whisper JSON transcript to plain text."""
    try:
        with open(json_path, 'r') as f:
            transcript_data = json.load(f)
        
        # Extract text from segments
        full_text = ""
        if "segments" in transcript_data:
            for segment in transcript_data["segments"]:
                if "text" in segment:
                    full_text += segment["text"] + " "
        elif "text" in transcript_data:
            # Some Whisper versions might store the full text directly
            full_text = transcript_data["text"]
        
        # Write to output file
        with open(output_path, 'w') as f:
            f.write(full_text)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return False

def batch_convert(input_dir: str, output_dir: str) -> None:
    """Convert all JSON files in input_dir to text files in output_dir."""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} JSON files to convert")
    
    # Process each file with progress bar
    success_count = 0
    for json_file in tqdm(json_files, desc="Converting"):
        json_path = os.path.join(input_dir, json_file)
        output_filename = os.path.splitext(json_file)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        
        if convert_json_to_text(json_path, output_path):
            success_count += 1
    
    logger.info(f"Conversion complete. Successfully converted {success_count}/{len(json_files)} files.")

def main():
    """Main function to batch convert JSON to text."""
    parser = argparse.ArgumentParser(description="Convert Whisper JSON transcripts to plain text")
    parser.add_argument("--input-dir", type=str, required=True, 
                        help="Directory containing Whisper JSON transcripts")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for text files")
    args = parser.parse_args()
    
    # Batch convert
    batch_convert(args.input_dir, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
