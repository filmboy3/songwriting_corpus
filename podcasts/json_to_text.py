#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path

def convert_json_to_text(json_file, output_dir=None):
    """
    Convert a Whisper JSON transcript to plain text format.
    
    Args:
        json_file: Path to the JSON transcript file
        output_dir: Directory to save the text file (defaults to same directory as JSON)
    
    Returns:
        Path to the saved text file
    """
    json_path = Path(json_file)
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    output_path = output_dir / f"{json_path.stem}.txt"
    
    # Read and parse JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {json_file} is not a valid JSON file")
            return None
    
    # Extract text content
    if 'text' in data:
        text_content = data['text']
    elif 'segments' in data:
        segments = [seg.get('text', '') for seg in data.get('segments', [])]
        text_content = ' '.join(segments)
    else:
        print(f"Error: Unknown JSON format in {json_file}")
        return None
    
    # Write text to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    print(f"Converted {json_path.name} to {output_path.name}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert Whisper JSON transcripts to text")
    parser.add_argument("json_file", help="Path to the JSON transcript file")
    parser.add_argument("--output_dir", help="Directory to save the text file")
    args = parser.parse_args()
    
    convert_json_to_text(args.json_file, args.output_dir)

if __name__ == "__main__":
    main()
