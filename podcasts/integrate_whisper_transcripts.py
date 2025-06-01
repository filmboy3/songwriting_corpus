#!/usr/bin/env python3
"""
Integrate Whisper Transcripts

This script processes Whisper JSON transcripts and integrates them into the songwriting corpus.
It handles:
1. Converting JSON transcripts to clean text
2. Normalizing text (removing filler words, timestamps, etc.)
3. Adding appropriate formatting tokens
4. Saving processed transcripts to the corpus directory

Usage:
  python integrate_whisper_transcripts.py --transcript-dir /path/to/transcripts --output-dir /path/to/corpus
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transcript_integration")

def clean_transcript_text(text: str) -> str:
    """Clean and normalize transcript text."""
    # Replace common filler words and hesitations
    fillers = ["um", "uh", "like", "you know", "I mean", "sort of", "kind of"]
    for filler in fillers:
        text = text.replace(f" {filler} ", " ")
    
    # Remove repeated words (simple case)
    words = text.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i > 0 and word.lower() == words[i-1].lower():
            continue
        cleaned_words.append(word)
    
    # Rejoin and normalize spacing
    text = " ".join(cleaned_words)
    text = " ".join(text.split())
    
    return text

def process_whisper_transcript(json_path: str) -> str:
    """Process a Whisper JSON transcript and return cleaned text."""
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
        
        # Clean the text
        cleaned_text = clean_transcript_text(full_text)
        return cleaned_text
    
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return ""

def format_for_corpus(text: str, filename: str) -> str:
    """Format transcript text for the corpus with appropriate tokens."""
    # Extract podcast name from filename
    podcast_name = os.path.splitext(os.path.basename(filename))[0]
    podcast_name = podcast_name.replace("_", " ")
    
    # Format with tokens
    formatted_text = f"<PODCAST>\n"
    formatted_text += f"<TITLE>{podcast_name}</TITLE>\n\n"
    formatted_text += f"<TRANSCRIPT>\n{text}\n</TRANSCRIPT>\n"
    formatted_text += f"</PODCAST>"
    
    return formatted_text

def integrate_transcripts(transcript_dir: str, output_dir: str) -> None:
    """Integrate Whisper transcripts into the corpus."""
    transcript_dir = os.path.abspath(transcript_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all JSON files in the transcript directory
    json_files = [f for f in os.listdir(transcript_dir) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} JSON transcripts to process")
    
    processed_count = 0
    for json_file in json_files:
        json_path = os.path.join(transcript_dir, json_file)
        
        # Process the transcript
        cleaned_text = process_whisper_transcript(json_path)
        if not cleaned_text:
            logger.warning(f"Skipping empty transcript: {json_file}")
            continue
        
        # Format for corpus
        formatted_text = format_for_corpus(cleaned_text, json_file)
        
        # Save to output directory
        output_filename = os.path.splitext(json_file)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            f.write(formatted_text)
        
        processed_count += 1
        logger.info(f"Processed {json_file} -> {output_filename}")
    
    logger.info(f"Integration complete. Processed {processed_count} transcripts.")

def main():
    """Main function to integrate Whisper transcripts."""
    parser = argparse.ArgumentParser(description="Integrate Whisper transcripts into the corpus")
    parser.add_argument("--transcript-dir", type=str, required=True, 
                        help="Directory containing Whisper JSON transcripts")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed transcripts")
    args = parser.parse_args()
    
    # Integrate transcripts
    integrate_transcripts(args.transcript_dir, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
