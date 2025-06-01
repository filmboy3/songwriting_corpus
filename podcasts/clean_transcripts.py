#!/usr/bin/env python3
"""
Clean Transcripts

This script cleans Whisper transcripts by:
1. Removing excessive repetitions
2. Cleaning up sponsor messages and advertisements
3. Normalizing formatting
4. Removing speaker labels if inconsistent

Usage:
  python clean_transcripts.py --input-dir /path/to/text/transcripts --output-dir /path/to/cleaned/transcripts
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_cleaning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transcript_cleaning")

def remove_repetitions(text):
    """Remove excessive repetitions of phrases."""
    # Find and remove exact repetitions of 3+ words that repeat more than 3 times
    words = text.split()
    cleaned_words = []
    i = 0
    
    while i < len(words):
        # Skip if we're near the end
        if i + 6 >= len(words):
            cleaned_words.extend(words[i:])
            break
        
        # Check for repetition patterns
        found_repetition = False
        for phrase_len in range(3, 8):  # Check phrases of length 3 to 7 words
            if i + phrase_len * 2 >= len(words):
                continue
                
            phrase1 = " ".join(words[i:i+phrase_len])
            phrase2 = " ".join(words[i+phrase_len:i+phrase_len*2])
            
            # If we found a repetition
            if phrase1 == phrase2:
                # Check how many times it repeats
                repeat_count = 2
                next_pos = i + phrase_len * 2
                
                while next_pos + phrase_len <= len(words):
                    next_phrase = " ".join(words[next_pos:next_pos+phrase_len])
                    if next_phrase == phrase1:
                        repeat_count += 1
                        next_pos += phrase_len
                    else:
                        break
                
                # If it repeats more than 3 times, only keep one instance
                if repeat_count >= 3:
                    cleaned_words.extend(words[i:i+phrase_len])
                    i = next_pos
                    found_repetition = True
                    break
        
        # If no repetition found, add current word and move on
        if not found_repetition:
            cleaned_words.append(words[i])
            i += 1
    
    return " ".join(cleaned_words)

def clean_sponsor_messages(text):
    """Clean up sponsor messages and advertisements."""
    # Common sponsor patterns in podcasts
    sponsor_patterns = [
        r"(?i)this episode is sponsored by.*?(?=\n\n|\.\s)",
        r"(?i)this podcast is brought to you by.*?(?=\n\n|\.\s)",
        r"(?i)today's show is supported by.*?(?=\n\n|\.\s)",
        r"(?i)use (promo|code) [A-Z0-9]+ for \d+% off",
        r"(?i)visit [a-z0-9]+\.com\/[a-z0-9]+ to get",
        r"(?i)for \d+% off (your first|mattresses|purchases)"
    ]
    
    # Replace lengthy sponsor messages with a placeholder
    for pattern in sponsor_patterns:
        text = re.sub(pattern, "[SPONSOR MESSAGE]", text)
    
    return text

def normalize_formatting(text):
    """Normalize text formatting."""
    # Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Fix multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix spacing after punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    # Fix common transcription errors
    text = re.sub(r'(?i)\b(i{2,})\b', 'I', text)  # Multiple 'i's to 'I'
    text = re.sub(r'(?i)\b(gonna|gotta)\b', lambda m: 'going to' if m.group(0).lower() == 'gonna' else 'got to', text)
    
    return text

def remove_speaker_labels(text):
    """Remove inconsistent speaker labels."""
    # Common speaker label patterns
    speaker_patterns = [
        r'\[?[A-Z][a-z]+\]?: ',  # [Name]: or Name:
        r'\[[A-Za-z ]+\] ',      # [Speaker 1]
        r'Speaker \d+: '         # Speaker 1:
    ]
    
    # Check if speaker labels are used consistently
    label_count = 0
    for pattern in speaker_patterns:
        label_count += len(re.findall(pattern, text))
    
    # If labels are inconsistent (less than 10% of paragraphs have labels),
    # remove them all for consistency
    paragraphs = text.split('\n\n')
    if label_count > 0 and label_count < len(paragraphs) * 0.1:
        for pattern in speaker_patterns:
            text = re.sub(pattern, '', text)
    
    return text

def clean_transcript(input_path, output_path):
    """Clean a single transcript file."""
    try:
        with open(input_path, 'r') as f:
            text = f.read()
        
        # Apply cleaning steps
        text = remove_repetitions(text)
        text = clean_sponsor_messages(text)
        text = normalize_formatting(text)
        text = remove_speaker_labels(text)
        
        # Write cleaned transcript
        with open(output_path, 'w') as f:
            f.write(text)
        
        return True
    
    except Exception as e:
        logger.error(f"Error cleaning {input_path}: {e}")
        return False

def clean_transcripts(input_dir, output_dir):
    """Clean all transcript files in directory."""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all text files
    text_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    logger.info(f"Found {len(text_files)} transcript files to clean")
    
    # Clean each file
    success_count = 0
    for text_file in tqdm(text_files, desc="Cleaning"):
        input_path = os.path.join(input_dir, text_file)
        output_path = os.path.join(output_dir, text_file)
        
        if clean_transcript(input_path, output_path):
            success_count += 1
    
    logger.info(f"Cleaning complete. Successfully cleaned {success_count}/{len(text_files)} files.")

def main():
    """Main function to clean transcripts."""
    parser = argparse.ArgumentParser(description="Clean Whisper transcripts")
    parser.add_argument("--input-dir", type=str, required=True, 
                        help="Directory containing transcript text files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for cleaned transcripts")
    args = parser.parse_args()
    
    # Clean transcripts
    clean_transcripts(args.input_dir, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
