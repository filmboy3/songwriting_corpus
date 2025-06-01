#!/usr/bin/env python3
"""
Complete Transcript Workflow

This script provides a complete workflow for processing Whisper transcripts:
1. Convert JSON transcripts to text
2. Clean and normalize the transcripts
3. Format and integrate them into the corpus
4. Generate a quality report

Usage:
  python complete_transcript_workflow.py --json-dir /path/to/json/transcripts --output-dir /path/to/corpus
"""

import os
import sys
import json
import re
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_workflow.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transcript_workflow")

def convert_json_to_text(json_path):
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
        
        return full_text.strip()
    
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return ""

def clean_transcript(text):
    """Clean and normalize transcript text."""
    # Remove excessive repetitions
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
    
    text = " ".join(cleaned_words)
    
    # Clean sponsor messages
    sponsor_patterns = [
        r"(?i)this episode is sponsored by.*?(?=\n\n|\.\s)",
        r"(?i)this podcast is brought to you by.*?(?=\n\n|\.\s)",
        r"(?i)today's show is supported by.*?(?=\n\n|\.\s)",
        r"(?i)use (promo|code) [A-Z0-9]+ for \d+% off",
        r"(?i)visit [a-z0-9]+\.com\/[a-z0-9]+ to get",
        r"(?i)for \d+% off (your first|mattresses|purchases)"
    ]
    
    for pattern in sponsor_patterns:
        text = re.sub(pattern, "[SPONSOR MESSAGE]", text)
    
    # Normalize formatting
    text = re.sub(r' +', ' ', text)  # Fix multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Fix multiple newlines
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)  # Fix spacing after punctuation
    
    # Replace common filler words
    fillers = ["um", "uh", "like", "you know", "I mean", "sort of", "kind of"]
    for filler in fillers:
        text = re.sub(f" {filler} ", " ", text)
    
    return text

def extract_songwriting_metadata(text):
    """Extract songwriting-related metadata from transcript text."""
    metadata = {}
    
    # Try to extract guests from text
    guest_patterns = [
        r"(?i)today'?s? guest is ([^.\n]+)",
        r"(?i)joining me today is ([^.\n]+)",
        r"(?i)I'?m joined by ([^.\n]+)",
        r"(?i)welcome ([^.\n]+) to the podcast",
        r"(?i)my guest today is ([^.\n]+)"
    ]
    
    guests = []
    for pattern in guest_patterns:
        matches = re.findall(pattern, text)
        guests.extend(matches)
    
    if guests:
        metadata['guests'] = guests
    
    # Extract songwriting-related topics
    songwriting_patterns = {
        'songwriting_process': [r'(?i)writing process', r'(?i)how (I|you|they) wrote', r'(?i)when (I|you|they) wrote'],
        'inspiration': [r'(?i)inspired by', r'(?i)inspiration for', r'(?i)came up with the idea'],
        'collaboration': [r'(?i)collaborated with', r'(?i)co-wrote', r'(?i)writing session'],
        'lyrics': [r'(?i)lyrics', r'(?i)words to the song', r'(?i)meaning of the song'],
        'melody': [r'(?i)melody', r'(?i)tune', r'(?i)musical hook'],
        'production': [r'(?i)produced', r'(?i)recording', r'(?i)in the studio']
    }
    
    topics = {}
    for topic, patterns in songwriting_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text):
                topics[topic] = True
                break
    
    if topics:
        metadata['songwriting_topics'] = list(topics.keys())
    
    return metadata

def format_for_corpus(text, filename):
    """Format transcript text for the corpus with appropriate tokens and songwriting metadata."""
    # Extract podcast name from filename
    podcast_name = os.path.splitext(os.path.basename(filename))[0]
    podcast_name = podcast_name.replace("_", " ")
    
    # Extract songwriting metadata
    metadata = extract_songwriting_metadata(text)
    
    # Format with tokens
    formatted_text = f"<PODCAST>\n"
    formatted_text += f"<TITLE>{podcast_name}</TITLE>\n"
    
    # Add metadata
    if 'guests' in metadata:
        formatted_text += f"<GUESTS>{', '.join(metadata['guests'])}</GUESTS>\n"
    
    if 'songwriting_topics' in metadata:
        formatted_text += f"<TOPICS>{', '.join(metadata['songwriting_topics'])}</TOPICS>\n"
    
    # Add transcript
    formatted_text += f"\n<TRANSCRIPT>\n{text}\n</TRANSCRIPT>\n"
    formatted_text += f"</PODCAST>"
    
    return formatted_text

def process_transcript(json_path, output_path):
    """Process a single transcript from JSON to formatted corpus text."""
    try:
        # Convert JSON to text
        raw_text = convert_json_to_text(json_path)
        if not raw_text:
            return False
        
        # Clean the transcript
        cleaned_text = clean_transcript(raw_text)
        
        # Format for corpus
        filename = os.path.basename(json_path)
        formatted_text = format_for_corpus(cleaned_text, filename)
        
        # Write to output file
        with open(output_path, 'w') as f:
            f.write(formatted_text)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return False

def process_transcripts(json_dir, output_dir, temp_dir=None):
    """Process all JSON transcripts and integrate them into the corpus."""
    json_dir = os.path.abspath(json_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory if specified
    if temp_dir:
        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} JSON transcripts to process")
    
    # Process each file
    success_count = 0
    for json_file in tqdm(json_files, desc="Processing"):
        json_path = os.path.join(json_dir, json_file)
        output_filename = os.path.splitext(json_file)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        
        # If temp directory is specified, use it for intermediate files
        if temp_dir:
            temp_path = os.path.join(temp_dir, output_filename)
            if process_transcript(json_path, temp_path):
                # Copy to final destination
                shutil.copy2(temp_path, output_path)
                success_count += 1
        else:
            if process_transcript(json_path, output_path):
                success_count += 1
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(json_files)} transcripts.")
    return success_count

def generate_quality_report(output_dir):
    """Generate a simple quality report for processed transcripts."""
    output_dir = os.path.abspath(output_dir)
    
    # Find all text files
    text_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    
    if not text_files:
        return "No transcripts found for quality report."
    
    # Calculate statistics
    total_files = len(text_files)
    word_counts = []
    char_counts = []
    
    for text_file in text_files:
        file_path = os.path.join(output_dir, text_file)
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                word_counts.append(len(content.split()))
                char_counts.append(len(content))
        except:
            continue
    
    if not word_counts:
        return "Could not read any transcript files for quality report."
    
    avg_word_count = sum(word_counts) / len(word_counts)
    avg_char_count = sum(char_counts) / len(char_counts)
    
    # Generate report
    report = "Transcript Processing Quality Report\n"
    report += "================================\n\n"
    report += f"Total transcripts processed: {total_files}\n"
    report += f"Average word count: {avg_word_count:.1f} words\n"
    report += f"Average character count: {avg_char_count:.1f} characters\n"
    
    return report

def main():
    """Main function to process transcripts."""
    parser = argparse.ArgumentParser(description="Process Whisper transcripts for corpus integration")
    parser.add_argument("--json-dir", type=str, required=True, 
                        help="Directory containing Whisper JSON transcripts")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed transcripts")
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="Temporary directory for intermediate files")
    args = parser.parse_args()
    
    # Process transcripts
    success_count = process_transcripts(args.json_dir, args.output_dir, args.temp_dir)
    
    # Generate quality report if any transcripts were processed
    if success_count > 0:
        report = generate_quality_report(args.output_dir)
        report_path = "transcript_quality_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Quality report saved to {report_path}")
        print("\nQuality Report:")
        print("==============")
        print("\n".join(report.split("\n")[:10]) + "\n...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
