#!/usr/bin/env python3
"""
Improved Transcript Cleaner

This module provides enhanced functions for cleaning and normalizing podcast transcripts,
with special focus on handling repetitive phrases and sponsor messages.
"""

import re
import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_cleaning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("transcript_cleaner")

def clean_transcript(text):
    """
    Clean and normalize transcript text with improved handling of repetitions.
    
    This function addresses common issues in Whisper transcripts:
    - Exact and near-repetitions of phrases
    - Sponsor messages and advertisements
    - Formatting inconsistencies
    - Filler sounds and words
    """
    # First, normalize whitespace to make pattern matching more reliable
    text = re.sub(r'\s+', ' ', text)
    
    # Handle exact repetitions of longer phrases (common in Whisper transcripts)
    # This uses a more efficient regex-based approach for exact repetitions
    for phrase_len in range(20, 3, -1):  # Start with longer phrases (more specific)
        pattern = r'((?:\b\w+\b[\s,]*){' + str(phrase_len) + r'})\s*\1+'
        text = re.sub(pattern, r'\1', text)
    
    # Handle near-repetitions with small variations (more common in podcast transcripts)
    words = text.split()
    cleaned_words = []
    i = 0
    
    while i < len(words):
        # Skip if we're near the end
        if i + 10 >= len(words):
            cleaned_words.extend(words[i:])
            break
        
        # Check for repetition patterns with fuzzy matching
        found_repetition = False
        for phrase_len in range(4, 12):  # Check phrases of length 4 to 11 words
            if i + phrase_len * 2 >= len(words):
                continue
                
            phrase1 = " ".join(words[i:i+phrase_len])
            phrase2 = " ".join(words[i+phrase_len:i+phrase_len*2])
            
            # Calculate similarity between phrases
            similarity = 0
            if len(phrase1) > 0 and len(phrase2) > 0:
                # Simple similarity: count of matching words / total words
                words1 = set(phrase1.lower().split())
                words2 = set(phrase2.lower().split())
                common_words = words1.intersection(words2)
                all_words = words1.union(words2)
                similarity = len(common_words) / len(all_words) if all_words else 0
            
            # If phrases are very similar (80% or more matching words)
            if similarity >= 0.8:
                # Check how many times similar phrases repeat
                repeat_count = 2
                next_pos = i + phrase_len * 2
                
                while next_pos + phrase_len <= len(words):
                    next_phrase = " ".join(words[next_pos:next_pos+phrase_len])
                    next_similarity = 0
                    
                    # Calculate similarity with the first phrase
                    words_next = set(next_phrase.lower().split())
                    common_words = words1.intersection(words_next)
                    all_words = words1.union(words_next)
                    next_similarity = len(common_words) / len(all_words) if all_words else 0
                    
                    if next_similarity >= 0.7:  # Slightly lower threshold for subsequent repetitions
                        repeat_count += 1
                        next_pos += phrase_len
                    else:
                        break
                
                # If it repeats more than 2 times, only keep one instance
                if repeat_count >= 2:
                    cleaned_words.extend(words[i:i+phrase_len])
                    i = next_pos
                    found_repetition = True
                    break
        
        # If no repetition found, add current word and move on
        if not found_repetition:
            cleaned_words.append(words[i])
            i += 1
    
    text = " ".join(cleaned_words)
    
    # Clean sponsor messages with expanded patterns
    sponsor_patterns = [
        # Common podcast sponsor introductions
        r"(?i)this episode is sponsored by.*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)this podcast is brought to you by.*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)today'?s show is supported by.*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)I'?d like to thank (?:our|my) sponsors?.*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)special thanks to our sponsors?.*?(?=\n\n|\.\s|\n\w|$)",
        
        # Promo codes and calls to action
        r"(?i)use (?:the )?(?:promo|code) [A-Za-z0-9_-]+ for \d+% off.*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)visit [a-z0-9][a-z0-9-]*\.[a-z]{2,}(?:/[a-z0-9-]+)? (?:and|to) (?:get|receive).*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)go to [a-z0-9][a-z0-9-]*\.[a-z]{2,}(?:/[a-z0-9-]+)?.*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)for \d+% off (?:your first|your next|all|everything|mattresses|purchases).*?(?=\n\n|\.\s|\n\w|$)",
        
        # Common podcast ad phrases
        r"(?i)that'?s [a-z0-9][a-z0-9-]*\.[a-z]{2,}(?:/[a-z0-9-]+)?.*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)offer code [A-Za-z0-9_-]+.*?(?=\n\n|\.\s|\n\w|$)",
        r"(?i)terms and conditions apply.*?(?=\n\n|\.\s|\n\w|$)"
    ]
    
    for pattern in sponsor_patterns:
        text = re.sub(pattern, "[SPONSOR MESSAGE]", text)
    
    # Normalize formatting
    text = re.sub(r' +', ' ', text)  # Fix multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Fix multiple newlines
    
    # Remove excessive [SPONSOR MESSAGE] repetitions
    text = re.sub(r'\[SPONSOR MESSAGE\](\s*\[SPONSOR MESSAGE\])+', '[SPONSOR MESSAGE]', text)
    
    # Fix common Whisper transcription artifacts
    text = re.sub(r'(?i)\b(i{2,})\b', 'I', text)  # Fix multiple i's ("iii" -> "I")
    text = re.sub(r'(?i)\bmm+\b', 'um', text)  # Fix "mmmm" -> "um"
    text = re.sub(r'(?i)\baa+h\b', 'ah', text)  # Fix "aaaah" -> "ah"
    
    # Remove lines that are just filler sounds
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are just filler sounds
        if re.match(r'^\s*(?:um+|uh+|hmm+|ah+|er+)\s*$', line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    return text

def extract_songwriting_content(text):
    """
    Extract songwriting-specific content from transcript text.
    
    This function identifies sections of the transcript that are specifically
    about songwriting techniques, processes, or discussions.
    """
    songwriting_sections = []
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Keywords related to songwriting
    songwriting_keywords = [
        r'\bwrit(?:e|ing|ten) (?:a |the )?(?:song|lyric|melody|chorus|verse)',
        r'\bsongwrit(?:e|ing|er)',
        r'\blyric(?:s|al)',
        r'\bmelod(?:y|ic|ies)',
        r'\bchorus',
        r'\bverse',
        r'\bbridge',
        r'\bhook',
        r'\bcompos(?:e|ing|er|ition)',
        r'\bmusic(?:al)? (?:idea|concept|theme)',
        r'\bharmoni(?:c|es|y)',
        r'\bchord progression',
        r'\bkey of [A-G]',
        r'\bminor key',
        r'\bmajor key',
        r'\brecord(?:ing|ed) (?:a |the )?(?:song|album|track)',
        r'\bstudio session',
        r'\bco-writ(?:e|ing|ten)',
        r'\bcollaborat(?:e|ion|ing|ed)',
        r'\binspir(?:e|ed|ation)'
    ]
    
    # Check each paragraph for songwriting content
    for paragraph in paragraphs:
        if len(paragraph.strip()) < 20:  # Skip very short paragraphs
            continue
            
        # Check if paragraph contains songwriting keywords
        is_songwriting_content = False
        for keyword in songwriting_keywords:
            if re.search(keyword, paragraph, re.IGNORECASE):
                is_songwriting_content = True
                break
                
        if is_songwriting_content:
            songwriting_sections.append(paragraph.strip())
    
    return songwriting_sections

def process_transcript(json_path, output_path):
    """Process a single transcript from JSON to cleaned text."""
    try:
        # Load JSON transcript
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
        
        # Clean the transcript
        cleaned_text = clean_transcript(full_text.strip())
        
        # Extract songwriting content
        songwriting_sections = extract_songwriting_content(cleaned_text)
        
        # Format output
        output = {
            "original_length": len(full_text.split()),
            "cleaned_length": len(cleaned_text.split()),
            "full_text": cleaned_text,
            "songwriting_sections": songwriting_sections,
            "has_songwriting_content": len(songwriting_sections) > 0
        }
        
        # Write to output file
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return False

def main():
    """Main function to process transcripts."""
    parser = argparse.ArgumentParser(description="Clean and process Whisper transcripts")
    parser.add_argument("--json-dir", type=str, required=True, 
                        help="Directory containing Whisper JSON transcripts")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed transcripts")
    parser.add_argument("--file", type=str, 
                        help="Process a single file instead of the entire directory")
    args = parser.parse_args()
    
    json_dir = os.path.abspath(args.json_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process a single file if specified
    if args.file:
        json_path = os.path.join(json_dir, args.file)
        output_path = os.path.join(output_dir, os.path.splitext(args.file)[0] + ".json")
        
        if process_transcript(json_path, output_path):
            logger.info(f"Successfully processed {args.file}")
        else:
            logger.error(f"Failed to process {args.file}")
        
        return 0
    
    # Process all files in the directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} JSON transcripts to process")
    
    success_count = 0
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        output_path = os.path.join(output_dir, os.path.splitext(json_file)[0] + ".json")
        
        if process_transcript(json_path, output_path):
            success_count += 1
            logger.info(f"Processed {json_file} ({success_count}/{len(json_files)})")
        else:
            logger.error(f"Failed to process {json_file}")
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(json_files)} transcripts.")
    
    # Generate summary
    songwriting_count = 0
    for output_file in os.listdir(output_dir):
        if not output_file.endswith('.json'):
            continue
            
        try:
            with open(os.path.join(output_dir, output_file), 'r') as f:
                data = json.load(f)
                if data.get('has_songwriting_content', False):
                    songwriting_count += 1
        except:
            continue
    
    logger.info(f"Found {songwriting_count} transcripts with songwriting content")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
