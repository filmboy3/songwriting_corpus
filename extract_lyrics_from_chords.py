#!/usr/bin/env python3
"""
Script to extract lyrics from chord files and create corresponding lyrics files.
This ensures that lyrics match the musical structure in the chord files.
"""

import os
import re
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("extract_lyrics_from_chords.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHORDS_DIR = os.path.join(BASE_DIR, "clean_chords")
LYRICS_DIR = os.path.join(BASE_DIR, "combined", "lyrics")
MINIMAL_ARTIST_LIST = os.path.join(BASE_DIR, "MinimalArtistList.txt")

def load_minimal_artists():
    """Load the list of prioritized artists."""
    if not os.path.exists(MINIMAL_ARTIST_LIST):
        logger.error(f"Minimal artist list file not found: {MINIMAL_ARTIST_LIST}")
        return []
    
    with open(MINIMAL_ARTIST_LIST, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_lyrics_from_chord_file(chord_file_path):
    """Extract lyrics from a chord file, removing chord notations."""
    with open(chord_file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Remove chord notations [X] where X is a chord
    content = re.sub(r'\[[A-Za-z0-9#/\+\-]+\]', '', content)
    
    # Keep section headers like [Verse], [Chorus], etc.
    section_headers = re.findall(r'\[(Verse|Chorus|Bridge|Pre-Chorus|Outro|Intro|Hook|Refrain|Instrumental|Solo|Breakdown|Interlude).*?\]', content)
    
    # Clean up the content
    lines = content.split('\n')
    cleaned_lines = []
    in_header = True  # Skip the header info at the beginning
    header_count = 0
    
    for line in lines:
        # Skip empty lines at the beginning
        if in_header and not line.strip():
            continue
        
        # Skip the first few lines (usually title, artist, etc.)
        if in_header:
            header_count += 1
            if header_count > 3:  # Skip the first 3 non-empty lines
                in_header = False
            continue
        
        # Clean the line
        cleaned_line = line.strip()
        
        # Skip lines that are just spaces or tabs or other formatting
        if not cleaned_line or cleaned_line.startswith('|') or cleaned_line.startswith('-'):
            continue
        
        # Keep section headers and lyrics
        if re.match(r'\[(Verse|Chorus|Bridge|Pre-Chorus|Outro|Intro|Hook|Refrain|Instrumental|Solo|Breakdown|Interlude).*?\]', cleaned_line):
            cleaned_lines.append('')  # Add blank line before section header
            cleaned_lines.append(cleaned_line)
            cleaned_lines.append('')  # Add blank line after section header
        elif cleaned_line:
            # This is a lyrics line
            cleaned_lines.append(cleaned_line)
    
    # Join the cleaned lines
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Remove consecutive blank lines
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    return cleaned_content.strip()

def process_chord_files(prioritized_only=True):
    """Process chord files and create corresponding lyrics files."""
    # Get the list of prioritized artists if needed
    prioritized_artists = None
    if prioritized_only:
        prioritized_artists = set(load_minimal_artists())
        logger.info(f"Loaded {len(prioritized_artists)} prioritized artists")
    
    processed_count = 0
    total_count = 0
    
    # Walk through the chords directory
    for root, dirs, files in os.walk(CHORDS_DIR):
        # Get the artist name from the directory path
        artist_name = os.path.basename(root)
        
        # Skip if we're only processing prioritized artists and this one isn't in the list
        if prioritized_only and artist_name not in prioritized_artists:
            continue
        
        # Process each chord file
        for file_name in files:
            if not file_name.endswith('.txt'):
                continue
                
            total_count += 1
            chord_file = os.path.join(root, file_name)
            song_title = os.path.splitext(file_name)[0]
            
            # Create the corresponding lyrics directory if it doesn't exist
            lyrics_artist_dir = os.path.join(LYRICS_DIR, artist_name)
            os.makedirs(lyrics_artist_dir, exist_ok=True)
            
            # Define the output lyrics file path
            lyrics_file = os.path.join(lyrics_artist_dir, f"{song_title}.txt")
            
            # Extract lyrics from the chord file
            try:
                logger.info(f"Extracting lyrics from chord file: {artist_name} - {song_title}")
                lyrics_content = extract_lyrics_from_chord_file(chord_file)
                
                # Save the lyrics to the output file
                with open(lyrics_file, 'w', encoding='utf-8') as f:
                    f.write(lyrics_content)
                
                processed_count += 1
                logger.info(f"Created lyrics file: {lyrics_file}")
            except Exception as e:
                logger.error(f"Error processing file {chord_file}: {e}")
    
    logger.info(f"Processed {total_count} chord files")
    logger.info(f"Created {processed_count} lyrics files")
    
    return processed_count

def main():
    """Main function to extract lyrics from chord files."""
    parser = argparse.ArgumentParser(description="Extract lyrics from chord files")
    parser.add_argument("--all", action="store_true", help="Process all artists, not just prioritized ones")
    parser.add_argument("--specific-file", type=str, help="Process a specific chord file (provide full path)")
    
    args = parser.parse_args()
    
    if args.specific_file:
        # Process a specific file
        if os.path.exists(args.specific_file):
            try:
                # Extract the artist and song title from the file path
                file_path = Path(args.specific_file)
                artist_name = file_path.parent.name
                song_title = file_path.stem
                
                # Create the corresponding lyrics directory if it doesn't exist
                lyrics_artist_dir = os.path.join(LYRICS_DIR, artist_name)
                os.makedirs(lyrics_artist_dir, exist_ok=True)
                
                # Define the output lyrics file path
                lyrics_file = os.path.join(lyrics_artist_dir, f"{song_title}.txt")
                
                # Extract lyrics from the chord file
                logger.info(f"Extracting lyrics from chord file: {artist_name} - {song_title}")
                lyrics_content = extract_lyrics_from_chord_file(args.specific_file)
                
                # Save the lyrics to the output file
                with open(lyrics_file, 'w', encoding='utf-8') as f:
                    f.write(lyrics_content)
                
                logger.info(f"Created lyrics file: {lyrics_file}")
                return 0
            except Exception as e:
                logger.error(f"Error processing file {args.specific_file}: {e}")
                return 1
        else:
            logger.error(f"File not found: {args.specific_file}")
            return 1
    else:
        # Process all files
        processed_count = process_chord_files(prioritized_only=not args.all)
        return 0 if processed_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
