#!/usr/bin/env python3
"""
Script to import chords and lyrics from a CSV file into the songwriting corpus.
This script processes a large CSV file containing chords and lyrics data,
extracts the relevant information, and adds it to the corpus.
"""

import os
import csv
import sys
import logging
import argparse
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("import_csv_chords_lyrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LYRICS_DIR = os.path.join(BASE_DIR, "combined", "lyrics")
CHORDS_DIR = os.path.join(BASE_DIR, "clean_chords")
MINIMAL_ARTIST_LIST = os.path.join(BASE_DIR, "MinimalArtistList.txt")
CSV_FILE = "/Users/jonathanschwartz/Downloads/archive/chords_and_lyrics.csv"

def load_minimal_artists():
    """Load the list of prioritized artists."""
    if not os.path.exists(MINIMAL_ARTIST_LIST):
        logger.error(f"Minimal artist list file not found: {MINIMAL_ARTIST_LIST}")
        return []
    
    with open(MINIMAL_ARTIST_LIST, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def clean_filename(name):
    """Clean a string to be used as a filename."""
    # Replace invalid filename characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip()

def extract_lyrics_from_chords_and_lyrics(chords_and_lyrics):
    """Extract just the lyrics from the combined chords and lyrics text."""
    if not chords_and_lyrics or pd.isna(chords_and_lyrics):
        return ""
    
    # Split into lines
    lines = chords_and_lyrics.split('\n')
    lyrics_lines = []
    section_header = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check if this is a section header
        if line.lower().startswith(('verse', 'chorus', 'bridge', 'pre-chorus', 'outro', 'intro', 'hook', 'refrain')):
            # Format as a proper section header
            section_name = line.split(':')[0].strip()
            section_header = f"[{section_name}]"
            lyrics_lines.append("")
            lyrics_lines.append(section_header)
            lyrics_lines.append("")
            continue
        
        # Skip lines that are just chord notations
        if all(c.isalpha() or c in "/#+" for c in line.replace(" ", "")):
            continue
        
        # Remove chord notations (typically at the beginning of lines or above lyrics)
        # This is a simplistic approach - might need refinement
        cleaned_line = line
        
        # Add the cleaned line to lyrics
        if cleaned_line:
            lyrics_lines.append(cleaned_line)
    
    # Join the lyrics lines
    lyrics = '\n'.join(lyrics_lines)
    
    return lyrics

def process_csv_file(prioritized_only=True):
    """Process the CSV file and import chords and lyrics into the corpus."""
    # Get the list of prioritized artists if needed
    prioritized_artists = None
    if prioritized_only:
        prioritized_artists = set(load_minimal_artists())
        logger.info(f"Loaded {len(prioritized_artists)} prioritized artists")
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        logger.error(f"CSV file not found: {CSV_FILE}")
        return 0
    
    # Load the CSV file
    logger.info(f"Loading CSV file: {CSV_FILE}")
    try:
        # Use pandas to handle large CSV files efficiently
        # Read in chunks to avoid memory issues
        chunk_size = 1000  # Adjust based on file size and available memory
        
        lyrics_count = 0
        chords_count = 0
        
        # Process the CSV file in chunks
        for chunk in pd.read_csv(CSV_FILE, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                try:
                    artist_name = row.get('artist_name', '')
                    song_name = row.get('song_name', '')
                    chords_and_lyrics = row.get('chords&lyrics', '')
                    
                    # Skip if artist or song name is missing
                    if not artist_name or not song_name or pd.isna(artist_name) or pd.isna(song_name):
                        continue
                    
                    # Skip if not a prioritized artist
                    if prioritized_only and artist_name not in prioritized_artists:
                        continue
                    
                    # Clean the artist and song names for use as filenames
                    clean_artist = clean_filename(artist_name)
                    clean_song = clean_filename(song_name)
                    
                    # Skip if no chords and lyrics data
                    if pd.isna(chords_and_lyrics) or not chords_and_lyrics:
                        continue
                    
                    # Create directories if they don't exist
                    lyrics_artist_dir = os.path.join(LYRICS_DIR, clean_artist)
                    chords_artist_dir = os.path.join(CHORDS_DIR, clean_artist)
                    os.makedirs(lyrics_artist_dir, exist_ok=True)
                    os.makedirs(chords_artist_dir, exist_ok=True)
                    
                    # Save chords and lyrics to file
                    chords_file = os.path.join(chords_artist_dir, f"{clean_song}.txt")
                    with open(chords_file, 'w', encoding='utf-8') as f:
                        f.write(f"{song_name}\n{artist_name}\n\n{chords_and_lyrics}")
                    chords_count += 1
                    
                    # Extract and save just the lyrics
                    lyrics = extract_lyrics_from_chords_and_lyrics(chords_and_lyrics)
                    if lyrics:
                        lyrics_file = os.path.join(lyrics_artist_dir, f"{clean_song}.txt")
                        with open(lyrics_file, 'w', encoding='utf-8') as f:
                            f.write(lyrics)
                        lyrics_count += 1
                    
                    if (lyrics_count + chords_count) % 100 == 0:
                        logger.info(f"Processed {lyrics_count} lyrics files and {chords_count} chord files")
                
                except Exception as e:
                    logger.error(f"Error processing row for {artist_name} - {song_name}: {e}")
        
        logger.info(f"Finished processing CSV file")
        logger.info(f"Created {lyrics_count} lyrics files and {chords_count} chord files")
        
        return lyrics_count + chords_count
    
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        return 0

def main():
    """Main function to import chords and lyrics from a CSV file."""
    parser = argparse.ArgumentParser(description="Import chords and lyrics from a CSV file")
    parser.add_argument("--all", action="store_true", help="Process all artists, not just prioritized ones")
    
    args = parser.parse_args()
    
    # Process the CSV file
    processed_count = process_csv_file(prioritized_only=not args.all)
    
    return 0 if processed_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
