#!/usr/bin/env python3
"""
Prioritized Song Collector for Songwriting Corpus
- Prioritizes artists from MinimalArtistList.txt
- Collects a specified number of songs per artist
- Incorporates chord progression analysis
- Runs continuously in the background to build corpus
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path

# Import existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from fetch_song_chords import process_artist_songs
    from clean_chord_data import create_clean_chord_file
    from analyze_chord_progressions import analyze_chord_file, generate_analysis_report
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all required scripts are in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("song_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MINIMAL_ARTIST_LIST = os.path.join(BASE_DIR, "MinimalArtistList.txt")
FULL_ARTIST_LIST = os.path.join(BASE_DIR, "artistList.txt")
LYRICS_DIR = os.path.join(BASE_DIR, "lyrics")
CHORDS_DIR = os.path.join(BASE_DIR, "chords")
COMBINED_DIR = os.path.join(BASE_DIR, "combined")
CLEAN_CHORDS_DIR = os.path.join(BASE_DIR, "clean_chords")
ANALYSIS_DIR = os.path.join(BASE_DIR, "chord_analysis")
PROGRESS_FILE = os.path.join(BASE_DIR, "collection_progress.json")

# Ensure directories exist
os.makedirs(LYRICS_DIR, exist_ok=True)
os.makedirs(CHORDS_DIR, exist_ok=True)
os.makedirs(CLEAN_CHORDS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def load_artist_list(file_path):
    """Load artist list from file."""
    if not os.path.exists(file_path):
        logger.error(f"Artist list file not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_progress():
    """Load collection progress from file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error parsing progress file. Starting fresh.")
    
    return {
        "minimal_artists": {},
        "full_artists": {},
        "last_run": None,
        "total_songs_collected": 0
    }

def save_progress(progress):
    """Save collection progress to file."""
    progress["last_run"] = datetime.now().isoformat()
    
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

def get_next_artist(progress, minimal_artists, full_artists, prioritize_minimal=True):
    """Get the next artist to process based on priority and progress."""
    # First, try to get an artist from the minimal list that hasn't reached the target
    if prioritize_minimal:
        for artist in minimal_artists:
            if artist not in progress["minimal_artists"] or progress["minimal_artists"][artist] < 50:
                return artist, "minimal"
    
    # If all minimal artists have reached the target or not prioritizing, try full list
    for artist in full_artists:
        if artist not in minimal_artists and (artist not in progress["full_artists"] or progress["full_artists"][artist] < 50):
            return artist, "full"
    
    # If all artists have reached the target, pick a random one for more songs
    if minimal_artists and prioritize_minimal:
        return random.choice(minimal_artists), "minimal"
    elif full_artists:
        return random.choice(full_artists), "full"
    
    return None, None

def process_artist(artist, artist_type, progress, songs_per_artist=50, analyze_chords=True):
    """Process an artist by fetching songs, cleaning, and analyzing."""
    logger.info(f"Processing artist: {artist} (from {artist_type} list)")
    
    # Create artist directories
    artist_lyrics_dir = os.path.join(LYRICS_DIR, artist)
    artist_chords_dir = os.path.join(CLEAN_CHORDS_DIR, artist)
    artist_analysis_dir = os.path.join(ANALYSIS_DIR, artist)
    
    os.makedirs(artist_lyrics_dir, exist_ok=True)
    os.makedirs(artist_chords_dir, exist_ok=True)
    os.makedirs(artist_analysis_dir, exist_ok=True)
    
    # Get current song count
    if os.path.exists(artist_lyrics_dir):
        current_count = len([f for f in os.listdir(artist_lyrics_dir) if f.endswith('.txt')])
    else:
        current_count = 0
    
    # Check if we already have enough songs
    if current_count >= songs_per_artist:
        logger.info(f"Already have {current_count} songs for {artist}, skipping")
        
        # Update progress
        if artist_type == "minimal":
            progress["minimal_artists"][artist] = current_count
        else:
            progress["full_artists"][artist] = current_count
        
        return 0
    
    # Fetch songs using the existing process_artist_songs function
    try:
        # Use the existing function to fetch songs and chords
        songs_added = process_artist_songs(artist, max_songs=songs_per_artist, delay_range=(1, 3))
        
        if songs_added == 0:
            logger.warning(f"No songs found or processed for {artist}")
            return 0
        
        # After fetching songs, analyze chord progressions if requested
        if analyze_chords:
            # Get all chord files for this artist
            chord_files = []
            if os.path.exists(os.path.join(CHORDS_DIR, artist)):
                chord_files = [f for f in os.listdir(os.path.join(CHORDS_DIR, artist)) if f.endswith('.txt')]
            
            for chord_file in chord_files:
                try:
                    full_chord_path = os.path.join(CHORDS_DIR, artist, chord_file)
                    song_name = os.path.splitext(chord_file)[0]
                    
                    # Clean chord data
                    clean_file = os.path.join(artist_chords_dir, chord_file)
                    create_clean_chord_file(full_chord_path, artist_chords_dir)
                    
                    # Analyze chord progressions
                    if os.path.exists(clean_file):
                        analysis = analyze_chord_file(clean_file)
                        if analysis:
                            analysis_file = os.path.join(artist_analysis_dir, f"{song_name}_analysis.md")
                            generate_analysis_report(analysis, analysis_file, clean_file)
                            
                            # Save analysis as JSON
                            json_file = os.path.join(artist_analysis_dir, f"{song_name}_analysis.json")
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(analysis, f, indent=2)
                except Exception as e:
                    logger.error(f"Error analyzing chords for {chord_file}: {e}")
        
        # Update progress
        if artist_type == "minimal":
            progress["minimal_artists"][artist] = current_count + songs_added
        else:
            progress["full_artists"][artist] = current_count + songs_added
        
        progress["total_songs_collected"] += songs_added
        
        logger.info(f"Added {songs_added} songs for {artist}")
        return songs_added
    
    except Exception as e:
        logger.error(f"Error processing artist {artist}: {e}")
        return 0

def continuous_collection(minimal_artists, full_artists, songs_per_batch=10, 
                         sleep_time=300, prioritize_minimal=True, analyze_chords=True):
    """Continuously collect songs in the background."""
    logger.info("Starting continuous song collection...")
    
    while True:
        progress = load_progress()
        songs_collected = 0
        
        for _ in range(songs_per_batch):
            artist, artist_type = get_next_artist(progress, minimal_artists, full_artists, prioritize_minimal)
            
            if not artist:
                logger.info("No more artists to process")
                break
            
            songs_added = process_artist(artist, artist_type, progress, 
                                        songs_per_artist=5, analyze_chords=analyze_chords)
            songs_collected += songs_added
            
            # Save progress after each artist
            save_progress(progress)
            
            # Small pause between artists to be nice to the servers
            time.sleep(5)
        
        logger.info(f"Batch complete. Collected {songs_collected} songs. Total: {progress['total_songs_collected']}")
        logger.info(f"Sleeping for {sleep_time} seconds before next batch...")
        
        # Sleep between batches
        time.sleep(sleep_time)

def initial_collection(minimal_artists, full_artists, songs_per_artist=50, 
                      prioritize_minimal=True, analyze_chords=True):
    """Perform initial collection of songs to quickly build corpus."""
    logger.info("Starting initial song collection...")
    
    progress = load_progress()
    total_collected = 0
    
    # First process minimal artists
    if prioritize_minimal:
        for artist in minimal_artists:
            songs_added = process_artist(artist, "minimal", progress, 
                                        songs_per_artist=songs_per_artist, 
                                        analyze_chords=analyze_chords)
            total_collected += songs_added
            save_progress(progress)
    
    # Then process full artists (if time permits)
    for artist in full_artists:
        if artist not in minimal_artists:
            songs_added = process_artist(artist, "full", progress, 
                                        songs_per_artist=songs_per_artist, 
                                        analyze_chords=analyze_chords)
            total_collected += songs_added
            save_progress(progress)
    
    logger.info(f"Initial collection complete. Total songs collected: {total_collected}")
    return total_collected

def main():
    """Main function to run the song collector."""
    parser = argparse.ArgumentParser(description="Prioritized Song Collector for Songwriting Corpus")
    parser.add_argument("--initial", action="store_true", help="Perform initial collection")
    parser.add_argument("--continuous", action="store_true", help="Run continuous collection in background")
    parser.add_argument("--songs-per-artist", type=int, default=50, help="Number of songs to collect per artist")
    parser.add_argument("--songs-per-batch", type=int, default=10, help="Number of songs to collect per batch in continuous mode")
    parser.add_argument("--sleep-time", type=int, default=300, help="Sleep time between batches in continuous mode (seconds)")
    parser.add_argument("--no-analyze", action="store_true", help="Skip chord analysis")
    parser.add_argument("--no-prioritize", action="store_true", help="Don't prioritize minimal artist list")
    
    args = parser.parse_args()
    
    # Load artist lists
    minimal_artists = load_artist_list(MINIMAL_ARTIST_LIST)
    full_artists = load_artist_list(FULL_ARTIST_LIST)
    
    logger.info(f"Loaded {len(minimal_artists)} artists from minimal list")
    logger.info(f"Loaded {len(full_artists)} artists from full list")
    
    # Run in the requested mode
    if args.initial:
        initial_collection(
            minimal_artists, 
            full_artists,
            songs_per_artist=args.songs_per_artist,
            prioritize_minimal=not args.no_prioritize,
            analyze_chords=not args.no_analyze
        )
    
    if args.continuous:
        if args.initial:
            # If both modes are requested, run continuous in a separate thread
            threading.Thread(
                target=continuous_collection,
                args=(minimal_artists, full_artists),
                kwargs={
                    'songs_per_batch': args.songs_per_batch,
                    'sleep_time': args.sleep_time,
                    'prioritize_minimal': not args.no_prioritize,
                    'analyze_chords': not args.no_analyze
                },
                daemon=True
            ).start()
        else:
            # Otherwise run continuous in the main thread
            continuous_collection(
                minimal_artists, 
                full_artists,
                songs_per_batch=args.songs_per_batch,
                sleep_time=args.sleep_time,
                prioritize_minimal=not args.no_prioritize,
                analyze_chords=not args.no_analyze
            )
    
    # If neither mode is specified, run initial collection
    if not args.initial and not args.continuous:
        initial_collection(
            minimal_artists, 
            full_artists,
            songs_per_artist=args.songs_per_artist,
            prioritize_minimal=not args.no_prioritize,
            analyze_chords=not args.no_analyze
        )

if __name__ == "__main__":
    main()
