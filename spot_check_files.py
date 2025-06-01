#!/usr/bin/env python3
"""
Script to spot check random lyrics and chord files from different artists.
"""

import os
import random
from pathlib import Path

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LYRICS_DIR = os.path.join(BASE_DIR, "combined", "lyrics")
CHORDS_DIR = os.path.join(BASE_DIR, "clean_chords")

def get_random_files(directory, file_type, count=5):
    """Get random files of the specified type from different artists."""
    all_artists = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    if not all_artists:
        print(f"No artist directories found in {directory}")
        return []
    
    # Shuffle the artists to ensure randomness
    random.shuffle(all_artists)
    
    selected_files = []
    selected_artists = set()
    
    # Try to get files from different artists
    for artist in all_artists:
        if len(selected_files) >= count:
            break
            
        artist_dir = os.path.join(directory, artist)
        files = [f for f in os.listdir(artist_dir) if f.endswith('.txt')]
        
        if files and artist not in selected_artists:
            random_file = random.choice(files)
            selected_files.append((artist, os.path.join(artist_dir, random_file)))
            selected_artists.add(artist)
    
    # If we still need more files, allow duplicates of artists
    if len(selected_files) < count:
        remaining_count = count - len(selected_files)
        print(f"Warning: Could not find {count} different artists. Using multiple files from same artists.")
        
        for artist in all_artists:
            if len(selected_files) >= count:
                break
                
            artist_dir = os.path.join(directory, artist)
            files = [f for f in os.listdir(artist_dir) if f.endswith('.txt')]
            
            if files:
                random_file = random.choice(files)
                selected_files.append((artist, os.path.join(artist_dir, random_file)))
    
    print(f"Selected {len(selected_files)} random {file_type} files from {len(selected_artists)} different artists")
    return selected_files

def display_file_content(file_path, max_lines=30):
    """Display the content of a file, limited to max_lines."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Get file name from path
        file_name = os.path.basename(file_path)
        artist_name = os.path.basename(os.path.dirname(file_path))
        
        print(f"\n{'='*80}")
        print(f"File: {file_name}")
        print(f"Artist: {artist_name}")
        print(f"{'='*80}")
        
        if len(lines) > max_lines:
            print("".join(lines[:max_lines]))
            print(f"\n... (truncated, {len(lines) - max_lines} more lines)")
        else:
            print("".join(lines))
        
        print(f"{'='*80}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

def main():
    """Main function to spot check random files."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get random lyrics files
    print("\nSpot checking random lyrics files:")
    lyrics_files = get_random_files(LYRICS_DIR, "lyrics")
    
    for artist, file_path in lyrics_files:
        display_file_content(file_path)
    
    # Get random chord files
    print("\nSpot checking random chord files:")
    chord_files = get_random_files(CHORDS_DIR, "chord")
    
    for artist, file_path in chord_files:
        display_file_content(file_path)

if __name__ == "__main__":
    main()
