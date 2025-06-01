#!/usr/bin/env python3
"""
Script to count the number of songs per artist in the songwriting corpus.
Displays artists sorted by the number of songs they have in the corpus.
"""

import os
import sys
from collections import Counter
from pathlib import Path

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LYRICS_DIR = os.path.join(BASE_DIR, "combined", "lyrics")
CHORDS_DIR = os.path.join(BASE_DIR, "clean_chords")
MINIMAL_ARTIST_LIST = os.path.join(BASE_DIR, "MinimalArtistList.txt")

def load_minimal_artists():
    """Load the list of prioritized artists."""
    if not os.path.exists(MINIMAL_ARTIST_LIST):
        print(f"Minimal artist list file not found: {MINIMAL_ARTIST_LIST}")
        return []
    
    with open(MINIMAL_ARTIST_LIST, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def count_songs_by_artist():
    """Count the number of songs per artist in the corpus."""
    # Get the list of prioritized artists
    prioritized_artists = set(load_minimal_artists())
    print(f"Loaded {len(prioritized_artists)} prioritized artists")
    
    # Count songs in lyrics directory
    lyrics_counts = Counter()
    if os.path.exists(LYRICS_DIR):
        for artist_dir in os.listdir(LYRICS_DIR):
            artist_path = os.path.join(LYRICS_DIR, artist_dir)
            if os.path.isdir(artist_path):
                song_count = len([f for f in os.listdir(artist_path) if f.endswith('.txt')])
                lyrics_counts[artist_dir] = song_count
    
    # Count songs in chords directory
    chords_counts = Counter()
    if os.path.exists(CHORDS_DIR):
        for artist_dir in os.listdir(CHORDS_DIR):
            artist_path = os.path.join(CHORDS_DIR, artist_dir)
            if os.path.isdir(artist_path):
                song_count = len([f for f in os.listdir(artist_path) if f.endswith('.txt')])
                chords_counts[artist_dir] = song_count
    
    # Combine counts
    all_artists = set(list(lyrics_counts.keys()) + list(chords_counts.keys()))
    combined_counts = []
    
    for artist in all_artists:
        lyrics_count = lyrics_counts.get(artist, 0)
        chords_count = chords_counts.get(artist, 0)
        is_prioritized = artist in prioritized_artists
        combined_counts.append((artist, lyrics_count, chords_count, is_prioritized))
    
    # Sort by total song count (lyrics + chords)
    combined_counts.sort(key=lambda x: x[1] + x[2], reverse=True)
    
    # Print results
    print("\nArtist Song Distribution (sorted by total song count):")
    print("-" * 80)
    print(f"{'Artist':<40} {'Lyrics':<10} {'Chords':<10} {'Total':<10} {'Prioritized'}")
    print("-" * 80)
    
    total_lyrics = 0
    total_chords = 0
    total_prioritized_lyrics = 0
    total_prioritized_chords = 0
    
    for artist, lyrics_count, chords_count, is_prioritized in combined_counts:
        total = lyrics_count + chords_count
        print(f"{artist[:39]:<40} {lyrics_count:<10} {chords_count:<10} {total:<10} {'Yes' if is_prioritized else 'No'}")
        
        total_lyrics += lyrics_count
        total_chords += chords_count
        if is_prioritized:
            total_prioritized_lyrics += lyrics_count
            total_prioritized_chords += chords_count
    
    print("-" * 80)
    print(f"{'TOTAL':<40} {total_lyrics:<10} {total_chords:<10} {total_lyrics + total_chords:<10}")
    print(f"{'PRIORITIZED TOTAL':<40} {total_prioritized_lyrics:<10} {total_prioritized_chords:<10} {total_prioritized_lyrics + total_prioritized_chords:<10}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total number of artists: {len(all_artists)}")
    print(f"Total number of prioritized artists: {len(prioritized_artists)}")
    print(f"Total number of lyrics files: {total_lyrics}")
    print(f"Total number of chord files: {total_chords}")
    print(f"Total number of prioritized artist lyrics files: {total_prioritized_lyrics}")
    print(f"Total number of prioritized artist chord files: {total_prioritized_chords}")
    
    # Print top 20 artists
    print("\nTop 20 Artists by Song Count:")
    for i, (artist, lyrics_count, chords_count, is_prioritized) in enumerate(combined_counts[:20], 1):
        total = lyrics_count + chords_count
        print(f"{i:2}. {artist} - {total} songs {'(prioritized)' if is_prioritized else ''}")
    
    return combined_counts

if __name__ == "__main__":
    count_songs_by_artist()
