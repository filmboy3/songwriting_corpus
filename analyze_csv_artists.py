#!/usr/bin/env python3
"""
Script to analyze the distribution of artists in the CSV file only.
"""

import os
import pandas as pd
from collections import Counter

# CSV file path
CSV_FILE = "/Users/jonathanschwartz/Downloads/archive/chords_and_lyrics.csv"

def analyze_csv_artists():
    """Analyze the distribution of artists in the CSV file."""
    print(f"Analyzing CSV file: {CSV_FILE}")
    
    # Load the CSV file in chunks to handle its large size
    artist_counter = Counter()
    chunk_size = 1000
    
    # Process the CSV file in chunks
    for chunk in pd.read_csv(CSV_FILE, chunksize=chunk_size):
        # Count artists in this chunk
        for artist in chunk['artist_name']:
            if pd.notna(artist) and artist:
                artist_counter[artist] += 1
    
    # Sort artists by song count
    sorted_artists = sorted(artist_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    print("\nArtist Distribution in CSV File (Top 50):")
    print("-" * 60)
    print(f"{'Rank':<5} {'Artist':<40} {'Song Count':<10}")
    print("-" * 60)
    
    for i, (artist, count) in enumerate(sorted_artists[:50], 1):
        print(f"{i:<5} {artist[:39]:<40} {count:<10}")
    
    # Print summary statistics
    total_artists = len(artist_counter)
    total_songs = sum(artist_counter.values())
    
    print("\nSummary Statistics for CSV File:")
    print(f"Total number of artists: {total_artists}")
    print(f"Total number of songs: {total_songs}")
    print(f"Average songs per artist: {total_songs / total_artists:.2f}")
    
    # Print distribution statistics
    artists_with_one_song = sum(1 for count in artist_counter.values() if count == 1)
    artists_with_over_100_songs = sum(1 for count in artist_counter.values() if count > 100)
    
    print(f"Artists with only one song: {artists_with_one_song} ({artists_with_one_song/total_artists*100:.2f}%)")
    print(f"Artists with over 100 songs: {artists_with_over_100_songs} ({artists_with_over_100_songs/total_artists*100:.2f}%)")
    
    return sorted_artists

if __name__ == "__main__":
    analyze_csv_artists()
