#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

def check_progress(mp3_dir, transcript_dir):
    """Check the progress of the transcription process."""
    mp3_dir = Path(mp3_dir)
    transcript_dir = Path(transcript_dir)
    
    # Get list of all MP3 files
    mp3_files = list(mp3_dir.glob("*.mp3"))
    total_mp3s = len(mp3_files)
    
    # Get list of all transcript files
    transcript_files = list(transcript_dir.glob("*.json"))
    completed_transcripts = len(transcript_files)
    
    # Calculate progress
    if total_mp3s > 0:
        progress_percentage = (completed_transcripts / total_mp3s) * 100
    else:
        progress_percentage = 0
    
    # Check the most recent transcript file to estimate time per file
    if transcript_files:
        transcript_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        most_recent = transcript_files[0]
        most_recent_time = datetime.fromtimestamp(most_recent.stat().st_mtime)
        time_now = datetime.now()
        time_diff = time_now - most_recent_time
        
        # Print progress information
        print(f"\nTranscription Progress Report ({time_now.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"{'='*50}")
        print(f"Total MP3 files: {total_mp3s}")
        print(f"Completed transcripts: {completed_transcripts}")
        print(f"Remaining files: {total_mp3s - completed_transcripts}")
        print(f"Progress: {progress_percentage:.1f}%")
        print(f"Most recent transcript: {most_recent.stem}")
        print(f"Last transcript completed: {time_diff.total_seconds():.0f} seconds ago")
        
        # List the 5 most recently completed transcripts
        print(f"\nRecently completed transcripts:")
        for i, tf in enumerate(transcript_files[:5]):
            file_time = datetime.fromtimestamp(tf.stat().st_mtime)
            print(f"  {i+1}. {tf.stem} ({file_time.strftime('%H:%M:%S')})")
        
        # Estimate remaining time
        if len(transcript_files) >= 2:
            # Get the second most recent file for better time estimation
            second_recent = transcript_files[1]
            second_recent_time = datetime.fromtimestamp(second_recent.stat().st_mtime)
            time_per_file = (most_recent_time - second_recent_time).total_seconds()
            
            remaining_files = total_mp3s - completed_transcripts
            estimated_seconds = remaining_files * time_per_file
            estimated_hours = estimated_seconds / 3600
            
            print(f"\nEstimated time per file: {time_per_file:.1f} seconds")
            print(f"Estimated time remaining: {estimated_hours:.1f} hours")
            print(f"Estimated completion: {(datetime.now().timestamp() + estimated_seconds):.0f}")
    else:
        print("No transcripts found yet.")

def main():
    parser = argparse.ArgumentParser(description="Check transcription progress")
    parser.add_argument("--mp3_dir", default="/Users/jonathanschwartz/CascadeProjects/songwriting_corpus/podcasts/mp3s", help="Directory containing MP3 files")
    parser.add_argument("--transcript_dir", default="/Users/jonathanschwartz/CascadeProjects/songwriting_corpus/podcasts/transcripts", help="Directory containing transcript files")
    args = parser.parse_args()
    
    check_progress(args.mp3_dir, args.transcript_dir)

if __name__ == "__main__":
    main()
