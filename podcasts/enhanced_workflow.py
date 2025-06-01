#!/usr/bin/env python3
"""
Enhanced Podcast Transcription Workflow

This script provides a unified workflow for the songwriting corpus podcast pipeline:
1. Download podcast episodes from RSS feeds
2. Transcribe audio with Whisper (if needed)
3. Clean and process transcripts
4. Extract songwriting content
5. Integrate into the corpus with proper formatting
6. Generate quality reports

Usage:
  python enhanced_workflow.py --mode download --rss-url URL --outdir DIR
  python enhanced_workflow.py --mode transcribe --mp3-dir DIR --transcript-dir DIR
  python enhanced_workflow.py --mode process --transcript-dir DIR --corpus-dir DIR
  python enhanced_workflow.py --mode all --rss-url URL --corpus-dir DIR
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_workflow")

def setup_directories(base_dir):
    """Set up the directory structure for the workflow."""
    dirs = {
        'mp3': os.path.join(base_dir, 'mp3s'),
        'transcripts': os.path.join(base_dir, 'transcripts'),
        'processed': os.path.join(base_dir, 'processed'),
        'corpus': os.path.join(base_dir, 'corpus')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return dirs

def download_podcasts(rss_url, output_dir, limit=None, newest_first=True):
    """Download podcast episodes from an RSS feed."""
    logger.info(f"Downloading podcasts from {rss_url} to {output_dir}")
    
    cmd = [
        "python", 
        "rss_to_mp3_downloader.py", 
        rss_url, 
        "--outdir", output_dir
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if newest_first:
        cmd.append("--newest-first")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading podcasts: {e}")
        logger.error(e.stderr)
        return False

def transcribe_audio(mp3_dir, transcript_dir, model="tiny", device="cpu"):
    """Transcribe audio files using Whisper."""
    logger.info(f"Transcribing audio files from {mp3_dir} to {transcript_dir}")
    
    # Find MP3 files that haven't been transcribed yet
    mp3_files = [f for f in os.listdir(mp3_dir) if f.lower().endswith('.mp3')]
    existing_transcripts = [os.path.splitext(f)[0] for f in os.listdir(transcript_dir) if f.endswith('.json')]
    
    to_transcribe = []
    for mp3_file in mp3_files:
        base_name = os.path.splitext(mp3_file)[0]
        if base_name not in existing_transcripts:
            to_transcribe.append(mp3_file)
    
    logger.info(f"Found {len(to_transcribe)} files to transcribe out of {len(mp3_files)} total MP3 files")
    
    if not to_transcribe:
        logger.info("No files to transcribe")
        return True
    
    # Check if whisper is installed
    try:
        import whisper
    except ImportError:
        logger.error("Whisper not installed. Please install it with: pip install openai-whisper")
        return False
    
    # Load the model
    try:
        logger.info(f"Loading Whisper model: {model}")
        whisper_model = whisper.load_model(model, device=device)
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        return False
    
    # Transcribe each file
    success_count = 0
    for mp3_file in tqdm(to_transcribe, desc="Transcribing"):
        mp3_path = os.path.join(mp3_dir, mp3_file)
        base_name = os.path.splitext(mp3_file)[0]
        transcript_path = os.path.join(transcript_dir, f"{base_name}.json")
        
        try:
            logger.info(f"Transcribing {mp3_file}...")
            result = whisper_model.transcribe(mp3_path)
            
            with open(transcript_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved transcript to {transcript_path}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error transcribing {mp3_file}: {e}")
    
    logger.info(f"Transcription complete. Successfully transcribed {success_count}/{len(to_transcribe)} files.")
    return success_count > 0

def process_transcripts(transcript_dir, corpus_dir, songwriting_only=False):
    """Process transcripts and integrate into the corpus."""
    logger.info(f"Processing transcripts from {transcript_dir} to {corpus_dir}")
    
    cmd = [
        "python", 
        "songwriting_corpus_integrator.py", 
        "--input-dir", transcript_dir,
        "--output-dir", corpus_dir
    ]
    
    if songwriting_only:
        cmd.append("--songwriting-only")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing transcripts: {e}")
        logger.error(e.stderr)
        return False

def check_progress(mp3_dir, transcript_dir, corpus_dir):
    """Check the progress of the transcription and processing pipeline."""
    mp3_files = [f for f in os.listdir(mp3_dir) if f.lower().endswith('.mp3')]
    transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.json')]
    corpus_files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    
    total_mp3s = len(mp3_files)
    completed_transcripts = len(transcript_files)
    integrated_corpus = len(corpus_files)
    
    if total_mp3s == 0:
        logger.warning("No MP3 files found")
        return
    
    transcript_progress = (completed_transcripts / total_mp3s) * 100
    corpus_progress = (integrated_corpus / total_mp3s) * 100
    
    logger.info(f"Progress Report:")
    logger.info(f"  Total MP3 files: {total_mp3s}")
    logger.info(f"  Completed transcripts: {completed_transcripts} ({transcript_progress:.1f}%)")
    logger.info(f"  Integrated corpus files: {integrated_corpus} ({corpus_progress:.1f}%)")
    
    # Check for songwriting content if we have processed files
    if os.path.exists("songwriting_corpus_report.txt"):
        with open("songwriting_corpus_report.txt", 'r') as f:
            report = f.read()
            logger.info("\nSongwriting Content Report:")
            for line in report.split('\n'):
                if "Transcripts with songwriting content" in line:
                    logger.info(f"  {line}")
                    break

def run_full_workflow(rss_url, base_dir, limit=None, songwriting_only=False):
    """Run the complete workflow from download to corpus integration."""
    # Setup directories
    dirs = setup_directories(base_dir)
    
    # Step 1: Download podcasts
    logger.info("Step 1: Downloading podcasts")
    if not download_podcasts(rss_url, dirs['mp3'], limit=limit):
        logger.error("Failed to download podcasts. Stopping workflow.")
        return False
    
    # Step 2: Transcribe audio
    logger.info("Step 2: Transcribing audio")
    if not transcribe_audio(dirs['mp3'], dirs['transcripts']):
        logger.warning("Transcription step had issues. Continuing with existing transcripts.")
    
    # Step 3: Process transcripts
    logger.info("Step 3: Processing transcripts and integrating into corpus")
    if not process_transcripts(dirs['transcripts'], dirs['corpus'], songwriting_only):
        logger.error("Failed to process transcripts. Stopping workflow.")
        return False
    
    # Check progress
    check_progress(dirs['mp3'], dirs['transcripts'], dirs['corpus'])
    
    logger.info("Full workflow completed successfully!")
    return True

def main():
    """Main function to run the enhanced workflow."""
    parser = argparse.ArgumentParser(description="Enhanced Podcast Transcription Workflow")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["download", "transcribe", "process", "all", "check"],
                        help="Workflow mode to run")
    parser.add_argument("--rss-url", type=str,
                        help="RSS feed URL for podcast downloads")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base directory for all workflow files")
    parser.add_argument("--mp3-dir", type=str,
                        help="Directory containing MP3 files")
    parser.add_argument("--transcript-dir", type=str,
                        help="Directory containing transcript files")
    parser.add_argument("--corpus-dir", type=str,
                        help="Output directory for corpus files")
    parser.add_argument("--limit", type=int,
                        help="Limit the number of episodes to download")
    parser.add_argument("--songwriting-only", action="store_true",
                        help="Only include transcripts with songwriting content")
    args = parser.parse_args()
    
    # Setup directories if base_dir is provided
    if args.base_dir != ".":
        dirs = setup_directories(args.base_dir)
        mp3_dir = dirs['mp3']
        transcript_dir = dirs['transcripts']
        corpus_dir = dirs['corpus']
    else:
        mp3_dir = args.mp3_dir
        transcript_dir = args.transcript_dir
        corpus_dir = args.corpus_dir
    
    # Run the selected mode
    if args.mode == "download":
        if not args.rss_url:
            logger.error("RSS URL is required for download mode")
            return 1
        if not mp3_dir:
            logger.error("MP3 directory is required for download mode")
            return 1
        
        download_podcasts(args.rss_url, mp3_dir, limit=args.limit)
    
    elif args.mode == "transcribe":
        if not mp3_dir:
            logger.error("MP3 directory is required for transcribe mode")
            return 1
        if not transcript_dir:
            logger.error("Transcript directory is required for transcribe mode")
            return 1
        
        transcribe_audio(mp3_dir, transcript_dir)
    
    elif args.mode == "process":
        if not transcript_dir:
            logger.error("Transcript directory is required for process mode")
            return 1
        if not corpus_dir:
            logger.error("Corpus directory is required for process mode")
            return 1
        
        process_transcripts(transcript_dir, corpus_dir, args.songwriting_only)
    
    elif args.mode == "all":
        if not args.rss_url:
            logger.error("RSS URL is required for all mode")
            return 1
        if not args.base_dir:
            logger.error("Base directory is required for all mode")
            return 1
        
        run_full_workflow(args.rss_url, args.base_dir, args.limit, args.songwriting_only)
    
    elif args.mode == "check":
        if not mp3_dir:
            logger.error("MP3 directory is required for check mode")
            return 1
        if not transcript_dir:
            logger.error("Transcript directory is required for check mode")
            return 1
        if not corpus_dir:
            logger.error("Corpus directory is required for check mode")
            return 1
        
        check_progress(mp3_dir, transcript_dir, corpus_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
