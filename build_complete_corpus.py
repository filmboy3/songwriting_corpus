#!/usr/bin/env python3
"""
Master script to build a complete songwriting corpus with integrated chord progressions.
This script orchestrates the entire corpus building process by running all necessary steps:

1. Extract lyrics from chord files
2. Import chord and lyrics data from CSV
3. Analyze artist distribution
4. Run batch chord analysis
5. Spot check random files for quality
6. Generate corpus statistics and report

Usage:
    python build_complete_corpus.py [--prioritized-only] [--skip-csv-import] [--skip-chord-analysis]
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("corpus_build.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("corpus_builder")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_DIR = os.path.join(BASE_DIR, "combined")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
CSV_FILE = os.path.join(BASE_DIR, "chords_and_lyrics.csv")

def print_header(message):
    """Print a formatted header message."""
    header = f"\n{'=' * 80}\n  {message}\n{'=' * 80}\n"
    logger.info(header)
    return header

def run_script(script_name, args=None, blocking=True, description=None):
    """Run a Python script with the given arguments."""
    if description:
        logger.info(f"Running: {description}")
    
    cmd = [sys.executable, os.path.join(BASE_DIR, script_name)]
    if args:
        cmd.extend(args)
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        if blocking:
            result = subprocess.run(cmd, check=True, cwd=BASE_DIR, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
            logger.info(f"Command completed with exit code: {result.returncode}")
            return result.returncode == 0
        else:
            subprocess.Popen(cmd, cwd=BASE_DIR)
            logger.info("Command started in background")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def ensure_directories():
    """Ensure all necessary directories exist."""
    dirs = [
        os.path.join(COMBINED_DIR, "lyrics"),
        os.path.join(COMBINED_DIR, "chords"),
        os.path.join(COMBINED_DIR, "analysis"),
        REPORT_DIR
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def extract_lyrics_from_chords(prioritized_only=True):
    """Extract lyrics from chord files."""
    print_header("EXTRACTING LYRICS FROM CHORD FILES")
    
    args = []
    if prioritized_only:
        args.append("--prioritized-only")
    
    return run_script("extract_lyrics_from_chords.py", args, 
                     description="Extracting lyrics from chord files")

def import_csv_data(prioritized_only=True):
    """Import chord and lyrics data from CSV."""
    print_header("IMPORTING CSV CHORD AND LYRICS DATA")
    
    if not os.path.exists(CSV_FILE):
        logger.error(f"CSV file not found: {CSV_FILE}")
        logger.info("Skipping CSV import")
        return False
    
    args = []
    if prioritized_only:
        args.append("--prioritized-only")
    
    return run_script("import_csv_chords_lyrics.py", args,
                     description="Importing chord and lyrics data from CSV")

def analyze_artist_distribution():
    """Analyze artist distribution in the corpus."""
    print_header("ANALYZING ARTIST DISTRIBUTION")
    
    # Run the count_songs_by_artist.py script
    success = run_script("count_songs_by_artist.py", ["--output", os.path.join(REPORT_DIR, "artist_distribution.txt")],
                        description="Counting songs by artist across the corpus")
    
    # Also analyze CSV artist distribution
    if os.path.exists(CSV_FILE):
        run_script("analyze_csv_artists.py", ["--output", os.path.join(REPORT_DIR, "csv_artist_distribution.txt")],
                  description="Analyzing artist distribution in CSV data")
    
    return success

def run_batch_chord_analysis(prioritized_only=True):
    """Run batch chord progression analysis."""
    print_header("RUNNING BATCH CHORD PROGRESSION ANALYSIS")
    
    args = ["--parallel", "4"]
    if prioritized_only:
        args.append("--prioritized-only")
    
    return run_script("batch_chord_analysis.py", args,
                     description="Running batch chord progression analysis")

def spot_check_files():
    """Spot check random files for quality."""
    print_header("SPOT CHECKING RANDOM FILES")
    
    output_file = os.path.join(REPORT_DIR, "spot_check_results.txt")
    return run_script("spot_check_files.py", ["--output", output_file],
                     description=f"Spot checking random files (results in {output_file})")

def generate_corpus_statistics():
    """Generate corpus statistics."""
    print_header("GENERATING CORPUS STATISTICS")
    
    # Count total files
    lyrics_count = len(list(Path(os.path.join(COMBINED_DIR, "lyrics")).glob("**/*.txt")))
    chords_count = len(list(Path(os.path.join(COMBINED_DIR, "chords")).glob("**/*.txt")))
    analysis_count = len(list(Path(os.path.join(COMBINED_DIR, "analysis")).glob("**/*.json")))
    
    # Generate report
    report = f"""
SONGWRITING CORPUS STATISTICS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total lyrics files: {lyrics_count}
Total chord files: {chords_count}
Total chord analysis files: {analysis_count}
Chord-to-lyrics ratio: {chords_count/lyrics_count:.2f if lyrics_count > 0 else 'N/A'}
Analysis coverage: {analysis_count/chords_count:.2f if chords_count > 0 else 'N/A'}
    """
    
    # Save report
    report_file = os.path.join(REPORT_DIR, "corpus_statistics.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Corpus statistics saved to {report_file}")
    logger.info(f"Lyrics files: {lyrics_count}, Chord files: {chords_count}, Analysis files: {analysis_count}")
    
    return True

def main():
    """Main function to build the complete corpus."""
    parser = argparse.ArgumentParser(description="Build a complete songwriting corpus with integrated chord progressions")
    parser.add_argument("--prioritized-only", action="store_true", help="Process only prioritized artists")
    parser.add_argument("--skip-csv-import", action="store_true", help="Skip importing data from CSV")
    parser.add_argument("--skip-chord-analysis", action="store_true", help="Skip batch chord analysis")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print_header("STARTING COMPLETE CORPUS BUILD")
    logger.info(f"Build started with options: {args}")
    
    # Ensure all directories exist
    ensure_directories()
    
    # Extract lyrics from chord files
    extract_lyrics_from_chords(args.prioritized_only)
    
    # Import CSV data if not skipped
    if not args.skip_csv_import:
        import_csv_data(args.prioritized_only)
    else:
        logger.info("Skipping CSV import as requested")
    
    # Analyze artist distribution
    analyze_artist_distribution()
    
    # Run batch chord analysis if not skipped
    if not args.skip_chord_analysis:
        run_batch_chord_analysis(args.prioritized_only)
    else:
        logger.info("Skipping batch chord analysis as requested")
    
    # Spot check files
    spot_check_files()
    
    # Generate corpus statistics
    generate_corpus_statistics()
    
    # Calculate total time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_header("CORPUS BUILD COMPLETE")
    logger.info(f"Total build time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Check the reports directory for detailed statistics and spot check results")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
