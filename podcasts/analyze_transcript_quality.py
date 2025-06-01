#!/usr/bin/env python3
"""
Analyze Transcript Quality

This script analyzes the quality of Whisper transcripts by:
1. Calculating transcript length and word count statistics
2. Identifying potential transcription issues (repeated phrases, truncations)
3. Generating a quality report for review

Usage:
  python analyze_transcript_quality.py --transcript-dir /path/to/text/transcripts
"""

import os
import sys
import re
import argparse
import logging
import statistics
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_quality.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transcript_quality")

def count_words(text):
    """Count words in text."""
    return len(text.split())

def find_repeated_phrases(text, min_length=4, min_repetitions=3):
    """Find repeated phrases in text."""
    words = text.split()
    phrases = []
    
    # Check phrases of different lengths
    for phrase_length in range(min_length, 8):
        phrase_counts = Counter()
        
        # Generate all phrases of current length
        for i in range(len(words) - phrase_length + 1):
            phrase = " ".join(words[i:i+phrase_length])
            phrase_counts[phrase] += 1
        
        # Find repeated phrases
        for phrase, count in phrase_counts.items():
            if count >= min_repetitions:
                phrases.append((phrase, count))
    
    return phrases

def check_for_truncation(text):
    """Check if transcript appears to be truncated."""
    # Check if text ends mid-sentence
    last_sentence = text.strip().split('.')[-1].strip()
    if last_sentence and not last_sentence.endswith(('.', '!', '?', '"')):
        return True
    
    # Check for abrupt endings
    endings = ['to be continued', 'we will continue', 'next time', 'part']
    for ending in endings:
        if ending in text.lower()[-50:]:
            return False
    
    return False

def analyze_transcript(file_path):
    """Analyze a single transcript file."""
    try:
        with open(file_path, 'r') as f:
            text = f.read()
        
        # Basic statistics
        word_count = count_words(text)
        char_count = len(text)
        
        # Quality checks
        repeated_phrases = find_repeated_phrases(text)
        is_truncated = check_for_truncation(text)
        
        return {
            'file': os.path.basename(file_path),
            'word_count': word_count,
            'char_count': char_count,
            'repeated_phrases': repeated_phrases,
            'is_truncated': is_truncated
        }
    
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return None

def analyze_transcripts(transcript_dir):
    """Analyze all transcript files in directory."""
    transcript_dir = os.path.abspath(transcript_dir)
    
    # Find all text files
    text_files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]
    logger.info(f"Found {len(text_files)} transcript files to analyze")
    
    # Analyze each file
    results = []
    for text_file in tqdm(text_files, desc="Analyzing"):
        file_path = os.path.join(transcript_dir, text_file)
        result = analyze_transcript(file_path)
        if result:
            results.append(result)
    
    return results

def generate_report(results):
    """Generate a quality report from analysis results."""
    if not results:
        return "No valid transcripts found for analysis."
    
    # Calculate statistics
    word_counts = [r['word_count'] for r in results]
    char_counts = [r['char_count'] for r in results]
    
    avg_word_count = statistics.mean(word_counts)
    median_word_count = statistics.median(word_counts)
    min_word_count = min(word_counts)
    max_word_count = max(word_counts)
    
    # Find potential issues
    files_with_repetitions = [r for r in results if r['repeated_phrases']]
    files_with_truncation = [r for r in results if r['is_truncated']]
    
    # Generate report
    report = "Transcript Quality Analysis Report\n"
    report += "================================\n\n"
    
    report += f"Total transcripts analyzed: {len(results)}\n\n"
    
    report += "Word Count Statistics:\n"
    report += f"  Average: {avg_word_count:.1f} words\n"
    report += f"  Median: {median_word_count} words\n"
    report += f"  Range: {min_word_count} - {max_word_count} words\n\n"
    
    report += "Potential Issues:\n"
    report += f"  Files with repeated phrases: {len(files_with_repetitions)} ({len(files_with_repetitions)/len(results)*100:.1f}%)\n"
    report += f"  Files with potential truncation: {len(files_with_truncation)} ({len(files_with_truncation)/len(results)*100:.1f}%)\n\n"
    
    # List top 5 shortest transcripts (might indicate issues)
    report += "Shortest Transcripts:\n"
    shortest = sorted(results, key=lambda x: x['word_count'])[:5]
    for r in shortest:
        report += f"  {r['file']}: {r['word_count']} words\n"
    
    report += "\nDetailed Issues:\n"
    
    # List files with most repetitions
    if files_with_repetitions:
        report += "\nFiles with significant phrase repetitions:\n"
        worst_repetitions = sorted(files_with_repetitions, 
                                  key=lambda x: sum(count for _, count in x['repeated_phrases']), 
                                  reverse=True)[:5]
        for r in worst_repetitions:
            report += f"  {r['file']}:\n"
            for phrase, count in sorted(r['repeated_phrases'], key=lambda x: x[1], reverse=True)[:3]:
                report += f"    '{phrase}' repeated {count} times\n"
    
    return report

def main():
    """Main function to analyze transcript quality."""
    parser = argparse.ArgumentParser(description="Analyze Whisper transcript quality")
    parser.add_argument("--transcript-dir", type=str, required=True, 
                        help="Directory containing transcript text files")
    parser.add_argument("--output-file", type=str, default="transcript_quality_report.txt",
                        help="Output file for quality report")
    args = parser.parse_args()
    
    # Analyze transcripts
    results = analyze_transcripts(args.transcript_dir)
    
    # Generate report
    report = generate_report(results)
    
    # Save report
    with open(args.output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Quality report saved to {args.output_file}")
    print("\nQuality Report Summary:")
    print("======================")
    print("\n".join(report.split("\n")[:15]) + "\n...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
