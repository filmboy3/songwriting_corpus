#!/usr/bin/env python3
"""
Songwriting Corpus Integrator

This script processes cleaned transcripts, extracts songwriting-specific content,
and integrates it into the songwriting corpus with appropriate formatting and metadata.

It builds on the existing transcript workflow but adds specialized handling for
songwriting content, including topic classification and metadata extraction.
"""

import os
import sys
import json
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import our improved transcript cleaner
from improved_transcript_cleaner import clean_transcript, extract_songwriting_content

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("songwriting_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("songwriting_integrator")

# Define songwriting topics for classification
SONGWRITING_TOPICS = {
    'process': [
        r'(?i)writing process', 
        r'(?i)how (I|you|they|we) (wrote|write|approach)', 
        r'(?i)when (I|you|they|we) (wrote|write|start)',
        r'(?i)start(ing|ed) (with|from)',
        r'(?i)(my|your|their|our) process'
    ],
    'inspiration': [
        r'(?i)inspir(e|ed|ation|ing)', 
        r'(?i)idea (came|started|began)',
        r'(?i)came (up with|to me)',
        r'(?i)influence',
        r'(?i)based on',
        r'(?i)story behind'
    ],
    'collaboration': [
        r'(?i)collaborat(e|ed|ion|ing)',
        r'(?i)co-writ(e|ing|ten)',
        r'(?i)writing session',
        r'(?i)writing partner',
        r'(?i)wrote (it )?(with|together)',
        r'(?i)(in the|a) room (with|together)'
    ],
    'lyrics': [
        r'(?i)lyric(s|al)',
        r'(?i)words (to|of|in) the song',
        r'(?i)meaning (of|behind)',
        r'(?i)tell(ing)? a story',
        r'(?i)narrative',
        r'(?i)message',
        r'(?i)theme'
    ],
    'melody': [
        r'(?i)melod(y|ic|ies)',
        r'(?i)tune',
        r'(?i)(musical|melodic) hook',
        r'(?i)catchy',
        r'(?i)chorus',
        r'(?i)verse',
        r'(?i)bridge',
        r'(?i)pre-chorus'
    ],
    'production': [
        r'(?i)produc(e|ed|ing|tion)',
        r'(?i)record(ed|ing)',
        r'(?i)(in the|at the) studio',
        r'(?i)track',
        r'(?i)mix(ed|ing)',
        r'(?i)arrang(e|ed|ing|ement)',
        r'(?i)instrument(s|ation)'
    ],
    'structure': [
        r'(?i)structur(e|al|ing)',
        r'(?i)arrangement',
        r'(?i)form',
        r'(?i)part(s)?',
        r'(?i)section(s)?',
        r'(?i)intro',
        r'(?i)outro',
        r'(?i)middle (eight|section)'
    ],
    'chords': [
        r'(?i)chord(s)?',
        r'(?i)progression',
        r'(?i)key (of|change)',
        r'(?i)major',
        r'(?i)minor',
        r'(?i)harmon(y|ic|ies)',
        r'(?i)scale'
    ]
}

def extract_podcast_metadata(filename):
    """Extract podcast metadata from filename and return as dict."""
    # Assume filename format: PodcastName_EpisodeTitle.json
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Try to extract podcast name and episode
    parts = base_name.split('_', 1)
    
    metadata = {
        'podcast_name': parts[0].replace('_', ' ') if parts else base_name,
        'episode_title': parts[1].replace('_', ' ') if len(parts) > 1 else '',
        'filename': base_name
    }
    
    return metadata

def classify_songwriting_topics(text):
    """Classify text into songwriting topics based on keyword patterns."""
    topics = {}
    
    for topic, patterns in SONGWRITING_TOPICS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                topics[topic] = True
                break
    
    return list(topics.keys())

def extract_guests(text):
    """Extract potential guest names from transcript text."""
    guest_patterns = [
        r"(?i)today'?s? guest is ([^.\n]+)",
        r"(?i)joining me today is ([^.\n]+)",
        r"(?i)I'?m joined by ([^.\n]+)",
        r"(?i)welcome ([^.\n]+) to the podcast",
        r"(?i)my guest today is ([^.\n]+)",
        r"(?i)special guest ([^.\n]+)",
        r"(?i)with me today is ([^.\n]+)"
    ]
    
    guests = []
    for pattern in guest_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Clean up guest names
            guest = match.strip()
            # Remove common phrases that might be captured
            guest = re.sub(r'(?i)to the show|to the podcast|today|this week', '', guest).strip()
            # Remove trailing punctuation
            guest = re.sub(r'[.,;:]$', '', guest).strip()
            
            if guest and len(guest) > 3 and guest.lower() not in ('me', 'you', 'today'):
                guests.append(guest)
    
    return guests

def format_for_corpus(text, metadata, songwriting_sections):
    """Format transcript for the songwriting corpus with appropriate tokens."""
    # Extract topics from songwriting sections
    all_topics = set()
    for section in songwriting_sections:
        topics = classify_songwriting_topics(section)
        all_topics.update(topics)
    
    # Extract guests
    guests = extract_guests(text)
    
    # Format with tokens
    formatted_text = f"<PODCAST>\n"
    
    # Add metadata
    formatted_text += f"<TITLE>{metadata['podcast_name']}</TITLE>\n"
    
    if metadata['episode_title']:
        formatted_text += f"<EPISODE>{metadata['episode_title']}</EPISODE>\n"
    
    if guests:
        formatted_text += f"<GUESTS>{', '.join(guests)}</GUESTS>\n"
    
    if all_topics:
        formatted_text += f"<TOPICS>{', '.join(sorted(all_topics))}</TOPICS>\n"
    
    # Add songwriting sections with special tags
    if songwriting_sections:
        formatted_text += "\n<SONGWRITING_CONTENT>\n"
        for i, section in enumerate(songwriting_sections):
            topics = classify_songwriting_topics(section)
            if topics:
                formatted_text += f"<SECTION topics=\"{','.join(topics)}\">\n"
            else:
                formatted_text += f"<SECTION>\n"
            
            formatted_text += f"{section}\n</SECTION>\n\n"
        formatted_text += "</SONGWRITING_CONTENT>\n"
    
    # Add full transcript
    formatted_text += f"\n<TRANSCRIPT>\n{text}\n</TRANSCRIPT>\n"
    formatted_text += f"</PODCAST>"
    
    return formatted_text

def process_transcript(input_path, output_path):
    """Process a transcript and integrate it into the songwriting corpus."""
    try:
        # Check if input is JSON or text
        if input_path.endswith('.json'):
            try:
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                # Check if this is already processed by our improved cleaner
                if 'full_text' in data and 'songwriting_sections' in data:
                    text = data['full_text']
                    songwriting_sections = data['songwriting_sections']
                else:
                    # Handle Whisper JSON format
                    text = ""
                    if "segments" in data:
                        for segment in data["segments"]:
                            if "text" in segment:
                                text += segment["text"] + " "
                    elif "text" in data:
                        text = data["text"]
                    
                    # Clean the transcript
                    text = clean_transcript(text.strip())
                    
                    # Extract songwriting content
                    songwriting_sections = extract_songwriting_content(text)
            except json.JSONDecodeError:
                # Not valid JSON, treat as text
                with open(input_path, 'r') as f:
                    text = f.read()
                
                # Clean the transcript
                text = clean_transcript(text)
                
                # Extract songwriting content
                songwriting_sections = extract_songwriting_content(text)
        else:
            # Regular text file
            with open(input_path, 'r') as f:
                text = f.read()
            
            # Clean the transcript
            text = clean_transcript(text)
            
            # Extract songwriting content
            songwriting_sections = extract_songwriting_content(text)
        
        # Extract metadata from filename
        metadata = extract_podcast_metadata(input_path)
        
        # Format for corpus
        formatted_text = format_for_corpus(text, metadata, songwriting_sections)
        
        # Write to output file
        with open(output_path, 'w') as f:
            f.write(formatted_text)
        
        return True, len(songwriting_sections) > 0
    
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False, False

def main():
    """Main function to process transcripts and integrate into corpus."""
    parser = argparse.ArgumentParser(description="Process transcripts for songwriting corpus")
    parser.add_argument("--input-dir", type=str, required=True, 
                        help="Directory containing transcript files (JSON or text)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed corpus files")
    parser.add_argument("--songwriting-only", action="store_true",
                        help="Only include transcripts with songwriting content")
    parser.add_argument("--file", type=str, 
                        help="Process a single file instead of the entire directory")
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process a single file if specified
    if args.file:
        input_path = os.path.join(input_dir, args.file)
        output_path = os.path.join(output_dir, os.path.splitext(args.file)[0] + ".txt")
        
        success, has_songwriting = process_transcript(input_path, output_path)
        
        if success:
            logger.info(f"Successfully processed {args.file}")
            if has_songwriting:
                logger.info(f"Found songwriting content in {args.file}")
            elif args.songwriting_only:
                # Remove file if no songwriting content and we only want songwriting content
                os.remove(output_path)
                logger.info(f"Removed {args.file} (no songwriting content)")
        else:
            logger.error(f"Failed to process {args.file}")
        
        return 0
    
    # Find all transcript files
    input_files = []
    for ext in ['.json', '.txt']:
        input_files.extend([f for f in os.listdir(input_dir) if f.endswith(ext)])
    
    logger.info(f"Found {len(input_files)} transcript files to process")
    
    # Process each file
    success_count = 0
    songwriting_count = 0
    
    for input_file in tqdm(input_files, desc="Processing"):
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, os.path.splitext(input_file)[0] + ".txt")
        
        success, has_songwriting = process_transcript(input_path, output_path)
        
        if success:
            success_count += 1
            if has_songwriting:
                songwriting_count += 1
            elif args.songwriting_only:
                # Remove file if no songwriting content and we only want songwriting content
                os.remove(output_path)
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(input_files)} transcripts.")
    logger.info(f"Found {songwriting_count} transcripts with songwriting content.")
    
    if args.songwriting_only:
        logger.info(f"Kept {songwriting_count} transcripts with songwriting content.")
    
    # Generate summary report
    report = "Songwriting Corpus Integration Report\n"
    report += "===================================\n\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Input directory: {input_dir}\n"
    report += f"Output directory: {output_dir}\n\n"
    report += f"Total transcripts processed: {len(input_files)}\n"
    report += f"Successfully processed: {success_count}\n"
    report += f"Transcripts with songwriting content: {songwriting_count} ({songwriting_count/len(input_files)*100:.1f}%)\n"
    
    if args.songwriting_only:
        report += f"\nOnly transcripts with songwriting content were kept in the corpus.\n"
    
    report_path = "songwriting_corpus_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
