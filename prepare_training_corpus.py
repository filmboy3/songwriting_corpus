#!/usr/bin/env python3
"""
Prepare Songwriting Corpus for Training

This script prepares the songwriting corpus for model training by:
1. Collecting and normalizing content from all sources (lyrics, podcasts, Instagram)
2. Applying basic cleaning and formatting
3. Creating training-ready files in JSONL format with appropriate metadata
4. Generating a combined corpus file for training

No complex tokenization is performed - just basic text preparation.
"""

import os
import json
import re
import glob
import random
from pathlib import Path
from tqdm import tqdm
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
LYRICS_DIR = BASE_DIR / "data" / "lyrics"
PODCASTS_DIR = BASE_DIR / "podcasts" / "transcripts"

# Additional paths to check for data
ALT_LYRICS_DIR = BASE_DIR / "lyrics"
INSTAGRAM_DIR = BASE_DIR / "instagram" / "data"
ALT_INSTAGRAM_DIR = Path("/Users/jonathanschwartz/CascadeProjects/instagramPull/instagram_data")

OUTPUT_DIR = BASE_DIR / "training_ready"

# Special tokens for formatting (simple approach)
SONG_START = "<|song|>"
SONG_END = "</|song|>"
PODCAST_START = "<|podcast|>"
PODCAST_END = "</|podcast|>"
INSTAGRAM_START = "<|instagram|>"
INSTAGRAM_END = "</|instagram|>"
VERSE_START = "<|verse|>"
VERSE_END = "</|verse|>"
CHORUS_START = "<|chorus|>"
CHORUS_END = "</|chorus|>"
BRIDGE_START = "<|bridge|>"
BRIDGE_END = "</|bridge|>"
SECTION_START = "<|section|>"
SECTION_END = "</|section|>"

def clean_text(text):
    """Basic text cleaning function."""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    return text.strip()

def identify_song_sections(lyrics):
    """
    Simple function to identify common song sections like verse, chorus, bridge.
    Returns the lyrics with section markers.
    """
    # This is a simplified approach - not as complex as full tokenization
    lines = lyrics.split('\n')
    formatted_lines = []
    in_section = False
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_section:
                if current_section == "verse":
                    formatted_lines.append(VERSE_END)
                elif current_section == "chorus":
                    formatted_lines.append(CHORUS_END)
                elif current_section == "bridge":
                    formatted_lines.append(BRIDGE_END)
                else:
                    formatted_lines.append(SECTION_END)
                in_section = False
            formatted_lines.append("")
            continue
            
        # Check for section headers
        lower_line = line.lower()
        if re.match(r'^verse|^v\d', lower_line):
            if in_section:
                if current_section == "verse":
                    formatted_lines.append(VERSE_END)
                elif current_section == "chorus":
                    formatted_lines.append(CHORUS_END)
                elif current_section == "bridge":
                    formatted_lines.append(BRIDGE_END)
                else:
                    formatted_lines.append(SECTION_END)
            formatted_lines.append(VERSE_START)
            current_section = "verse"
            in_section = True
            continue
        elif re.match(r'^chorus|^c\d|^refrain', lower_line):
            if in_section:
                if current_section == "verse":
                    formatted_lines.append(VERSE_END)
                elif current_section == "chorus":
                    formatted_lines.append(CHORUS_END)
                elif current_section == "bridge":
                    formatted_lines.append(BRIDGE_END)
                else:
                    formatted_lines.append(SECTION_END)
            formatted_lines.append(CHORUS_START)
            current_section = "chorus"
            in_section = True
            continue
        elif re.match(r'^bridge|^b\d', lower_line):
            if in_section:
                if current_section == "verse":
                    formatted_lines.append(VERSE_END)
                elif current_section == "chorus":
                    formatted_lines.append(CHORUS_END)
                elif current_section == "bridge":
                    formatted_lines.append(BRIDGE_END)
                else:
                    formatted_lines.append(SECTION_END)
            formatted_lines.append(BRIDGE_START)
            current_section = "bridge"
            in_section = True
            continue
        elif re.match(r'^(pre-chorus|intro|outro|interlude|solo|hook)', lower_line):
            if in_section:
                if current_section == "verse":
                    formatted_lines.append(VERSE_END)
                elif current_section == "chorus":
                    formatted_lines.append(CHORUS_END)
                elif current_section == "bridge":
                    formatted_lines.append(BRIDGE_END)
                else:
                    formatted_lines.append(SECTION_END)
            formatted_lines.append(f"{SECTION_START} {line}")
            current_section = "other"
            in_section = True
            continue
            
        formatted_lines.append(line)
    
    # Close any open section at the end
    if in_section:
        if current_section == "verse":
            formatted_lines.append(VERSE_END)
        elif current_section == "chorus":
            formatted_lines.append(CHORUS_END)
        elif current_section == "bridge":
            formatted_lines.append(BRIDGE_END)
        else:
            formatted_lines.append(SECTION_END)
    
    return '\n'.join(formatted_lines)

def process_lyrics():
    """Process lyrics files and prepare them for training."""
    logger.info("Processing lyrics files...")
    output_path = OUTPUT_DIR / "lyrics" / "lyrics_corpus.jsonl"
    
    # Find all lyrics files from multiple possible locations
    lyrics_files = []
    for lyrics_path in [LYRICS_DIR, ALT_LYRICS_DIR]:
        if lyrics_path.exists():
            for ext in ['*.txt', '*.json']:
                lyrics_files.extend(glob.glob(str(lyrics_path / "**" / ext), recursive=True))
    
    logger.info(f"Found {len(lyrics_files)} lyrics files")
    
    processed_count = 0
    with open(output_path, 'w') as outfile:
        for file_path in tqdm(lyrics_files, desc="Processing lyrics"):
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'lyrics' in data:
                            lyrics = data['lyrics']
                            artist = data.get('artist', 'Unknown')
                            title = data.get('title', 'Unknown')
                        else:
                            continue
                else:  # Text file
                    with open(file_path, 'r') as f:
                        lyrics = f.read()
                        # Try to extract artist and title from filename
                        filename = os.path.basename(file_path)
                        parts = filename.replace('.txt', '').split(' - ', 1)
                        if len(parts) > 1:
                            artist, title = parts
                        else:
                            artist = 'Unknown'
                            title = parts[0]
                
                # Clean and format the lyrics
                lyrics = clean_text(lyrics)
                formatted_lyrics = identify_song_sections(lyrics)
                
                # Create a training example
                example = {
                    "text": f"{SONG_START}\nTitle: {title}\nArtist: {artist}\n\n{formatted_lyrics}\n{SONG_END}",
                    "metadata": {
                        "source": "lyrics",
                        "artist": artist,
                        "title": title,
                        "file": os.path.basename(file_path)
                    }
                }
                
                outfile.write(json.dumps(example) + '\n')
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Processed {processed_count} lyrics files")
    return processed_count

def process_podcasts():
    """Process podcast transcripts and prepare them for training."""
    logger.info("Processing podcast transcripts...")
    output_path = OUTPUT_DIR / "podcasts" / "podcast_corpus.jsonl"
    
    # Find all transcript files
    transcript_files = []
    for ext in ['*.txt', '*.json']:
        transcript_files.extend(glob.glob(str(PODCASTS_DIR / "**" / ext), recursive=True))
    
    logger.info(f"Found {len(transcript_files)} podcast transcript files")
    
    processed_count = 0
    with open(output_path, 'w') as outfile:
        for file_path in tqdm(transcript_files, desc="Processing podcasts"):
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'text' in data:
                            transcript = data['text']
                        else:
                            continue
                else:  # Text file
                    with open(file_path, 'r') as f:
                        transcript = f.read()
                
                # Extract title from filename
                filename = os.path.basename(file_path)
                title = filename.replace('.txt', '').replace('.json', '')
                
                # Clean the transcript
                transcript = clean_text(transcript)
                
                # Create a training example
                example = {
                    "text": f"{PODCAST_START}\nTitle: {title}\n\n{transcript}\n{PODCAST_END}",
                    "metadata": {
                        "source": "podcast",
                        "title": title,
                        "file": os.path.basename(file_path)
                    }
                }
                
                outfile.write(json.dumps(example) + '\n')
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Processed {processed_count} podcast files")
    return processed_count

def process_instagram():
    """Process Instagram posts and prepare them for training."""
    logger.info("Processing Instagram posts...")
    output_path = OUTPUT_DIR / "instagram" / "instagram_corpus.jsonl"
    
    # Find all Instagram data files from multiple possible locations
    instagram_files = []
    for instagram_path in [INSTAGRAM_DIR, ALT_INSTAGRAM_DIR]:
        if instagram_path.exists():
            instagram_files.extend(glob.glob(str(instagram_path / "*_data.json")))
    
    logger.info(f"Found {len(instagram_files)} Instagram data files")
    
    processed_count = 0
    with open(output_path, 'w') as outfile:
        for file_path in tqdm(instagram_files, desc="Processing Instagram"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract username from filename
                filename = os.path.basename(file_path)
                username = filename.replace('_data.json', '')
                
                # Process each post - handle both list and dictionary formats
                if isinstance(data, list):
                    # List format (as seen in the actual data)
                    for post_data in data:
                        if not isinstance(post_data, dict):
                            continue
                        
                        post_id = post_data.get('shortcode', '')
                        caption = post_data.get('caption', '')
                        date = post_data.get('date_utc', '')
                        
                        # Extract categories/topics from the data
                        categories = []
                        if 'hashtags' in post_data and post_data['hashtags']:
                            categories.extend(post_data['hashtags'])
                        if 'topics' in post_data and post_data['topics']:
                            categories.extend(post_data['topics'])
                        if 'is_songwriting' in post_data and post_data['is_songwriting']:
                            categories.append('songwriting')
                        
                        ocr_text = post_data.get('ocr_text', '')
                        
                        # Combine caption and OCR text
                        content = caption
                        if ocr_text:
                            content += f"\n\nImage Text:\n{ocr_text}"
                        
                        # Clean the content
                        content = clean_text(content)
                        
                        # Skip if content is too short
                        if len(content) < 50:
                            continue
                        
                        # Create a training example
                        example = {
                            "text": f"{INSTAGRAM_START}\nUsername: {username}\nDate: {date}\nCategories: {', '.join(categories)}\n\n{content}\n{INSTAGRAM_END}",
                            "metadata": {
                                "source": "instagram",
                                "username": username,
                                "post_id": post_id,
                                "date": date,
                                "categories": categories
                            }
                        }
                        
                        outfile.write(json.dumps(example) + '\n')
                        processed_count += 1
                        
                elif isinstance(data, dict):
                    # Dictionary format (as originally expected)
                    for post_id, post_data in data.items():
                        if not isinstance(post_data, dict):
                            continue
                        
                        caption = post_data.get('caption', '')
                        date = post_data.get('date', '')
                        categories = post_data.get('categories', [])
                        ocr_text = post_data.get('ocr_text', '')
                        
                        # Combine caption and OCR text
                        content = caption
                        if ocr_text:
                            content += f"\n\nImage Text:\n{ocr_text}"
                        
                        # Clean the content
                        content = clean_text(content)
                        
                        # Skip if content is too short
                        if len(content) < 50:
                            continue
                        
                        # Create a training example
                        example = {
                            "text": f"{INSTAGRAM_START}\nUsername: {username}\nDate: {date}\nCategories: {', '.join(categories)}\n\n{content}\n{INSTAGRAM_END}",
                            "metadata": {
                                "source": "instagram",
                                "username": username,
                                "post_id": post_id,
                                "date": date,
                                "categories": categories
                            }
                        }
                        
                        outfile.write(json.dumps(example) + '\n')
                        processed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Processed {processed_count} Instagram posts")
    return processed_count

def create_combined_corpus():
    """Combine all processed data into a single training corpus."""
    logger.info("Creating combined training corpus...")
    
    # Paths to individual corpus files
    lyrics_path = OUTPUT_DIR / "lyrics" / "lyrics_corpus.jsonl"
    podcasts_path = OUTPUT_DIR / "podcasts" / "podcast_corpus.jsonl"
    instagram_path = OUTPUT_DIR / "instagram" / "instagram_corpus.jsonl"
    
    # Output path for combined corpus
    combined_path = OUTPUT_DIR / "combined" / "songwriting_corpus.jsonl"
    
    # Load all examples
    all_examples = []
    
    # Load lyrics
    if lyrics_path.exists():
        with open(lyrics_path, 'r') as f:
            for line in f:
                all_examples.append(json.loads(line))
    
    # Load podcasts
    if podcasts_path.exists():
        with open(podcasts_path, 'r') as f:
            for line in f:
                all_examples.append(json.loads(line))
    
    # Load Instagram
    if instagram_path.exists():
        with open(instagram_path, 'r') as f:
            for line in f:
                all_examples.append(json.loads(line))
    
    logger.info(f"Loaded {len(all_examples)} total examples")
    
    # Shuffle the examples
    random.shuffle(all_examples)
    
    # Write combined corpus
    with open(combined_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Created combined corpus with {len(all_examples)} examples at {combined_path}")
    return len(all_examples)

def create_train_val_split():
    """Split the combined corpus into training and validation sets."""
    logger.info("Creating train/validation split...")
    
    # Paths
    combined_path = OUTPUT_DIR / "combined" / "songwriting_corpus.jsonl"
    train_path = OUTPUT_DIR / "combined" / "train.jsonl"
    val_path = OUTPUT_DIR / "combined" / "val.jsonl"
    
    # Load all examples
    all_examples = []
    with open(combined_path, 'r') as f:
        for line in f:
            all_examples.append(line)
    
    # Shuffle again
    random.shuffle(all_examples)
    
    # Split 90/10
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Write train set
    with open(train_path, 'w') as f:
        for example in train_examples:
            f.write(example)
    
    # Write validation set
    with open(val_path, 'w') as f:
        for example in val_examples:
            f.write(example)
    
    logger.info(f"Created training set with {len(train_examples)} examples")
    logger.info(f"Created validation set with {len(val_examples)} examples")
    
    return len(train_examples), len(val_examples)

def create_plain_text_corpus():
    """Create a plain text version of the corpus for simple tokenizer training."""
    logger.info("Creating plain text corpus...")
    
    # Paths
    combined_path = OUTPUT_DIR / "combined" / "songwriting_corpus.jsonl"
    plain_text_path = OUTPUT_DIR / "combined" / "songwriting_corpus.txt"
    
    # Extract text from all examples
    with open(combined_path, 'r') as f, open(plain_text_path, 'w') as out_f:
        for line in f:
            example = json.loads(line)
            out_f.write(example["text"])
            out_f.write("\n\n")
    
    logger.info(f"Created plain text corpus at {plain_text_path}")

def main():
    """Main function to prepare the corpus for training."""
    logger.info("Starting corpus preparation for training...")
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR / "lyrics", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "podcasts", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "instagram", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "combined", exist_ok=True)
    
    # Process each data source
    lyrics_count = process_lyrics()
    podcast_count = process_podcasts()
    instagram_count = process_instagram()
    
    # Create combined corpus
    total_examples = create_combined_corpus()
    
    # Create train/val split
    train_count, val_count = create_train_val_split()
    
    # Create plain text corpus for simple tokenizer training
    create_plain_text_corpus()
    
    # Print summary
    logger.info("=== Corpus Preparation Complete ===")
    logger.info(f"Processed {lyrics_count} lyrics files")
    logger.info(f"Processed {podcast_count} podcast transcripts")
    logger.info(f"Processed {instagram_count} Instagram posts")
    logger.info(f"Total examples in combined corpus: {total_examples}")
    logger.info(f"Training examples: {train_count}")
    logger.info(f"Validation examples: {val_count}")
    logger.info(f"All files saved to {OUTPUT_DIR}")
    logger.info("The corpus is now ready for training!")

if __name__ == "__main__":
    main()
