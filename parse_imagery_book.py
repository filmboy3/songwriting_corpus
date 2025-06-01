#!/usr/bin/env python3
"""
Script to parse the imagery happiness book and create a structured dataset
for use in the prompt injection engine and lyric generation.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("imagery_parsing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("imagery_parser")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, "music_theory_reference", "imagery_resources")
OUTPUT_DIR = RESOURCES_DIR

def print_header(message):
    """Print a formatted header message."""
    header = f"\n{'=' * 80}\n  {message}\n{'=' * 80}\n"
    logger.info(header)
    return header

def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {RESOURCES_DIR}")

def parse_imagery_book(input_file):
    """Parse the imagery happiness book into a structured format."""
    print_header(f"PARSING IMAGERY BOOK: {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None
    
    try:
        # Read the file
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Split into lines and process
        lines = content.split('\n')
        
        # Initialize data structure
        imagery_data = {
            "categories": {},
            "emotions": {},
            "imagery_items": []
        }
        
        # Process lines
        current_category = None
        current_emotion = None
        
        for line in tqdm(lines, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a category header (all caps)
            if line.isupper() and len(line) > 3:
                current_category = line.title()
                if current_category not in imagery_data["categories"]:
                    imagery_data["categories"][current_category] = []
                continue
            
            # Check if this is an emotion indicator (has ":" and is short)
            if ":" in line and len(line) < 30:
                parts = line.split(":", 1)
                if len(parts) == 2 and parts[0].strip() and len(parts[0]) < 20:
                    current_emotion = parts[0].strip()
                    if current_emotion not in imagery_data["emotions"]:
                        imagery_data["emotions"][current_emotion] = []
                    continue
            
            # Otherwise, it's an imagery item
            if line and not line.startswith('#') and not line.startswith('//'):
                item = {
                    "text": line,
                    "category": current_category,
                    "emotion": current_emotion
                }
                
                # Add to main list
                imagery_data["imagery_items"].append(item)
                
                # Add to category and emotion lists
                if current_category:
                    imagery_data["categories"][current_category].append(line)
                if current_emotion:
                    imagery_data["emotions"][current_emotion].append(line)
        
        logger.info(f"Parsed {len(imagery_data['imagery_items'])} imagery items")
        logger.info(f"Found {len(imagery_data['categories'])} categories")
        logger.info(f"Found {len(imagery_data['emotions'])} emotions")
        
        return imagery_data
        
    except Exception as e:
        logger.error(f"Error parsing imagery book: {e}")
        return None

def save_imagery_data(imagery_data, output_file):
    """Save the parsed imagery data to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(imagery_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved imagery data to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving imagery data: {e}")
        return False

def create_sample_data():
    """Create a sample imagery dataset if no input file is provided."""
    print_header("CREATING SAMPLE IMAGERY DATASET")
    
    sample_data = {
        "categories": {
            "Nature": [
                "Sunlight filtering through autumn leaves",
                "Ocean waves crashing against weathered cliffs",
                "Dewdrops glistening on spider webs at dawn",
                "Wind whispering through tall pine trees"
            ],
            "Urban": [
                "Neon lights reflecting in rain-slicked streets",
                "Silhouettes of skyscrapers against a sunset sky",
                "Street musicians playing in an empty subway station",
                "Coffee shop windows fogged with condensation"
            ],
            "Emotional": [
                "The weight of unspoken words between old friends",
                "A laughter that echoes long after the joke",
                "The first breath after crying",
                "Fingertips barely touching across a table"
            ]
        },
        "emotions": {
            "Joy": [
                "Dancing barefoot on warm sand",
                "Finding an unexpected letter from someone you miss",
                "The first bite of a meal you've been craving"
            ],
            "Melancholy": [
                "Empty swings moving slightly in the wind",
                "The fading scent of someone's perfume",
                "Dog-eared pages of a book you'll never finish"
            ],
            "Longing": [
                "Watching planes disappear into clouds",
                "Tracing the outline of a phone number you won't call",
                "The space between your fingers that once held another's"
            ]
        },
        "imagery_items": []
    }
    
    # Populate the imagery_items list
    for category, items in sample_data["categories"].items():
        for item in items:
            sample_data["imagery_items"].append({
                "text": item,
                "category": category,
                "emotion": None
            })
    
    for emotion, items in sample_data["emotions"].items():
        for item in items:
            sample_data["imagery_items"].append({
                "text": item,
                "category": None,
                "emotion": emotion
            })
    
    logger.info(f"Created sample dataset with {len(sample_data['imagery_items'])} imagery items")
    return sample_data

def main():
    """Main function to parse the imagery book."""
    parser = argparse.ArgumentParser(description="Parse the imagery happiness book into a structured dataset")
    parser.add_argument("--input", type=str, help="Path to the input imagery book file")
    parser.add_argument("--output", type=str, help="Path to save the output JSON file")
    parser.add_argument("--sample", action="store_true", help="Create a sample dataset if no input file is provided")
    args = parser.parse_args()
    
    print_header("IMAGERY BOOK PARSER")
    
    # Ensure directories exist
    ensure_directories()
    
    # Determine input and output files
    input_file = args.input
    output_file = args.output or os.path.join(OUTPUT_DIR, "imagery_dataset.json")
    
    # Parse the imagery book or create sample data
    if input_file and os.path.exists(input_file):
        imagery_data = parse_imagery_book(input_file)
    elif args.sample or not input_file:
        logger.info("No input file provided or --sample flag used. Creating sample dataset.")
        imagery_data = create_sample_data()
    else:
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    # Save the data
    if imagery_data:
        success = save_imagery_data(imagery_data, output_file)
        if success:
            print_header("PARSING COMPLETE")
            logger.info(f"Imagery data saved to {output_file}")
            logger.info(f"Total imagery items: {len(imagery_data['imagery_items'])}")
            logger.info(f"Categories: {', '.join(imagery_data['categories'].keys())}")
            logger.info(f"Emotions: {', '.join(imagery_data['emotions'].keys())}")
            return 0
    
    print_header("PARSING FAILED")
    return 1

if __name__ == "__main__":
    sys.exit(main())
