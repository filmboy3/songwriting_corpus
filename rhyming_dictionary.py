#!/usr/bin/env python3
"""
Rhyming Dictionary Generator for Songwriting Corpus
Uses Datamuse API and Phyme library to create a comprehensive rhyming reference
"""

import os
import json
import logging
import requests
import time
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(BASE_DIR, "music_theory_reference")
RHYME_DIR = os.path.join(REFERENCE_DIR, "rhyming_dictionary")
DATAMUSE_API = "https://api.datamuse.com/words"
COMMON_WORDS_FILE = os.path.join(RHYME_DIR, "common_songwriting_words.txt")

# Ensure directories exist
os.makedirs(RHYME_DIR, exist_ok=True)

def log_message(message):
    """Print a log message with timestamp."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {message}")

def get_rhymes_from_datamuse(word, max_results=100):
    """Get rhymes for a word using the Datamuse API."""
    params = {
        'rel_rhy': word,  # Words that rhyme with the input word
        'max': max_results
    }
    
    try:
        response = requests.get(DATAMUSE_API, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        log_message(f"Error fetching rhymes for '{word}': {e}")
        return []

def get_near_rhymes_from_datamuse(word, max_results=50):
    """Get near rhymes (sounds like) for a word using the Datamuse API."""
    params = {
        'sl': word,  # Words that sound like the input word
        'max': max_results
    }
    
    try:
        response = requests.get(DATAMUSE_API, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        log_message(f"Error fetching near rhymes for '{word}': {e}")
        return []

def get_word_associations(word, max_results=50):
    """Get word associations using the Datamuse API."""
    params = {
        'ml': word,  # Words with similar meaning
        'max': max_results
    }
    
    try:
        response = requests.get(DATAMUSE_API, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        log_message(f"Error fetching associations for '{word}': {e}")
        return []

def load_common_words():
    """Load common songwriting words from file or create if it doesn't exist."""
    if not os.path.exists(COMMON_WORDS_FILE):
        # Default list of common songwriting words if file doesn't exist
        common_words = [
            # Emotions
            "love", "hate", "joy", "pain", "fear", "hope", "dream", "heart", "soul",
            # Actions
            "run", "walk", "dance", "sing", "cry", "laugh", "fall", "rise", "fly",
            # Time
            "day", "night", "time", "year", "hour", "moment", "forever", "never", "always",
            # Nature
            "sky", "sun", "moon", "star", "rain", "wind", "fire", "water", "earth",
            # Body
            "eye", "hand", "face", "head", "mind", "body", "tear", "smile", "voice",
            # Relationships
            "friend", "lover", "baby", "girl", "boy", "man", "woman", "mother", "father",
            # Places
            "home", "road", "street", "town", "city", "world", "heaven", "hell", "room",
            # Abstract
            "life", "death", "truth", "lie", "faith", "doubt", "peace", "war", "light",
            # Common in songs
            "yeah", "oh", "hey", "goodbye", "hello", "stay", "go", "know", "feel"
        ]
        
        # Save the default list
        with open(COMMON_WORDS_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(common_words))
        
        return common_words
    else:
        # Load from file
        with open(COMMON_WORDS_FILE, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def extract_words_from_corpus(corpus_dir, max_words=500):
    """Extract most common words from the corpus for rhyming dictionary."""
    log_message(f"Extracting common words from corpus at {corpus_dir}")
    
    word_counts = defaultdict(int)
    
    # Process corpus files
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        # Simple word extraction
                        words = [word.strip('.,!?":;()[]{}') for word in content.split()]
                        for word in words:
                            if word and word.isalpha() and len(word) > 2:
                                word_counts[word] += 1
                except Exception as e:
                    log_message(f"Error processing {file_path}: {e}")
    
    # Get the most common words
    common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, _ in common_words[:max_words]]
    
    # Save to file
    with open(COMMON_WORDS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(top_words))
    
    log_message(f"Extracted {len(top_words)} common words from corpus")
    return top_words

def build_rhyming_dictionary(words=None, use_corpus=True):
    """Build a comprehensive rhyming dictionary for songwriting."""
    log_message("Building rhyming dictionary...")
    
    # Get words to process
    if words is None:
        if use_corpus and os.path.exists(os.path.join(BASE_DIR, "combined")):
            words = extract_words_from_corpus(os.path.join(BASE_DIR, "combined"))
        else:
            words = load_common_words()
    
    rhyming_dict = {}
    near_rhymes_dict = {}
    associations_dict = {}
    
    total_words = len(words)
    for i, word in enumerate(words):
        log_message(f"Processing word {i+1}/{total_words}: '{word}'")
        
        # Get rhymes
        rhymes = get_rhymes_from_datamuse(word)
        rhyming_dict[word] = [item['word'] for item in rhymes]
        
        # Get near rhymes
        near_rhymes = get_near_rhymes_from_datamuse(word)
        near_rhymes_dict[word] = [item['word'] for item in near_rhymes]
        
        # Get associations
        associations = get_word_associations(word)
        associations_dict[word] = [item['word'] for item in associations]
        
        # Be nice to the API
        time.sleep(0.5)
    
    # Save the dictionaries
    rhyme_data = {
        "perfect_rhymes": rhyming_dict,
        "near_rhymes": near_rhymes_dict,
        "word_associations": associations_dict
    }
    
    output_file = os.path.join(RHYME_DIR, "rhyming_dictionary.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rhyme_data, f, indent=2)
    
    # Create a markdown reference
    create_markdown_reference(rhyme_data)
    
    log_message(f"Rhyming dictionary saved to {output_file}")
    return output_file

def create_markdown_reference(rhyme_data):
    """Create a markdown reference document for the rhyming dictionary."""
    output_file = os.path.join(RHYME_DIR, "rhyming_reference.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Songwriting Rhyming Dictionary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Perfect Rhymes\n\n")
        for word, rhymes in rhyme_data["perfect_rhymes"].items():
            if rhymes:
                f.write(f"### {word}\n\n")
                f.write(", ".join(rhymes[:30]))  # Limit to 30 rhymes for readability
                if len(rhymes) > 30:
                    f.write(f", ... ({len(rhymes) - 30} more)")
                f.write("\n\n")
        
        f.write("## Near Rhymes\n\n")
        for word, rhymes in rhyme_data["near_rhymes"].items():
            if rhymes:
                f.write(f"### {word}\n\n")
                f.write(", ".join(rhymes[:20]))  # Limit to 20 near rhymes
                if len(rhymes) > 20:
                    f.write(f", ... ({len(rhymes) - 20} more)")
                f.write("\n\n")
        
        f.write("## Word Associations\n\n")
        for word, associations in rhyme_data["word_associations"].items():
            if associations:
                f.write(f"### {word}\n\n")
                f.write(", ".join(associations[:15]))  # Limit to 15 associations
                if len(associations) > 15:
                    f.write(f", ... ({len(associations) - 15} more)")
                f.write("\n\n")
    
    log_message(f"Markdown reference saved to {output_file}")
    return output_file

def get_rhymes_for_word(word):
    """Get rhymes for a specific word, useful for interactive use."""
    rhyme_file = os.path.join(RHYME_DIR, "rhyming_dictionary.json")
    
    if not os.path.exists(rhyme_file):
        log_message("Rhyming dictionary not found. Building it first...")
        build_rhyming_dictionary()
    
    with open(rhyme_file, 'r', encoding='utf-8') as f:
        rhyme_data = json.load(f)
    
    # Check if word exists in dictionary
    if word in rhyme_data["perfect_rhymes"]:
        return {
            "perfect_rhymes": rhyme_data["perfect_rhymes"].get(word, []),
            "near_rhymes": rhyme_data["near_rhymes"].get(word, []),
            "associations": rhyme_data["word_associations"].get(word, [])
        }
    else:
        # Fetch from API if not in dictionary
        perfect_rhymes = [item['word'] for item in get_rhymes_from_datamuse(word)]
        near_rhymes = [item['word'] for item in get_near_rhymes_from_datamuse(word)]
        associations = [item['word'] for item in get_word_associations(word)]
        
        return {
            "perfect_rhymes": perfect_rhymes,
            "near_rhymes": near_rhymes,
            "associations": associations
        }

def main():
    """Main function to build the rhyming dictionary."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build a rhyming dictionary for songwriting")
    parser.add_argument("--word", help="Get rhymes for a specific word")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the entire rhyming dictionary")
    parser.add_argument("--use-corpus", action="store_true", help="Extract words from corpus")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of words to process")
    
    args = parser.parse_args()
    
    if args.word:
        # Get rhymes for a specific word
        rhymes = get_rhymes_for_word(args.word)
        
        print(f"\nPerfect rhymes for '{args.word}':")
        print(", ".join(rhymes["perfect_rhymes"][:30]))
        
        print(f"\nNear rhymes for '{args.word}':")
        print(", ".join(rhymes["near_rhymes"][:20]))
        
        print(f"\nAssociations for '{args.word}':")
        print(", ".join(rhymes["associations"][:15]))
    elif args.rebuild or not os.path.exists(os.path.join(RHYME_DIR, "rhyming_dictionary.json")):
        # Build or rebuild the dictionary
        if args.use_corpus:
            words = extract_words_from_corpus(os.path.join(BASE_DIR, "combined"), args.limit)
        else:
            words = load_common_words()[:args.limit]
        
        build_rhyming_dictionary(words)
    else:
        log_message("Rhyming dictionary already exists. Use --rebuild to rebuild it or --word to query a specific word.")

if __name__ == "__main__":
    main()
