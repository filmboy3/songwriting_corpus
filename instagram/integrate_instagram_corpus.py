#!/usr/bin/env python3
"""
Integrate Instagram corpus with the main songwriting corpus pipeline.
This script:
1. Copies normalized Instagram content to the main corpus
2. Updates corpus metadata to include Instagram sources
3. Integrates extracted songwriting techniques with other sources
4. Prepares data for tokenization
"""

import os
import json
import shutil
from pathlib import Path
import logging
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("instagram_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("instagram_integration")

# Paths
INSTAGRAM_NORMALIZED_DIR = Path("songwriting_corpus/instagram_normalized")
INSTAGRAM_ADVICE_DIR = Path("songwriting_corpus/instagram_advice")
INSTAGRAM_TECHNIQUES_FILE = Path("songwriting_corpus/techniques/andrea_stolpe_techniques.json")

# Main corpus paths
MAIN_CORPUS_DIR = Path("/Users/jonathanschwartz/CascadeProjects/songwriting_corpus")
CORPUS_INSTAGRAM_DIR = MAIN_CORPUS_DIR / "instagram"
CORPUS_TECHNIQUES_DIR = MAIN_CORPUS_DIR / "techniques"
CORPUS_METADATA_FILE = MAIN_CORPUS_DIR / "corpus_metadata.json"

def ensure_directories():
    """Ensure all required directories exist"""
    CORPUS_INSTAGRAM_DIR.mkdir(exist_ok=True, parents=True)
    CORPUS_TECHNIQUES_DIR.mkdir(exist_ok=True, parents=True)

def copy_normalized_content():
    """Copy normalized Instagram content to main corpus"""
    logger.info("Copying normalized Instagram content to main corpus")
    
    # Count files before copying
    existing_files = list(CORPUS_INSTAGRAM_DIR.glob("*.txt"))
    existing_count = len(existing_files)
    logger.info(f"Found {existing_count} existing Instagram files in corpus")
    
    # Copy normalized files
    normalized_files = list(INSTAGRAM_NORMALIZED_DIR.glob("*.txt"))
    copied_count = 0
    
    for file_path in normalized_files:
        target_path = CORPUS_INSTAGRAM_DIR / file_path.name
        if not target_path.exists():
            shutil.copy2(file_path, target_path)
            copied_count += 1
    
    logger.info(f"Copied {copied_count} new Instagram files to corpus")
    return copied_count

def update_corpus_metadata():
    """Update corpus metadata with Instagram information"""
    logger.info("Updating corpus metadata")
    
    # Create metadata if it doesn't exist
    if not CORPUS_METADATA_FILE.exists():
        metadata = {
            "sources": [],
            "last_updated": str(datetime.datetime.now()),
            "total_files": 0,
            "source_counts": {}
        }
    else:
        with open(CORPUS_METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Check if Instagram source already exists
    instagram_source = next((s for s in metadata.get("sources", []) if s.get("name") == "Instagram - Andrea Stolpe"), None)
    
    if not instagram_source:
        # Add Instagram source
        instagram_source = {
            "name": "Instagram - Andrea Stolpe",
            "type": "social_media",
            "description": "Songwriting advice and techniques from Andrea Stolpe's Instagram account",
            "url": "https://www.instagram.com/andreastolpeofficial/",
            "date_added": str(datetime.datetime.now())
        }
        metadata.setdefault("sources", []).append(instagram_source)
    
    # Update file counts
    instagram_files = list(CORPUS_INSTAGRAM_DIR.glob("*.txt"))
    metadata["source_counts"]["Instagram - Andrea Stolpe"] = len(instagram_files)
    
    # Update total files
    total_files = sum(metadata.get("source_counts", {}).values())
    metadata["total_files"] = total_files
    metadata["last_updated"] = str(datetime.datetime.now())
    
    # Save updated metadata
    with open(CORPUS_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Updated corpus metadata with {len(instagram_files)} Instagram files")
    return metadata

def integrate_techniques():
    """Integrate Instagram techniques with other songwriting techniques"""
    logger.info("Integrating songwriting techniques")
    
    # Load Instagram techniques
    with open(INSTAGRAM_TECHNIQUES_FILE, 'r', encoding='utf-8') as f:
        instagram_techniques = json.load(f)
    
    # Create combined techniques file if it doesn't exist
    combined_file = CORPUS_TECHNIQUES_DIR / "combined_techniques.json"
    if not combined_file.exists():
        combined_techniques = []
    else:
        with open(combined_file, 'r', encoding='utf-8') as f:
            combined_techniques = json.load(f)
    
    # Add source identifier to Instagram techniques
    for technique in instagram_techniques:
        technique["source_name"] = "Instagram - Andrea Stolpe"
        technique["source_type"] = "social_media"
    
    # Remove any existing Instagram techniques to avoid duplicates
    combined_techniques = [t for t in combined_techniques if t.get("source_name") != "Instagram - Andrea Stolpe"]
    
    # Add Instagram techniques
    combined_techniques.extend(instagram_techniques)
    
    # Save combined techniques
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_techniques, f, indent=2)
    
    # Also copy the markdown reference
    shutil.copy2(
        Path("songwriting_corpus/techniques/andrea_stolpe_techniques.md"),
        CORPUS_TECHNIQUES_DIR / "andrea_stolpe_techniques.md"
    )
    
    logger.info(f"Integrated {len(instagram_techniques)} techniques into combined techniques file")
    return len(instagram_techniques)

def prepare_for_tokenization():
    """Prepare Instagram content for tokenization"""
    logger.info("Preparing Instagram content for tokenization")
    
    # Create a file listing all Instagram content for tokenization
    instagram_files = list(CORPUS_INSTAGRAM_DIR.glob("*.txt"))
    
    with open(MAIN_CORPUS_DIR / "instagram_files_for_tokenization.txt", 'w', encoding='utf-8') as f:
        for file_path in instagram_files:
            f.write(f"{file_path}\n")
    
    logger.info(f"Prepared {len(instagram_files)} Instagram files for tokenization")
    return len(instagram_files)

def main():
    """Main function to integrate Instagram corpus"""
    logger.info("Starting Instagram corpus integration")
    
    # Ensure directories exist
    ensure_directories()
    
    # Copy normalized content
    copied_count = copy_normalized_content()
    
    # Update corpus metadata
    metadata = update_corpus_metadata()
    
    # Integrate techniques
    techniques_count = integrate_techniques()
    
    # Prepare for tokenization
    files_for_tokenization = prepare_for_tokenization()
    
    print(f"\n✅ Finished integrating Instagram corpus")
    print(f"📊 Summary:")
    print(f"  - {copied_count} new Instagram files added to corpus")
    print(f"  - {metadata['total_files']} total files in corpus")
    print(f"  - {techniques_count} songwriting techniques integrated")
    print(f"  - {files_for_tokenization} files prepared for tokenization")
    print(f"\n📁 Output locations:")
    print(f"  - Corpus files: {CORPUS_INSTAGRAM_DIR}")
    print(f"  - Techniques: {CORPUS_TECHNIQUES_DIR}")
    print(f"  - Metadata: {CORPUS_METADATA_FILE}")
    print(f"\n🔍 Next steps:")
    print("  1. Run tokenization on the integrated corpus")
    print("  2. Update your model training pipeline to include Instagram content")
    print("  3. Consider adding more Instagram accounts for additional songwriting content")

if __name__ == "__main__":
    main()
