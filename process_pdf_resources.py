#!/usr/bin/env python3
"""
Script to process PDF resources for the songwriting corpus.
Extracts text from PDFs and organizes them into reference materials.
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Installing...")
    import subprocess
    subprocess.check_call(["/Users/jonathanschwartz/Documents/Transcriptions/whisperenv/bin/pip", "install", "PyPDF2"])
    import PyPDF2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pdf_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PDFS_DIR = os.path.join(BASE_DIR, "raw_pdfs")
REFERENCE_DIR = os.path.join(BASE_DIR, "music_theory_reference")
RHYMING_DIR = os.path.join(REFERENCE_DIR, "rhyming_dictionary")
IMAGERY_DIR = os.path.join(REFERENCE_DIR, "imagery_resources")

# Ensure directories exist
os.makedirs(RAW_PDFS_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(RHYMING_DIR, exist_ok=True)
os.makedirs(IMAGERY_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            logger.info(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
                # Log progress for large PDFs
                if page_num % 10 == 0 and page_num > 0:
                    logger.info(f"Processed {page_num}/{num_pages} pages")
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def process_rhyming_dictionary(pdf_path, output_dir=None):
    """Process a rhyming dictionary PDF and extract structured data."""
    if output_dir is None:
        output_dir = RHYMING_DIR
    
    logger.info(f"Processing rhyming dictionary: {pdf_path}")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.error("Failed to extract text from PDF")
        return False
    
    # Save raw text
    pdf_name = os.path.basename(pdf_path)
    base_name = os.path.splitext(pdf_name)[0]
    raw_text_file = os.path.join(output_dir, f"{base_name}_raw.txt")
    
    with open(raw_text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Raw text saved to {raw_text_file}")
    
    # Try to extract structured rhyming data
    # This is a simplified approach and might need customization based on the PDF structure
    rhyme_data = {}
    current_word = None
    
    # Simple pattern matching to find word entries and their rhymes
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts with a word (capitalized, followed by rhymes)
        word_match = re.match(r'^([A-Z][A-Za-z\'-]+)(\s+.+)?$', line)
        if word_match:
            current_word = word_match.group(1).lower()
            rhyme_data[current_word] = []
            
            # If there are rhymes on the same line
            if word_match.group(2):
                rhymes = word_match.group(2).strip().split()
                rhyme_data[current_word].extend([r.lower() for r in rhymes])
        elif current_word and not re.match(r'^[0-9]+$', line):
            # Assume this line contains rhymes for the current word
            # Skip lines that are just page numbers
            rhymes = line.split()
            rhyme_data[current_word].extend([r.lower() for r in rhymes])
    
    # Save structured data
    json_file = os.path.join(output_dir, f"{base_name}_structured.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(rhyme_data, f, indent=2)
    
    logger.info(f"Structured rhyming data saved to {json_file}")
    
    # Create a markdown reference
    md_file = os.path.join(output_dir, f"{base_name}_reference.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# {base_name} - Rhyming Dictionary\n\n")
        f.write(f"Extracted from: {pdf_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        for word, rhymes in rhyme_data.items():
            if rhymes:
                f.write(f"## {word}\n\n")
                f.write(", ".join(rhymes[:50]))
                if len(rhymes) > 50:
                    f.write(f", ... ({len(rhymes) - 50} more)")
                f.write("\n\n")
    
    logger.info(f"Markdown reference saved to {md_file}")
    return True

def process_imagery_resource(pdf_path, output_dir=None):
    """Process an imagery resource PDF and extract content."""
    if output_dir is None:
        output_dir = IMAGERY_DIR
    
    logger.info(f"Processing imagery resource: {pdf_path}")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.error("Failed to extract text from PDF")
        return False
    
    # Save raw text
    pdf_name = os.path.basename(pdf_path)
    base_name = os.path.splitext(pdf_name)[0]
    raw_text_file = os.path.join(output_dir, f"{base_name}_raw.txt")
    
    with open(raw_text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Raw text saved to {raw_text_file}")
    
    # Try to extract structured imagery data
    # For "14,000 things to be happy about", we'll try to extract the list items
    happy_things = []
    
    # Simple pattern matching to find list items
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Skip lines that are likely headers, page numbers, etc.
        if len(line) < 100 and not re.match(r'^[0-9]+$', line):
            # Split by common separators
            items = re.split(r'[•·,;]', line)
            for item in items:
                item = item.strip()
                if item and len(item) > 3:
                    happy_things.append(item)
    
    # Save structured data
    json_file = os.path.join(output_dir, f"{base_name}_structured.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({"items": happy_things}, f, indent=2)
    
    logger.info(f"Structured imagery data saved to {json_file}")
    
    # Create a markdown reference
    md_file = os.path.join(output_dir, f"{base_name}_reference.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# {base_name} - Imagery Resource\n\n")
        f.write(f"Extracted from: {pdf_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Items\n\n")
        for i, item in enumerate(happy_things):
            f.write(f"- {item}\n")
            if i > 0 and i % 100 == 0:
                f.write("\n")
    
    logger.info(f"Markdown reference saved to {md_file}")
    return True

def main():
    """Main function to process PDF resources."""
    parser = argparse.ArgumentParser(description="Process PDF resources for the songwriting corpus")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--type", choices=["rhyming", "imagery"], default="rhyming", help="Type of resource")
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        return
    
    # Process based on type
    if args.type == "rhyming":
        process_rhyming_dictionary(args.pdf_path)
    elif args.type == "imagery":
        process_imagery_resource(args.pdf_path)

if __name__ == "__main__":
    main()
