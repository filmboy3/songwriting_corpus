#!/usr/bin/env python3
"""
Normalize and clean Instagram corpus files for the songwriting corpus.
This script processes the raw Instagram corpus files to:
1. Normalize text formatting
2. Remove emojis and special characters
3. Clean up hashtags and mentions
4. Standardize line breaks and spacing
5. Extract and categorize songwriting advice
"""

import os
import re
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("instagram_normalization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("instagram_normalization")

# Paths
CORPUS_DIR = Path("songwriting_corpus/instagram")
NORMALIZED_DIR = Path("songwriting_corpus/instagram_normalized")
ADVICE_DIR = Path("songwriting_corpus/instagram_advice")
METADATA_FILE = Path("songwriting_corpus/instagram_metadata.json")

# Create directories if they don't exist
NORMALIZED_DIR.mkdir(exist_ok=True, parents=True)
ADVICE_DIR.mkdir(exist_ok=True, parents=True)

# Regex patterns
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+", flags=re.UNICODE
)

HASHTAG_PATTERN = re.compile(r'#\w+')
MENTION_PATTERN = re.compile(r'@\w+')
URL_PATTERN = re.compile(r'https?://\S+')
EXTRA_SPACES_PATTERN = re.compile(r'\s{2,}')
NEWLINE_PATTERN = re.compile(r'\n{3,}')

# Songwriting advice patterns
ADVICE_PATTERNS = [
    (r'(?i)try this[:\s]+(.*?)(?=\n\n|\Z)', 'technique'),
    (r'(?i)how to[:\s]+(.*?)(?=\n\n|\Z)', 'instruction'),
    (r'(?i)tip[:\s]+(.*?)(?=\n\n|\Z)', 'tip'),
    (r'(?i)advice[:\s]+(.*?)(?=\n\n|\Z)', 'advice'),
    (r'(?i)here\'s a (technique|method|approach|strategy)[:\s]+(.*?)(?=\n\n|\Z)', 'technique'),
    (r'(?i)(\d+)[\s\.]+(steps?|ways?|techniques?|tips?)[:\s]+(.*?)(?=\n\n|\Z)', 'list'),
    (r'(?i)when you[\'re\s]+(writing|stuck|creating)[:\s]+(.*?)(?=\n\n|\Z)', 'situation'),
    (r'(?i)if you[\'re\s]+(struggling|trying|wanting)[:\s]+(.*?)(?=\n\n|\Z)', 'situation')
]

def remove_emojis(text):
    """Remove emojis from text"""
    return EMOJI_PATTERN.sub(r'', text)

def clean_hashtags(text):
    """Extract hashtags and clean the text"""
    hashtags = HASHTAG_PATTERN.findall(text)
    cleaned_text = HASHTAG_PATTERN.sub(r'', text)
    return cleaned_text, hashtags

def clean_mentions(text):
    """Extract mentions and clean the text"""
    mentions = MENTION_PATTERN.findall(text)
    cleaned_text = MENTION_PATTERN.sub(r'', text)
    return cleaned_text, mentions

def clean_urls(text):
    """Remove URLs from text"""
    return URL_PATTERN.sub(r'', text)

def normalize_whitespace(text):
    """Normalize whitespace and line breaks"""
    text = EXTRA_SPACES_PATTERN.sub(' ', text)
    text = NEWLINE_PATTERN.sub('\n\n', text)
    return text.strip()

def extract_songwriting_advice(text):
    """Extract songwriting advice from text"""
    advice_items = []
    
    for pattern, advice_type in ADVICE_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            if advice_type == 'list':
                advice_items.append({
                    'type': advice_type,
                    'number': match.group(1),
                    'category': match.group(2),
                    'content': match.group(3).strip()
                })
            elif advice_type in ['technique', 'instruction', 'tip', 'advice']:
                advice_items.append({
                    'type': advice_type,
                    'content': match.group(1).strip()
                })
            elif advice_type == 'situation':
                advice_items.append({
                    'type': advice_type,
                    'situation': match.group(1).strip(),
                    'advice': match.group(2).strip()
                })
    
    return advice_items

def process_corpus_file(file_path):
    """Process a single corpus file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata from the content
        author_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', content)
        date_match = re.search(r'<DATE>(.*?)</DATE>', content)
        topics_match = re.search(r'<TOPICS>(.*?)</TOPICS>', content)
        url_match = re.search(r'<URL>(.*?)</URL>', content)
        content_match = re.search(r'<CONTENT>(.*?)</CONTENT>', content, re.DOTALL)
        image_text_match = re.search(r'<IMAGE_TEXT>(.*?)</IMAGE_TEXT>', content, re.DOTALL)
        hashtags_match = re.search(r'<HASHTAGS>(.*?)</HASHTAGS>', content, re.DOTALL)
        
        metadata = {
            'author': author_match.group(1) if author_match else '',
            'date': date_match.group(1) if date_match else '',
            'topics': topics_match.group(1).split(', ') if topics_match else [],
            'url': url_match.group(1) if url_match else '',
            'original_file': str(file_path)
        }
        
        # Process content
        if content_match:
            raw_content = content_match.group(1).strip()
            
            # Clean and normalize
            content_no_emojis = remove_emojis(raw_content)
            content_no_urls = clean_urls(content_no_emojis)
            content_clean, extracted_hashtags = clean_hashtags(content_no_urls)
            content_clean, extracted_mentions = clean_mentions(content_clean)
            normalized_content = normalize_whitespace(content_clean)
            
            # Extract advice
            advice_items = extract_songwriting_advice(normalized_content)
            
            # Add to metadata
            metadata['extracted_hashtags'] = extracted_hashtags
            metadata['extracted_mentions'] = extracted_mentions
            metadata['has_advice'] = len(advice_items) > 0
            metadata['advice_count'] = len(advice_items)
            
            # Process image text if available
            if image_text_match:
                image_text = image_text_match.group(1).strip()
                if image_text and image_text != "*No image found*" and not image_text.startswith("*OCR failed"):
                    normalized_image_text = normalize_whitespace(remove_emojis(image_text))
                    image_advice = extract_songwriting_advice(normalized_image_text)
                    advice_items.extend(image_advice)
                    metadata['has_image_advice'] = len(image_advice) > 0
                    metadata['image_advice_count'] = len(image_advice)
            
            # Create normalized content
            normalized_file_content = f"""<INSTAGRAM_NORMALIZED>
<AUTHOR>{metadata['author']}</AUTHOR>
<DATE>{metadata['date']}</DATE>
<TOPICS>{', '.join(metadata['topics'])}</TOPICS>
<URL>{metadata['url']}</URL>

<CONTENT>
{normalized_content}
</CONTENT>

<IMAGE_TEXT>
{normalized_image_text if 'normalized_image_text' in locals() else ''}
</IMAGE_TEXT>

<ADVICE_COUNT>{len(advice_items)}</ADVICE_COUNT>
</INSTAGRAM_NORMALIZED>
"""
            
            # Write normalized file
            normalized_file_path = NORMALIZED_DIR / file_path.name
            with open(normalized_file_path, 'w', encoding='utf-8') as f:
                f.write(normalized_file_content)
            
            # Write advice file if advice exists
            if advice_items:
                advice_file_path = ADVICE_DIR / f"advice_{file_path.name}"
                advice_content = f"""<SONGWRITING_ADVICE>
<AUTHOR>{metadata['author']}</AUTHOR>
<DATE>{metadata['date']}</DATE>
<TOPICS>{', '.join(metadata['topics'])}</TOPICS>
<URL>{metadata['url']}</URL>

<ADVICE_ITEMS>
"""
                for i, advice in enumerate(advice_items):
                    advice_content += f"<ADVICE item=\"{i+1}\" type=\"{advice['type']}\">\n"
                    
                    if advice['type'] == 'list':
                        advice_content += f"<NUMBER>{advice['number']}</NUMBER>\n"
                        advice_content += f"<CATEGORY>{advice['category']}</CATEGORY>\n"
                        advice_content += f"<CONTENT>{advice['content']}</CONTENT>\n"
                    elif advice['type'] == 'situation':
                        advice_content += f"<SITUATION>{advice['situation']}</SITUATION>\n"
                        advice_content += f"<SOLUTION>{advice['advice']}</SOLUTION>\n"
                    else:
                        advice_content += f"<CONTENT>{advice['content']}</CONTENT>\n"
                    
                    advice_content += "</ADVICE>\n\n"
                
                advice_content += "</ADVICE_ITEMS>\n</SONGWRITING_ADVICE>"
                
                with open(advice_file_path, 'w', encoding='utf-8') as f:
                    f.write(advice_content)
            
            return metadata
        
        else:
            logger.warning(f"No content found in {file_path}")
            return None
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def main():
    """Main function to process all corpus files"""
    logger.info("Starting Instagram corpus normalization")
    
    # Get all corpus files
    corpus_files = list(CORPUS_DIR.glob("*.txt"))
    logger.info(f"Found {len(corpus_files)} corpus files to process")
    
    # Process files
    metadata_list = []
    
    print(f"Processing {len(corpus_files)} files...")
    for i, file_path in enumerate(corpus_files):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(corpus_files)} files processed")
        metadata = process_corpus_file(file_path)
        if metadata:
            metadata_list.append(metadata)
    
    # Write metadata
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)
    
    # Generate summary statistics
    total_files = len(metadata_list)
    files_with_advice = sum(1 for m in metadata_list if m.get('has_advice', False))
    total_advice_items = sum(m.get('advice_count', 0) for m in metadata_list)
    
    logger.info(f"Processed {total_files} files")
    logger.info(f"Files with advice: {files_with_advice} ({files_with_advice/total_files*100:.1f}%)")
    logger.info(f"Total advice items extracted: {total_advice_items}")
    
    print(f"\n✅ Finished normalizing Instagram corpus files")
    print(f"📊 Summary:")
    print(f"  - {total_files} files processed")
    print(f"  - {files_with_advice} files with songwriting advice ({files_with_advice/total_files*100:.1f}%)")
    print(f"  - {total_advice_items} advice items extracted")
    print(f"\n📁 Output files:")
    print(f"  - Normalized corpus: {NORMALIZED_DIR}")
    print(f"  - Extracted advice: {ADVICE_DIR}")
    print(f"  - Metadata: {METADATA_FILE}")

if __name__ == "__main__":
    main()
