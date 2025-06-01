#!/usr/bin/env python3
"""
Script to download and set up the CMU Pronouncing Dictionary for phoneme and rhyme analysis.
This dictionary is essential for the rhyme intelligence module and syllable counting.
"""

import os
import sys
import nltk
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cmudict_download.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cmudict_downloader")

def print_header(message):
    """Print a formatted header message."""
    header = f"\n{'=' * 80}\n  {message}\n{'=' * 80}\n"
    logger.info(header)
    return header

def download_cmudict():
    """Download the CMU Pronouncing Dictionary using NLTK."""
    print_header("DOWNLOADING CMU PRONOUNCING DICTIONARY")
    
    try:
        # Download the CMU dictionary
        logger.info("Downloading CMU Pronouncing Dictionary...")
        nltk.download('cmudict')
        
        # Verify the download
        from nltk.corpus import cmudict
        d = cmudict.dict()
        
        # Test a few words
        test_words = ['hello', 'world', 'songwriting', 'music']
        for word in test_words:
            if word in d:
                logger.info(f"'{word}' pronunciation: {d[word][0]}")
            else:
                logger.warning(f"'{word}' not found in dictionary")
        
        # Create a directory for phoneme analysis if it doesn't exist
        phoneme_dir = Path('music_theory_reference/phoneme_analysis')
        phoneme_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple test file to verify functionality
        test_file = phoneme_dir / 'cmudict_test.py'
        with open(test_file, 'w') as f:
            f.write("""#!/usr/bin/env python3
'''
Test script for CMU Pronouncing Dictionary functionality.
'''

from nltk.corpus import cmudict

# Load the dictionary
d = cmudict.dict()

def get_phonemes(word):
    '''Get the phonemes for a word.'''
    return d.get(word.lower(), [[""]])[0]

def count_syllables(word):
    '''Count the number of syllables in a word.'''
    phonemes = get_phonemes(word)
    return len([p for p in phonemes if p[-1].isdigit()])

def does_rhyme(word1, word2):
    '''Check if two words rhyme.'''
    p1 = get_phonemes(word1)
    p2 = get_phonemes(word2)
    return p1[-2:] == p2[-2:] if len(p1) >= 2 and len(p2) >= 2 else False

# Test with some words
test_words = [
    ('hello', 'yellow'),
    ('song', 'wrong'),
    ('music', 'acoustic'),
    ('rhyme', 'time')
]

for w1, w2 in test_words:
    print(f"{w1} ({count_syllables(w1)} syllables) and {w2} ({count_syllables(w2)} syllables)")
    print(f"Phonemes for {w1}: {get_phonemes(w1)}")
    print(f"Phonemes for {w2}: {get_phonemes(w2)}")
    print(f"Do they rhyme? {does_rhyme(w1, w2)}\\n")
""")
        
        logger.info(f"Created test script at {test_file}")
        logger.info("Making the test script executable...")
        os.chmod(test_file, 0o755)
        
        logger.info("CMU Pronouncing Dictionary successfully downloaded and set up!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading CMU dictionary: {e}")
        return False

def main():
    """Main function to download the CMU dictionary."""
    print_header("CMU DICTIONARY DOWNLOADER")
    
    success = download_cmudict()
    
    if success:
        print_header("DOWNLOAD COMPLETE")
        logger.info("The CMU Pronouncing Dictionary has been successfully downloaded.")
        logger.info("You can now use it for phoneme analysis, syllable counting, and rhyme detection.")
        logger.info("Test the functionality by running: python music_theory_reference/phoneme_analysis/cmudict_test.py")
    else:
        print_header("DOWNLOAD FAILED")
        logger.error("Failed to download the CMU Pronouncing Dictionary.")
        logger.error("Please check your internet connection and try again.")
        logger.error("You can also manually download it by running: python -c 'import nltk; nltk.download(\"cmudict\")'")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
