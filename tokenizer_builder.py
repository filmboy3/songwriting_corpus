#!/usr/bin/env python3
"""
Tokenizer Builder for Songwriting Corpus

This script prepares a custom tokenizer for the songwriting model by:
1. Adding special tokens for chord progressions (e.g., <C>, <Am>, <F>, <G>)
2. Adding special tokens for song sections (e.g., <VERSE>, <CHORUS>, <BRIDGE>)
3. Adding special tokens for line breaks and formatting
4. Training the tokenizer on the combined corpus
5. Saving the tokenizer for use in model training

The tokenizer can be based on an existing model (e.g., GPT-2) or trained from scratch.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import re
from typing import List, Dict, Set, Optional
from tqdm import tqdm

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tokenizer_builder.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("tokenizer_builder")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_DIR = os.path.join(BASE_DIR, "combined")
LYRICS_DIR = os.path.join(COMBINED_DIR, "lyrics")
CHORDS_DIR = os.path.join(COMBINED_DIR, "chords")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_output", "tokenizer")
MUSIC_THEORY_DIR = os.path.join(BASE_DIR, "music_theory_reference")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_header(message):
    """Print a formatted header message."""
    header = f"\n{'=' * 80}\n  {message}\n{'=' * 80}\n"
    logger.info(header)
    return header

class TokenizerBuilder:
    """Class for building a custom tokenizer with chord and section tokens."""
    
    def __init__(self, base_model="gpt2", vocab_size=50257):
        """Initialize the TokenizerBuilder class."""
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.chord_tokens = set()
        self.section_tokens = set()
        self.special_tokens = set()
        self.all_special_tokens = set()
        
        # Initialize tokenizer
        self.initialize_tokenizer()
        
        # Generate tokens
        self.generate_chord_tokens()
        self.generate_section_tokens()
        self.generate_special_tokens()
        
        # Combine all special tokens
        self.combine_special_tokens()
    
    def initialize_tokenizer(self):
        """Initialize the tokenizer based on the specified base model."""
        print_header(f"INITIALIZING TOKENIZER FROM {self.base_model}")
        
        try:
            # Initialize from an existing model
            logger.info(f"Initializing tokenizer from {self.base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            logger.info(f"Loaded tokenizer with vocabulary size: {len(self.tokenizer)}")
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise
    
    def generate_chord_tokens(self):
        """Generate special tokens for chords."""
        print_header("GENERATING CHORD TOKENS")
        
        # Basic chord types
        root_notes = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
        chord_types = ["", "m", "7", "m7", "maj7", "dim", "aug", "sus2", "sus4", "6", "m6", "9", "m9", "add9", "dim7"]
        
        # Generate chord tokens
        for root in root_notes:
            for chord_type in chord_types:
                chord = f"{root}{chord_type}"
                token = f"<{chord}>"
                self.chord_tokens.add(token)
        
        # Add slash chords for common combinations
        common_slash_combinations = [
            ("C", "E"), ("C", "G"), ("Am", "G"), ("F", "C"), ("G", "B"),
            ("D", "F#"), ("Em", "D"), ("G", "D"), ("A", "C#"), ("E", "G#")
        ]
        
        for chord_root, bass_note in common_slash_combinations:
            slash_chord = f"<{chord_root}/{bass_note}>"
            self.chord_tokens.add(slash_chord)
        
        logger.info(f"Generated {len(self.chord_tokens)} chord tokens")
    
    def generate_section_tokens(self):
        """Generate special tokens for song sections."""
        print_header("GENERATING SECTION TOKENS")
        
        # Basic section types
        section_types = [
            "INTRO", "VERSE", "PRE-CHORUS", "CHORUS", "BRIDGE", "OUTRO", "SOLO",
            "INSTRUMENTAL", "BREAKDOWN", "HOOK", "REFRAIN", "INTERLUDE", "CODA"
        ]
        
        # Generate section tokens
        for section in section_types:
            start_token = f"<{section}>"
            end_token = f"</{section}>"
            self.section_tokens.add(start_token)
            self.section_tokens.add(end_token)
        
        # Add numbered sections
        for section in ["VERSE", "CHORUS", "BRIDGE"]:
            for i in range(1, 6):  # Up to 5 verses, choruses, bridges
                start_token = f"<{section}{i}>"
                end_token = f"</{section}{i}>"
                self.section_tokens.add(start_token)
                self.section_tokens.add(end_token)
        
        logger.info(f"Generated {len(self.section_tokens)} section tokens")
    
    def generate_special_tokens(self):
        """Generate other special tokens for formatting and control."""
        print_header("GENERATING SPECIAL TOKENS")
        
        # Formatting tokens
        formatting_tokens = [
            "<LINEBREAK>",
            "<NEWLINE>",
            "<TAB>",
            "<SPACE>",
            "<INDENT>",
            "<CHORD_SEQUENCE>",
            "</CHORD_SEQUENCE>",
            "<LYRICS>",
            "</LYRICS>",
            "<TITLE>",
            "</TITLE>",
            "<ARTIST>",
            "</ARTIST>",
            "<CAPO>",
            "</CAPO>",
            "<KEY>",
            "</KEY>",
            "<BPM>",
            "</BPM>",
            "<TIME_SIGNATURE>",
            "</TIME_SIGNATURE>"
        ]
        
        # Control tokens
        control_tokens = [
            "<|startoftext|>",
            "
