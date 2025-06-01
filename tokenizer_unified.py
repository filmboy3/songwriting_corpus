#!/usr/bin/env python3
"""
Unified Tokenizer Preparation Script

This script prepares a custom tokenizer for the songwriting model with:
- Chord tokens (e.g., <C>, <Am>, <F/G>)
- Section tokens (e.g., <VERSE>, <CHORUS>)
- Formatting tokens
- Control tokens

Usage:
  python tokenizer_unified.py [--base-model gpt2] [--output-dir ./model_output/tokenizer]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Set, List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tokenizer_preparation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tokenizer_preparation")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "model_output", "tokenizer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Token generation functions
def generate_chord_tokens() -> Set[str]:
    """Generate special tokens for chords."""
    chord_tokens = set()
    
    # Basic chord types
    root_notes = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
    chord_types = ["", "m", "7", "m7", "maj7", "dim", "aug", "sus2", "sus4", "6", "m6", "9", "m9", "add9", "dim7"]
    
    # Generate chord tokens
    for root in root_notes:
        for chord_type in chord_types:
            chord = f"{root}{chord_type}"
            token = f"<{chord}>"
            chord_tokens.add(token)
    
    # Add slash chords
    for chord in list(chord_tokens):
        if chord.startswith("<") and chord.endswith(">"):
            chord_name = chord[1:-1]
            for bass_note in root_notes:
                slash_chord = f"<{chord_name}/{bass_note}>"
                chord_tokens.add(slash_chord)
    
    logger.info(f"Generated {len(chord_tokens)} chord tokens")
    return chord_tokens

def generate_section_tokens() -> Set[str]:
    """Generate special tokens for song sections."""
    section_tokens = set()
    
    # Basic section types
    section_types = [
        "INTRO", "VERSE", "PRE-CHORUS", "CHORUS", "BRIDGE", "OUTRO", "SOLO",
        "INSTRUMENTAL", "BREAKDOWN", "HOOK", "REFRAIN", "INTERLUDE", "CODA"
    ]
    
    # Generate section tokens
    for section in section_types:
        start_token = f"<{section}>"
        end_token = f"</{section}>"
        section_tokens.add(start_token)
        section_tokens.add(end_token)
    
    # Add numbered sections
    for section in ["VERSE", "CHORUS", "BRIDGE"]:
        for i in range(1, 6):  # Up to 5 verses, choruses, bridges
            start_token = f"<{section}{i}>"
            end_token = f"</{section}{i}>"
            section_tokens.add(start_token)
            section_tokens.add(end_token)
    
    logger.info(f"Generated {len(section_tokens)} section tokens")
    return section_tokens

def generate_formatting_tokens() -> Set[str]:
    """Generate formatting tokens."""
    formatting_tokens = set([
        "<LINEBREAK>", "<NEWLINE>", "<TAB>", "<SPACE>", "<INDENT>",
        "<CHORD_SEQUENCE>", "</CHORD_SEQUENCE>",
        "<LYRICS>", "</LYRICS>",
        "<TITLE>", "</TITLE>",
        "<ARTIST>", "</ARTIST>",
        "<CAPO>", "</CAPO>",
        "<KEY>", "</KEY>",
        "<BPM>", "</BPM>",
        "<TIME_SIGNATURE>", "</TIME_SIGNATURE>",
        "<TRANSCRIPT>", "</TRANSCRIPT>",
        "<PODCAST>", "</PODCAST>",
        "<BOOK_EXCERPT>", "</BOOK_EXCERPT>",
        "<RHYME_PATTERN>", "</RHYME_PATTERN>"
    ])
    
    logger.info(f"Generated {len(formatting_tokens)} formatting tokens")
    return formatting_tokens

def generate_control_tokens() -> Set[str]:
    """Generate control tokens."""
    control_tokens = set([
        "<|startoftext|>",
        "
