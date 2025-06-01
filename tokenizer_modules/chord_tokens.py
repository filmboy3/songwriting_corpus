#!/usr/bin/env python3
"""Chord tokens for the tokenizer."""

import logging
from typing import Set

logger = logging.getLogger("tokenizer_preparation")

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
