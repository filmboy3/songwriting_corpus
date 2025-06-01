#!/usr/bin/env python3
"""Formatting tokens for the tokenizer."""

import logging
from typing import Set

logger = logging.getLogger("tokenizer_preparation")

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
