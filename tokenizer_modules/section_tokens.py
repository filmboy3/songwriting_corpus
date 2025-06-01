#!/usr/bin/env python3
"""Section tokens for the tokenizer."""

import logging
from typing import Set

logger = logging.getLogger("tokenizer_preparation")

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
