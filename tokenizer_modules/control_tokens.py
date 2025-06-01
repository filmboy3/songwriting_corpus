#!/usr/bin/env python3
"""Control tokens for the tokenizer."""

import logging
from typing import Set

logger = logging.getLogger("tokenizer_preparation")

def generate_control_tokens() -> Set[str]:
    """Generate control tokens for model control and instruction."""
    control_tokens = set([
        "<|startoftext|>",
        "