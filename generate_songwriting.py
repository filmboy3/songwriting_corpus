#!/usr/bin/env python3
"""
Generate Songwriting Content

This script generates songwriting content using a trained language model.
It provides various prompts for different types of content (lyrics, podcast, Instagram).
"""

import os
import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = BASE_DIR / "model_output"

# Special tokens
SONG_START = "<|song|>"
PODCAST_START = "<|podcast|>"
INSTAGRAM_START = "<|instagram|>"
VERSE_START = "<|verse|>"
CHORUS_START = "<|chorus|>"
BRIDGE_START = "<|bridge|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate songwriting content")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_DIR),
        help="Path to the trained model",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["lyrics", "podcast", "instagram", "custom"],
        default="lyrics",
        help="Type of content to generate",
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default="",
        help="Custom prompt for generation (used when prompt_type is 'custom')",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of generated text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()

def get_prompt(prompt_type, custom_prompt=None):
    """Generate appropriate prompt based on type."""
    base_prompt = ""
    
    if prompt_type == "lyrics":
        if custom_prompt:
            # Extract a potential title from the custom prompt
            potential_title = custom_prompt.replace("Write a song about ", "").strip()
            base_prompt = f"{SONG_START}\nTitle: Song About {potential_title}\nArtist: AI Songwriter\n\n{VERSE_START}\n"
        else:
            base_prompt = f"{SONG_START}\nTitle: New Song\nArtist: AI Songwriter\n\n{VERSE_START}\n"
    elif prompt_type == "podcast":
        if custom_prompt:
            base_prompt = f"{PODCAST_START}\nHost: Welcome to Songwriting Insights. Today we're discussing {custom_prompt}\n"
        else:
            base_prompt = f"{PODCAST_START}\nHost: Welcome to Songwriting Insights. Today we're discussing songwriting techniques\n"
    elif prompt_type == "instagram":
        if custom_prompt:
            base_prompt = f"{INSTAGRAM_START}\nSongwriting tip about {custom_prompt}:\n"
        else:
            base_prompt = f"{INSTAGRAM_START}\nSongwriting tip of the day:\n"
    elif prompt_type == "custom" and custom_prompt:
        # For completely custom prompts
        return custom_prompt
    else:
        base_prompt = f"{SONG_START}\n"
        
    return base_prompt

def generate_text(model, tokenizer, prompt, args):
    """Generate text using the model."""
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Set seed for reproducibility if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Generate text
    output = model.generate(
        input_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_return_sequences=args.num_return_sequences,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated text
    generated_texts = []
    for i, generated_sequence in enumerate(output):
        text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
        generated_texts.append(text)
    
    return generated_texts

def main():
    """Main function."""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}. Please train a model first.")
        return
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    # Get prompt
    prompt = get_prompt(args.prompt_type, args.custom_prompt)
    logger.info(f"Using prompt: {prompt}")
    
    # Generate text
    logger.info("Generating text...")
    generated_texts = generate_text(model, tokenizer, prompt, args)
    
    # Print generated text
    logger.info("Generated text:")
    for i, text in enumerate(generated_texts):
        print(f"\n--- Generated Text {i+1} ---\n")
        print(text)
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
