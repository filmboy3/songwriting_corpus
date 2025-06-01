#!/usr/bin/env python3
"""
Script to train a songwriting assistant model using the collected corpus data.
This script trains a transformer-based model on lyrics, chord progressions,
and other songwriting resources.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_DIR = os.path.join(BASE_DIR, "combined")
LYRICS_DIR = os.path.join(COMBINED_DIR, "lyrics")
CHORDS_DIR = os.path.join(COMBINED_DIR, "chords")
ANALYSIS_DIR = os.path.join(COMBINED_DIR, "analysis")
REFERENCE_DIR = os.path.join(BASE_DIR, "music_theory_reference")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "model_output")

# Ensure output directory exists
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

class SongwritingDataset(Dataset):
    """Dataset for songwriting corpus training."""
    
    def __init__(self, tokenizer, corpus_dirs, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info("Loading corpus data...")
        
        # Load lyrics
        for artist_dir in os.listdir(LYRICS_DIR):
            artist_path = os.path.join(LYRICS_DIR, artist_dir)
            if os.path.isdir(artist_path):
                for song_file in os.listdir(artist_path):
                    if song_file.endswith('.json'):
                        song_path = os.path.join(artist_path, song_file)
                        try:
                            with open(song_path, 'r', encoding='utf-8') as f:
                                song_data = json.load(f)
                                
                                # Format the song data for training
                                song_text = self._format_song_data(song_data)
                                self.examples.append(song_text)
                        except Exception as e:
                            logger.error(f"Error loading song file {song_path}: {e}")
        
        # Load chord progressions and analysis
        for artist_dir in os.listdir(CHORDS_DIR):
            artist_path = os.path.join(CHORDS_DIR, artist_dir)
            if os.path.isdir(artist_path):
                for song_file in os.listdir(artist_path):
                    if song_file.endswith('_clean.json'):
                        chord_path = os.path.join(artist_path, song_file)
                        analysis_file = song_file.replace('_clean.json', '_analysis.json')
                        analysis_path = os.path.join(ANALYSIS_DIR, artist_dir, analysis_file)
                        
                        if os.path.exists(analysis_path):
                            try:
                                # Load chord data
                                with open(chord_path, 'r', encoding='utf-8') as f:
                                    chord_data = json.load(f)
                                
                                # Load analysis data
                                with open(analysis_path, 'r', encoding='utf-8') as f:
                                    analysis_data = json.load(f)
                                
                                # Format the chord and analysis data for training
                                chord_text = self._format_chord_data(chord_data, analysis_data)
                                self.examples.append(chord_text)
                            except Exception as e:
                                logger.error(f"Error loading chord/analysis files {chord_path}, {analysis_path}: {e}")
        
        # Load rhyming dictionary data
        rhyming_dir = os.path.join(REFERENCE_DIR, "rhyming_dictionary")
        if os.path.exists(rhyming_dir):
            for rhyme_file in os.listdir(rhyming_dir):
                if rhyme_file.endswith('_structured.json'):
                    rhyme_path = os.path.join(rhyming_dir, rhyme_file)
                    try:
                        with open(rhyme_path, 'r', encoding='utf-8') as f:
                            rhyme_data = json.load(f)
                            
                            # Format the rhyming data for training
                            rhyme_text = self._format_rhyme_data(rhyme_data)
                            self.examples.append(rhyme_text)
                    except Exception as e:
                        logger.error(f"Error loading rhyme file {rhyme_path}: {e}")
        
        # Load imagery resources
        imagery_dir = os.path.join(REFERENCE_DIR, "imagery_resources")
        if os.path.exists(imagery_dir):
            for imagery_file in os.listdir(imagery_dir):
                if imagery_file.endswith('_structured.json'):
                    imagery_path = os.path.join(imagery_dir, imagery_file)
                    try:
                        with open(imagery_path, 'r', encoding='utf-8') as f:
                            imagery_data = json.load(f)
                            
                            # Format the imagery data for training
                            imagery_text = self._format_imagery_data(imagery_data)
                            self.examples.append(imagery_text)
                    except Exception as e:
                        logger.error(f"Error loading imagery file {imagery_path}: {e}")
        
        logger.info(f"Loaded {len(self.examples)} examples for training")
        
        # Tokenize all examples
        self.encodings = tokenizer(
            self.examples,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    def _format_song_data(self, song_data):
        """Format song data for training."""
        artist = song_data.get('artist', 'Unknown Artist')
        title = song_data.get('title', 'Unknown Title')
        lyrics = song_data.get('lyrics', '')
        
        formatted_text = f"<|song|>\n<|artist|>{artist}<|/artist|>\n<|title|>{title}<|/title|>\n<|lyrics|>{lyrics}<|/lyrics|>\n<|/song|>"
        return formatted_text
    
    def _format_chord_data(self, chord_data, analysis_data):
        """Format chord and analysis data for training."""
        artist = chord_data.get('artist', 'Unknown Artist')
        title = chord_data.get('title', 'Unknown Title')
        key = analysis_data.get('key', 'Unknown')
        
        # Format chord progressions by section
        sections_text = ""
        sections = chord_data.get('sections', {})
        for section_name, section_data in sections.items():
            chords = section_data.get('chords', [])
            if chords:
                chords_str = " ".join(chords)
                sections_text += f"<|section|>{section_name}<|chords|>{chords_str}<|/chords|><|/section|>\n"
        
        # Format common progressions
        progressions_text = ""
        common_progressions = analysis_data.get('common_progressions', [])
        for prog in common_progressions:
            if isinstance(prog, dict):
                chords = prog.get('chords', [])
                roman = prog.get('roman', [])
                if chords and roman:
                    chords_str = " ".join(chords)
                    roman_str = " ".join(roman)
                    progressions_text += f"<|progression|><|chords|>{chords_str}<|/chords|><|roman|>{roman_str}<|/roman|><|/progression|>\n"
        
        formatted_text = f"<|chord_analysis|>\n<|artist|>{artist}<|/artist|>\n<|title|>{title}<|/title|>\n<|key|>{key}<|/key|>\n{sections_text}{progressions_text}<|/chord_analysis|>"
        return formatted_text
    
    def _format_rhyme_data(self, rhyme_data):
        """Format rhyming dictionary data for training."""
        formatted_text = "<|rhyming_dictionary|>\n"
        
        # Take a subset of words to avoid too large examples
        word_count = 0
        for word, rhymes in rhyme_data.items():
            if word_count >= 100:  # Limit to 100 words per example
                break
                
            if rhymes:
                rhymes_str = ", ".join(rhymes[:50])  # Limit to 50 rhymes per word
                formatted_text += f"<|word|>{word}<|rhymes|>{rhymes_str}<|/rhymes|><|/word|>\n"
                word_count += 1
        
        formatted_text += "<|/rhyming_dictionary|>"
        return formatted_text
    
    def _format_imagery_data(self, imagery_data):
        """Format imagery resource data for training."""
        formatted_text = "<|imagery_resource|>\n"
        
        items = imagery_data.get('items', [])
        # Take a subset of items to avoid too large examples
        for i, item in enumerate(items[:500]):  # Limit to 500 items per example
            formatted_text += f"<|item|>{item}<|/item|>\n"
        
        formatted_text += "<|/imagery_resource|>"
        return formatted_text
    
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings.input_ids[idx],
            'attention_mask': self.encodings.attention_mask[idx],
            'labels': self.encodings.input_ids[idx].clone()
        }

def train_model(args):
    """Train the songwriting model."""
    logger.info("Starting model training...")
    
    # Initialize tokenizer and add special tokens
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    special_tokens = {
        'additional_special_tokens': [
            '<|song|>', '<|/song|>', 
            '<|artist|>', '<|/artist|>', 
            '<|title|>', '<|/title|>', 
            '<|lyrics|>', '<|/lyrics|>',
            '<|chord_analysis|>', '<|/chord_analysis|>',
            '<|section|>', '<|/section|>',
            '<|chords|>', '<|/chords|>',
            '<|key|>', '<|/key|>',
            '<|progression|>', '<|/progression|>',
            '<|roman|>', '<|/roman|>',
            '<|rhyming_dictionary|>', '<|/rhyming_dictionary|>',
            '<|word|>', '<|/word|>',
            '<|rhymes|>', '<|/rhymes|>',
            '<|imagery_resource|>', '<|/imagery_resource|>',
            '<|item|>', '<|/item|>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize model
    if args.resume_from:
        logger.info(f"Resuming training from {args.resume_from}")
        model = GPT2LMHeadModel.from_pretrained(args.resume_from)
    else:
        logger.info(f"Initializing new model from {args.base_model}")
        config = GPT2Config.from_pretrained(args.base_model)
        model = GPT2LMHeadModel.from_pretrained(args.base_model, config=config)
    
    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Create dataset
    corpus_dirs = [LYRICS_DIR, CHORDS_DIR, ANALYSIS_DIR, REFERENCE_DIR]
    dataset = SongwritingDataset(tokenizer, corpus_dirs, max_length=args.max_length)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_OUTPUT_DIR, f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs'),
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train model
    logger.info("Training model...")
    trainer.train(resume_from_checkpoint=args.resume_from)
    
    # Save final model and tokenizer
    final_output_dir = os.path.join(MODEL_OUTPUT_DIR, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info(f"Model training complete. Final model saved to {final_output_dir}")

def main():
    """Main function to train the songwriting model."""
    parser = argparse.ArgumentParser(description="Train a songwriting assistant model")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume training from")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Maximum number of checkpoints to keep")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()
