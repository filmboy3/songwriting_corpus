#!/usr/bin/env python3
"""
Train a Songwriting Language Model

This script trains a language model on the prepared songwriting corpus using
Hugging Face's transformers library. It uses a pre-trained model as a starting
point and fine-tunes it on the songwriting corpus.

The script avoids complex tokenization by using a pre-trained tokenizer.
"""

import os
import json
import logging
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = BASE_DIR / "training_ready" / "combined"
OUTPUT_DIR = BASE_DIR / "model_output"

def parse_args():
    parser = argparse.ArgumentParser(description="Train a songwriting language model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Pre-trained model to use as base (e.g., gpt2, gpt2-medium, EleutherAI/pythia-70m)",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(CORPUS_DIR / "train.jsonl"),
        help="Path to training data file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=str(CORPUS_DIR / "val.jsonl"),
        help="Path to validation data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save the model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    return parser.parse_args()

class SongwritingDataset(Dataset):
    """Dataset for songwriting corpus."""
    
    def __init__(self, file_path, tokenizer, max_length):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading dataset from {file_path}")
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Loading dataset"):
                example = json.loads(line)
                self.examples.append(example["text"])
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone()
        }

def train_model():
    """Train the language model on the songwriting corpus."""
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pre-trained model and tokenizer
    logger.info(f"Loading pre-trained model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Set padding token (required for batch processing)
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token since it was not set")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens if they don't exist
    special_tokens = [
        "<|song|>", "</|song|>",
        "<|podcast|>", "</|podcast|>",
        "<|instagram|>", "</|instagram|>",
        "<|verse|>", "</|verse|>",
        "<|chorus|>", "</|chorus|>",
        "<|bridge|>", "</|bridge|>",
        "<|section|>", "</|section|>"
    ]
    
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} special tokens to the tokenizer")
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Load datasets
    train_dataset = SongwritingDataset(args.train_file, tokenizer, args.max_length)
    val_dataset = SongwritingDataset(args.val_file, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Simplified arguments for compatibility
        do_eval=True,
        eval_steps=args.save_steps
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")

def main():
    """Main function."""
    logger.info("Starting songwriting model training...")
    
    # Check if corpus exists
    train_file = CORPUS_DIR / "train.jsonl"
    val_file = CORPUS_DIR / "val.jsonl"
    
    if not train_file.exists() or not val_file.exists():
        logger.error("Training corpus not found. Please run prepare_training_corpus.py first.")
        return
    
    # Train the model
    train_model()

if __name__ == "__main__":
    main()
