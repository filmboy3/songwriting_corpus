# Podcast Transcription and Model Fine-Tuning Pipeline

This directory contains a complete pipeline for transcribing podcasts using OpenAI's Whisper model, integrating the transcripts into a songwriting corpus, and fine-tuning language models on the processed data.

## Overview

The pipeline consists of the following steps:

1. Download MP3s from podcast RSS feeds
2. Transcribe the audio using Whisper
3. Clean and normalize transcripts with improved handling of repetitions
4. Extract songwriting-specific content and metadata
5. Format and integrate transcripts into the corpus with proper tagging
6. Generate quality reports and songwriting content analysis
7. Fine-tune language models (GPT-2, TinyLlama, Llama 3) on the processed data
8. Implement RAG (Retrieval-Augmented Generation) for enhanced content generation

## Directory Structure

- `mp3s/`: Downloaded podcast episodes
- `transcripts/`: Whisper JSON transcripts
- `processed/`: Cleaned and processed transcripts
- `corpus/`: Final transcripts integrated into the corpus

## Scripts

### Core Pipeline Scripts

- `rss_to_mp3_downloader.py`: Downloads podcast episodes from RSS feeds with improved error handling and filename sanitization
- `improved_transcript_cleaner.py`: Enhanced transcript cleaning with better handling of repetitive phrases and sponsor messages
- `songwriting_corpus_integrator.py`: Extracts songwriting content and integrates transcripts into the corpus with metadata
- `enhanced_workflow.py`: Unified workflow script that ties everything together

### Utility Scripts

- `check_transcription_progress.py`: Checks the progress of the transcription process
- `debug_rss.py`: Helps debug issues with RSS feed parsing
- `complete_transcript_workflow.py`: Original transcript workflow (maintained for compatibility)

## Features

### Improved RSS Downloader
- `integrate_whisper_transcripts.py` - Formats and integrates transcripts into the corpus
- `complete_transcript_workflow.py` - End-to-end pipeline combining all processing steps

### Analysis

- `analyze_transcript_quality.py` - Analyzes transcript quality and generates reports

## Llama 3 Fine-Tuning

This directory includes a comprehensive fine-tuning pipeline for Llama 3 models optimized for Mac MPS hardware:

### Key Scripts

- `train_llama3_podcast.py` - Main fine-tuning script with PEFT/LoRA for memory efficiency
- `llama3_rag_generator.py` - RAG-enhanced generation using fine-tuned models
- `authenticate_llama3.py` - Authentication with Hugging Face for accessing gated models

### Memory Optimization Techniques

- **PEFT/LoRA Adapters**: Fine-tunes only ~0.02% of model parameters
- **Gradient Checkpointing**: Reduces memory usage during backpropagation
- **Gradient Accumulation**: Simulates larger batch sizes with small memory footprint
- **Special Mac MPS Loading**: Loads model on CPU first, then moves trainable parameters to MPS
- **Reduced Sequence Length**: Balances context window vs. memory usage
- **Environmental Variables**: Sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to maximize available memory

### Data Processing

- Formats all data sources into instruction-following examples
- Combines podcast transcripts, lyrics, chord analysis, music theory, and songwriting techniques
- Properly tokenizes and prepares datasets for Llama 3 fine-tuning

### Training Configuration

- **Model**: meta-llama/Meta-Llama-3-8B
- **LoRA Rank**: 4 (reduced from 8 for Mac MPS compatibility)
- **Batch Size**: 1 with gradient accumulation of 16
- **Learning Rate**: 2e-4
- **Sequence Length**: 256 (optimized for memory efficiency)
- **Training Epochs**: 3

## Usage

### Check Transcription Progress

```bash
python podcasts/check_transcription_progress.py
```

### Convert JSON to Text

```bash
python podcasts/batch_json_to_text.py --json-dir podcasts/transcripts --output-dir podcasts/text_transcripts
```

### Clean Transcripts

```bash
python podcasts/clean_transcripts.py --input-dir podcasts/text_transcripts --output-dir podcasts/cleaned_transcripts
```

### Complete Workflow

```bash
python podcasts/complete_transcript_workflow.py --json-dir podcasts/transcripts --output-dir combined/corpus/transcripts
```

## Transcript Quality

The Whisper tiny model produces good quality transcripts with the following characteristics:

- Average length: ~10,000 words per transcript
- Common issues addressed by cleaning scripts:
  - Repetitive phrases (removed)
  - Sponsor messages (replaced with placeholders)
  - Inconsistent formatting (normalized)
  - Speaker labels (standardized or removed if inconsistent)

## Corpus Integration

Transcripts are formatted for the corpus with the following structure:

```
<PODCAST>
<TITLE>Podcast Name</TITLE>

<TRANSCRIPT>
Cleaned transcript content...
</TRANSCRIPT>
</PODCAST>
```

## Current Status

As of May 31, 2025:
- Total MP3 files: 328
- Completed transcripts: 82 (25%)
- Estimated completion time: ~3.7 hours
