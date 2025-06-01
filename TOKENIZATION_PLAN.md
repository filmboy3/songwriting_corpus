# Songwriting Corpus Tokenization Plan

## Overview

This document outlines the comprehensive tokenization strategy for the songwriting corpus project, which integrates multiple data sources including lyrics, chord progressions, podcast transcripts, and Instagram content. The goal is to create a unified tokenization approach that preserves the structure and specialized elements of songwriting while preparing the data for effective model training.

## Data Sources

The tokenization pipeline will process the following data sources:

1. **Song Lyrics and Chord Progressions**
   - 2,497+ lyrics files with associated chord progressions
   - Various artists, genres, and styles
   - Structured with verse, chorus, bridge sections

2. **Podcast Transcripts**
   - 328 podcast episodes (82+ completed, 25% of total)
   - Cleaned and normalized to remove repetitions and artifacts
   - Rich in songwriting advice, techniques, and industry insights

3. **Instagram Content**
   - 100+ posts from songwriting experts (Andrea Stolpe, Mel Robbins)
   - Normalized and cleaned text from captions and OCR
   - Structured songwriting techniques and advice

4. **Additional Resources**
   - Rhyming dictionary data
   - Imagery resources
   - Songwriting technique references

## Special Tokens

The tokenization will include the following special tokens:

### 1. Structural Tokens

```
<SONG_START>
<SONG_END>
<VERSE>
<CHORUS>
<BRIDGE>
<PRE_CHORUS>
<OUTRO>
<INTRO>
<HOOK>
```

### 2. Chord Tokens

```
<CHORD:Am>
<CHORD:C>
<CHORD:G>
<CHORD:F>
...
```

### 3. Source Tokens

```
<LYRICS>
<PODCAST>
<INSTAGRAM>
<TECHNIQUE>
```

### 4. Metadata Tokens

```
<ARTIST:name>
<GENRE:type>
<STYLE:description>
<TEMPO:bpm>
<KEY:key>
```

### 5. Technique Tokens

```
<TECHNIQUE:rhyme>
<TECHNIQUE:imagery>
<TECHNIQUE:storytelling>
<TECHNIQUE:melody>
<TECHNIQUE:harmony>
```

## Tokenization Process

### 1. Preprocessing

- **Text Normalization**: Standardize formatting, whitespace, and punctuation
- **Structure Preservation**: Maintain song structure with appropriate tokens
- **Chord Integration**: Insert chord tokens at appropriate positions
- **Metadata Enrichment**: Add relevant metadata tokens

### 2. Token Definition

- **Base Vocabulary**: Start with a standard vocabulary (e.g., GPT-2 tokenizer)
- **Special Token Addition**: Add the specialized tokens defined above
- **Frequency Analysis**: Analyze token frequency in the corpus
- **Vocabulary Optimization**: Optimize vocabulary size based on analysis

### 3. Tokenization Implementation

- **Hugging Face Tokenizers**: Use the Hugging Face tokenizers library
- **BPE Tokenization**: Apply Byte-Pair Encoding for efficient tokenization
- **Custom Tokenizer Training**: Train the tokenizer on the combined corpus
- **Tokenizer Saving**: Save the tokenizer configuration for model training

## File Format

The tokenized corpus will be stored in the following formats:

1. **Raw Text Format**: Preprocessed text with special tokens
   ```
   <SONG_START><ARTIST:Taylor Swift><GENRE:Pop>
   <VERSE>
   Lyrics for verse with <CHORD:G> at appropriate positions
   </VERSE>
   <CHORUS>
   Chorus lyrics with <CHORD:C> chord markers
   </CHORUS>
   <SONG_END>
   ```

2. **Tokenized Format**: Sequences of token IDs
   ```
   [1023, 5042, 8901, 2341, 5678, ...]
   ```

3. **Training Examples**: Formatted for model training
   ```
   {
     "input_ids": [1023, 5042, 8901, ...],
     "attention_mask": [1, 1, 1, ...],
     "labels": [1023, 5042, 8901, ...]
   }
   ```

## Implementation Plan

### Phase 1: Data Preparation (Current)

1. **Complete Data Collection**:
   - Finish podcast transcription (75% remaining)
   - Complete Instagram content scraping
   - Finalize lyrics and chord collection

2. **Data Cleaning and Normalization**:
   - Apply consistent formatting across all sources
   - Remove artifacts, repetitions, and irrelevant content
   - Structure data with appropriate section markers

### Phase 2: Tokenizer Development

1. **Token Definition**:
   - Create comprehensive list of special tokens
   - Define token format and structure
   - Document token usage guidelines

2. **Tokenizer Training**:
   - Prepare combined corpus for tokenizer training
   - Train custom BPE tokenizer
   - Evaluate tokenizer performance

3. **Tokenizer Integration**:
   - Integrate tokenizer with preprocessing pipeline
   - Create efficient tokenization workflow
   - Implement batch processing for large corpus

### Phase 3: Training Data Generation

1. **Format Conversion**:
   - Convert all corpus files to tokenized format
   - Create training examples with appropriate context windows
   - Split data into training, validation, and test sets

2. **Data Augmentation**:
   - Generate additional examples through permutations
   - Create specialized training examples for specific tasks
   - Balance representation across sources and styles

3. **Training Pipeline**:
   - Set up efficient data loading for training
   - Implement caching and optimization
   - Create evaluation metrics for model assessment

## Technical Challenges and Solutions

### Challenge 1: Control Token Truncation

**Problem**: Control tokens files getting truncated during processing.

**Solution**:
- Implement robust file handling with proper error checking
- Use streaming processing for large files
- Add checkpointing to prevent data loss
- Validate token files after generation

### Challenge 2: Balancing Source Representation

**Problem**: Ensuring balanced representation across different sources.

**Solution**:
- Implement weighted sampling during training
- Create source-specific evaluation metrics
- Monitor token distribution across sources
- Adjust preprocessing to normalize source contributions

### Challenge 3: Handling Long Context Windows

**Problem**: Podcast transcripts and song collections can exceed model context windows.

**Solution**:
- Implement intelligent chunking strategies
- Create overlapping context windows
- Use hierarchical tokenization for long documents
- Develop special handling for cross-document references

## Next Steps

1. **Complete Data Collection**: Finish podcast transcription and Instagram scraping
2. **Develop Tokenizer**: Implement custom tokenizer with special tokens
3. **Process Corpus**: Apply tokenization to all data sources
4. **Generate Training Data**: Create formatted training examples
5. **Set Up Training Pipeline**: Prepare for model training

## Resources and References

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- [BPE Tokenization Paper](https://arxiv.org/abs/1508.07909)
- [GPT-2 Tokenizer Implementation](https://github.com/openai/gpt-2)
- [Music Transformer Paper](https://arxiv.org/abs/1809.04281)
