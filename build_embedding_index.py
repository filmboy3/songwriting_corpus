#!/usr/bin/env python3
"""
Script to build a FAISS embedding index for the songwriting corpus.
This enables semantic search across lyrics and chord progressions.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import faiss
from typing import List, Dict, Any, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_index.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("embedding_index")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_DIR = os.path.join(BASE_DIR, "combined")
LYRICS_DIR = os.path.join(COMBINED_DIR, "lyrics")
CHORDS_DIR = os.path.join(COMBINED_DIR, "chords")
ANALYSIS_DIR = os.path.join(COMBINED_DIR, "analysis")
INDEX_DIR = os.path.join(BASE_DIR, "embedding_index")
EMBEDDING_DIM = 1536  # OpenAI's text-embedding-3-small dimension

# Ensure the index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

def print_header(message):
    """Print a formatted header message."""
    header = f"\n{'=' * 80}\n  {message}\n{'=' * 80}\n"
    logger.info(header)
    return header

def load_openai():
    """Load the OpenAI API with proper error handling."""
    try:
        from openai import OpenAI
        
        # Check for API key in environment or config file
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            config_path = os.path.join(BASE_DIR, "api_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    api_key = config.get("openai_api_key")
        
        if not api_key:
            logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add it to api_config.json")
            return None
            
        client = OpenAI(api_key=api_key)
        return client
    except ImportError:
        logger.error("OpenAI package not installed. Please run: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        return None

def get_embedding(text: str, client, model="text-embedding-3-small") -> Optional[np.ndarray]:
    """Get embedding for a text using OpenAI API."""
    if not client:
        logger.error("OpenAI client not initialized")
        return None
        
    try:
        # Truncate text if too long (OpenAI has token limits)
        if len(text) > 8000:
            logger.warning(f"Text too long ({len(text)} chars), truncating to 8000 chars")
            text = text[:8000]
            
        response = client.embeddings.create(
            input=text,
            model=model
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def collect_song_data() -> List[Dict[str, Any]]:
    """Collect song data from the corpus, including lyrics and chord files."""
    print_header("COLLECTING SONG DATA")
    
    songs = []
    artists = [d for d in os.listdir(LYRICS_DIR) if os.path.isdir(os.path.join(LYRICS_DIR, d))]
    
    for artist in tqdm(artists, desc="Processing artists"):
        artist_dir = os.path.join(LYRICS_DIR, artist)
        for song_file in os.listdir(artist_dir):
            if song_file.endswith('.txt'):
                song_name = os.path.splitext(song_file)[0]
                
                # Get lyrics
                lyrics_path = os.path.join(artist_dir, song_file)
                with open(lyrics_path, 'r', encoding='utf-8', errors='replace') as f:
                    lyrics = f.read()
                
                # Check for chord file
                chord_file = os.path.join(CHORDS_DIR, artist, song_file)
                chords = None
                if os.path.exists(chord_file):
                    with open(chord_file, 'r', encoding='utf-8', errors='replace') as f:
                        chords = f.read()
                
                # Check for analysis file
                analysis_file = os.path.join(ANALYSIS_DIR, "chord_progressions", artist, f"{song_name}_chord_progressions.json")
                analysis = None
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r', encoding='utf-8', errors='replace') as f:
                        try:
                            analysis = json.load(f)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {analysis_file}")
                
                # Create song entry
                song = {
                    'artist': artist,
                    'title': song_name,
                    'lyrics': lyrics,
                    'lyrics_path': lyrics_path,
                }
                
                if chords:
                    song['chords'] = chords
                    song['chords_path'] = chord_file
                
                if analysis:
                    song['analysis'] = analysis
                    song['analysis_path'] = analysis_file
                
                # Extract sections if present
                sections = {}
                current_section = "default"
                section_lines = []
                
                for line in lyrics.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if this is a section header
                    if line.startswith('[') and line.endswith(']'):
                        if section_lines:
                            sections[current_section] = '\n'.join(section_lines)
                            section_lines = []
                        current_section = line[1:-1]  # Remove brackets
                    else:
                        section_lines.append(line)
                
                # Add the last section
                if section_lines:
                    sections[current_section] = '\n'.join(section_lines)
                
                if sections:
                    song['sections'] = sections
                
                songs.append(song)
    
    logger.info(f"Collected data for {len(songs)} songs from {len(artists)} artists")
    return songs

def build_index(songs: List[Dict[str, Any]], client) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Build a FAISS index from the song data."""
    print_header("BUILDING FAISS INDEX")
    
    # Initialize empty index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # Track which songs have embeddings
    indexed_songs = []
    embeddings = []
    
    # Process songs
    for song in tqdm(songs, desc="Embedding songs"):
        # Combine lyrics and chords if available
        text_to_embed = song['lyrics']
        if 'chords' in song:
            text_to_embed += f"\n\nChords:\n{song['chords']}"
        
        # Get embedding
        embedding = get_embedding(text_to_embed, client)
        if embedding is not None:
            embeddings.append(embedding)
            indexed_songs.append(song)
    
    # Add all embeddings to the index
    if embeddings:
        embeddings_array = np.array(embeddings, dtype=np.float32)
        index.add(embeddings_array)
        logger.info(f"Added {len(embeddings)} embeddings to the index")
    else:
        logger.warning("No embeddings were generated")
    
    return index, indexed_songs

def build_section_index(songs: List[Dict[str, Any]], client) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Build a FAISS index for song sections (verses, choruses, etc.)."""
    print_header("BUILDING SECTION INDEX")
    
    # Initialize empty index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # Track section data
    section_data = []
    embeddings = []
    
    # Process songs
    for song in tqdm(songs, desc="Embedding song sections"):
        if 'sections' not in song:
            continue
            
        for section_name, section_text in song['sections'].items():
            if len(section_text.strip()) < 10:  # Skip very short sections
                continue
                
            # Get embedding for this section
            embedding = get_embedding(section_text, client)
            if embedding is not None:
                embeddings.append(embedding)
                section_data.append({
                    'artist': song['artist'],
                    'title': song['title'],
                    'section_name': section_name,
                    'section_text': section_text,
                    'lyrics_path': song['lyrics_path']
                })
    
    # Add all embeddings to the index
    if embeddings:
        embeddings_array = np.array(embeddings, dtype=np.float32)
        index.add(embeddings_array)
        logger.info(f"Added {len(embeddings)} section embeddings to the index")
    else:
        logger.warning("No section embeddings were generated")
    
    return index, section_data

def save_index(index, metadata, filename):
    """Save the FAISS index and metadata."""
    index_path = os.path.join(INDEX_DIR, f"{filename}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{filename}_metadata.json")
    
    # Save the index
    faiss.write_index(index, index_path)
    logger.info(f"Saved index to {index_path}")
    
    # Save the metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {metadata_path}")

def search_index_demo(index, metadata, client, query="a song about love and heartbreak"):
    """Demonstrate searching the index."""
    print_header(f"SEARCH DEMO: '{query}'")
    
    # Get query embedding
    query_embedding = get_embedding(query, client)
    if query_embedding is None:
        logger.error("Failed to get embedding for query")
        return
    
    # Search the index
    k = 5  # Number of results to return
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    
    # Display results
    logger.info(f"Top {k} results for query: '{query}'")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(metadata):
            song = metadata[idx]
            logger.info(f"{i+1}. {song['artist']} - {song['title']} (Distance: {dist:.4f})")
            logger.info(f"   Excerpt: {song['lyrics'][:100]}...")
            logger.info("")

def main():
    """Main function to build the embedding index."""
    parser = argparse.ArgumentParser(description="Build a FAISS embedding index for the songwriting corpus")
    parser.add_argument("--sections", action="store_true", help="Build a separate index for song sections")
    parser.add_argument("--demo", action="store_true", help="Run a search demo after building the index")
    parser.add_argument("--query", type=str, default="a song about love and heartbreak", 
                        help="Query to use for the search demo")
    args = parser.parse_args()
    
    print_header("BUILDING EMBEDDING INDEX")
    
    # Initialize OpenAI client
    client = load_openai()
    if not client:
        logger.error("Failed to initialize OpenAI client. Exiting.")
        return 1
    
    # Collect song data
    songs = collect_song_data()
    if not songs:
        logger.error("No songs found in the corpus. Exiting.")
        return 1
    
    # Build song index
    song_index, indexed_songs = build_index(songs, client)
    
    # Save song index
    save_index(song_index, indexed_songs, "songs")
    
    # Build section index if requested
    if args.sections:
        section_index, section_data = build_section_index(songs, client)
        save_index(section_index, section_data, "sections")
    
    # Run search demo if requested
    if args.demo:
        search_index_demo(song_index, indexed_songs, client, args.query)
        
        if args.sections:
            print_header(f"SECTION SEARCH DEMO: '{args.query}'")
            search_index_demo(section_index, section_data, client, args.query)
    
    print_header("INDEX BUILDING COMPLETE")
    logger.info(f"Successfully built embedding index for {len(indexed_songs)} songs")
    if args.sections:
        logger.info(f"Also built section index with {len(section_data)} sections")
    logger.info(f"Indexes and metadata saved to {INDEX_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
