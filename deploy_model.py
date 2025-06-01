#!/usr/bin/env python3
"""
Script to deploy the trained songwriting assistant model as a web service.
This allows accessing the model from anywhere via a REST API.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
from bs4 import BeautifulSoup
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("model_deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_output", "final")
API_CONFIG_FILE = os.path.join(BASE_DIR, "api_config.json")

# Load API configuration
with open(API_CONFIG_FILE, 'r') as f:
    API_CONFIG = json.load(f)

# Global variables
model = None
tokenizer = None
generation_queue = queue.Queue()
background_thread = None
is_running = True

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class SongwritingAssistant:
    """Songwriting Assistant class for generating lyrics, chord progressions, etc."""
    
    def __init__(self, model_path):
        """Initialize the songwriting assistant."""
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
    
    def generate_lyrics(self, prompt, max_length=200, temperature=0.8, top_p=0.9, top_k=50, num_return_sequences=1):
        """Generate lyrics based on a prompt."""
        # Format the prompt
        formatted_prompt = f"<|song|>\n{prompt}<|lyrics|>"
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            no_repeat_ngram_size=3
        )
        
        # Decode the generated text
        generated_texts = []
        for i in range(num_return_sequences):
            generated_text = self.tokenizer.decode(output[i], skip_special_tokens=False)
            # Extract only the lyrics part
            lyrics_start = generated_text.find("<|lyrics|>") + len("<|lyrics|>")
            lyrics_end = generated_text.find("<|/lyrics|>")
            if lyrics_end == -1:
                lyrics = generated_text[lyrics_start:]
            else:
                lyrics = generated_text[lyrics_start:lyrics_end]
            generated_texts.append(lyrics.strip())
        
        return generated_texts
    
    def generate_chord_progression(self, key, section_type, max_length=50, temperature=0.7):
        """Generate a chord progression for a specific key and section type."""
        # Format the prompt
        formatted_prompt = f"<|chord_analysis|>\n<|key|>{key}<|/key|>\n<|section|>{section_type}<|chords|>"
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract only the chords part
        chords_start = generated_text.find("<|chords|>") + len("<|chords|>")
        chords_end = generated_text.find("<|/chords|>")
        if chords_end == -1:
            chords = generated_text[chords_start:]
        else:
            chords = generated_text[chords_start:chords_end]
        
        # Split into individual chords
        chord_progression = chords.strip().split()
        
        return chord_progression
    
    def find_rhymes(self, word, num_rhymes=10):
        """Find rhymes for a given word."""
        # Format the prompt
        formatted_prompt = f"<|rhyming_dictionary|>\n<|word|>{word}<|rhymes|>"
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        output = self.model.generate(
            input_ids,
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract only the rhymes part
        rhymes_start = generated_text.find("<|rhymes|>") + len("<|rhymes|>")
        rhymes_end = generated_text.find("<|/rhymes|>")
        if rhymes_end == -1:
            rhymes = generated_text[rhymes_start:]
        else:
            rhymes = generated_text[rhymes_start:rhymes_end]
        
        # Split into individual rhymes
        rhyme_list = [r.strip() for r in rhymes.strip().split(",")]
        
        # Return the specified number of rhymes
        return rhyme_list[:num_rhymes]
    
    def suggest_imagery(self, theme, num_suggestions=5):
        """Suggest imagery based on a theme."""
        # Format the prompt
        formatted_prompt = f"<|imagery_resource|>\n<|item|>{theme}"
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        output = self.model.generate(
            input_ids,
            max_length=200,
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract imagery items
        items = []
        start_pos = 0
        while True:
            item_start = generated_text.find("<|item|>", start_pos)
            if item_start == -1:
                break
            
            item_start += len("<|item|>")
            item_end = generated_text.find("<|/item|>", item_start)
            
            if item_end == -1:
                break
            
            item = generated_text[item_start:item_end].strip()
            if item and item != theme:  # Don't include the original theme
                items.append(item)
            
            start_pos = item_end + len("<|/item|>")
            
            if len(items) >= num_suggestions:
                break
        
        return items

def fetch_song_info(artist, title):
    """Fetch song information from Genius API."""
    try:
        # Prepare the search query
        search_url = f"https://api.genius.com/search?q={artist} {title}"
        headers = {"Authorization": f"Bearer {API_CONFIG['genius_token']}"}
        
        # Make the API request
        response = requests.get(search_url, headers=headers)
        data = response.json()
        
        # Check if we got any hits
        if 'response' in data and 'hits' in data['response'] and data['response']['hits']:
            # Get the first hit
            hit = data['response']['hits'][0]
            song_id = hit['result']['id']
            song_url = hit['result']['url']
            
            # Fetch the song lyrics
            lyrics = fetch_lyrics_from_genius(song_url)
            
            return {
                'artist': hit['result']['primary_artist']['name'],
                'title': hit['result']['title'],
                'lyrics': lyrics,
                'url': song_url
            }
        
        return None
    except Exception as e:
        logger.error(f"Error fetching song info: {e}")
        return None

def fetch_lyrics_from_genius(url):
    """Fetch lyrics from a Genius song page."""
    try:
        # Make the request
        response = requests.get(url)
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the lyrics div
        lyrics_div = soup.find('div', class_='lyrics')
        if lyrics_div:
            return lyrics_div.get_text().strip()
        
        # Try alternative method if the above fails
        lyrics_containers = soup.find_all('div', class_=lambda c: c and 'Lyrics__Container' in c)
        if lyrics_containers:
            lyrics = '\n'.join([container.get_text() for container in lyrics_containers])
            return lyrics.strip()
        
        return "Lyrics not found"
    except Exception as e:
        logger.error(f"Error fetching lyrics: {e}")
        return "Error fetching lyrics"

def background_worker():
    """Background worker to process generation requests."""
    global is_running
    
    while is_running:
        try:
            # Get a task from the queue with a timeout
            task = generation_queue.get(timeout=1)
            
            # Process the task
            task_type = task['type']
            result = None
            
            if task_type == 'lyrics':
                result = model.generate_lyrics(
                    task['prompt'],
                    max_length=task.get('max_length', 200),
                    temperature=task.get('temperature', 0.8),
                    num_return_sequences=task.get('num_return_sequences', 1)
                )
            elif task_type == 'chord_progression':
                result = model.generate_chord_progression(
                    task['key'],
                    task['section_type'],
                    max_length=task.get('max_length', 50),
                    temperature=task.get('temperature', 0.7)
                )
            elif task_type == 'rhymes':
                result = model.find_rhymes(
                    task['word'],
                    num_rhymes=task.get('num_rhymes', 10)
                )
            elif task_type == 'imagery':
                result = model.suggest_imagery(
                    task['theme'],
                    num_suggestions=task.get('num_suggestions', 5)
                )
            
            # Store the result
            task['result'] = result
            task['status'] = 'completed'
            
            # Mark the task as done
            generation_queue.task_done()
        except queue.Empty:
            # No tasks in the queue, just continue
            continue
        except Exception as e:
            logger.error(f"Error in background worker: {e}")
            if 'task' in locals():
                task['status'] = 'error'
                task['error'] = str(e)
                generation_queue.task_done()

@app.route('/api/generate/lyrics', methods=['POST'])
def api_generate_lyrics():
    """API endpoint to generate lyrics."""
    data = request.json
    
    # Validate input
    if 'prompt' not in data:
        return jsonify({'error': 'Missing prompt parameter'}), 400
    
    # Create a task
    task_id = f"lyrics_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    task = {
        'id': task_id,
        'type': 'lyrics',
        'prompt': data['prompt'],
        'max_length': data.get('max_length', 200),
        'temperature': data.get('temperature', 0.8),
        'num_return_sequences': data.get('num_return_sequences', 1),
        'status': 'pending'
    }
    
    # Add the task to the queue
    generation_queue.put(task)
    
    # Return the task ID
    return jsonify({'task_id': task_id})

@app.route('/api/generate/chord_progression', methods=['POST'])
def api_generate_chord_progression():
    """API endpoint to generate a chord progression."""
    data = request.json
    
    # Validate input
    if 'key' not in data or 'section_type' not in data:
        return jsonify({'error': 'Missing key or section_type parameter'}), 400
    
    # Create a task
    task_id = f"chord_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    task = {
        'id': task_id,
        'type': 'chord_progression',
        'key': data['key'],
        'section_type': data['section_type'],
        'max_length': data.get('max_length', 50),
        'temperature': data.get('temperature', 0.7),
        'status': 'pending'
    }
    
    # Add the task to the queue
    generation_queue.put(task)
    
    # Return the task ID
    return jsonify({'task_id': task_id})

@app.route('/api/find/rhymes', methods=['POST'])
def api_find_rhymes():
    """API endpoint to find rhymes for a word."""
    data = request.json
    
    # Validate input
    if 'word' not in data:
        return jsonify({'error': 'Missing word parameter'}), 400
    
    # Create a task
    task_id = f"rhymes_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    task = {
        'id': task_id,
        'type': 'rhymes',
        'word': data['word'],
        'num_rhymes': data.get('num_rhymes', 10),
        'status': 'pending'
    }
    
    # Add the task to the queue
    generation_queue.put(task)
    
    # Return the task ID
    return jsonify({'task_id': task_id})

@app.route('/api/suggest/imagery', methods=['POST'])
def api_suggest_imagery():
    """API endpoint to suggest imagery based on a theme."""
    data = request.json
    
    # Validate input
    if 'theme' not in data:
        return jsonify({'error': 'Missing theme parameter'}), 400
    
    # Create a task
    task_id = f"imagery_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    task = {
        'id': task_id,
        'type': 'imagery',
        'theme': data['theme'],
        'num_suggestions': data.get('num_suggestions', 5),
        'status': 'pending'
    }
    
    # Add the task to the queue
    generation_queue.put(task)
    
    # Return the task ID
    return jsonify({'task_id': task_id})

@app.route('/api/fetch/song', methods=['POST'])
def api_fetch_song():
    """API endpoint to fetch song information."""
    data = request.json
    
    # Validate input
    if 'artist' not in data or 'title' not in data:
        return jsonify({'error': 'Missing artist or title parameter'}), 400
    
    # Fetch the song information
    song_info = fetch_song_info(data['artist'], data['title'])
    
    if song_info:
        return jsonify(song_info)
    else:
        return jsonify({'error': 'Song not found'}), 404

@app.route('/api/task/<task_id>', methods=['GET'])
def api_get_task(task_id):
    """API endpoint to get the status of a task."""
    # Find the task in the queue
    for task in list(generation_queue.queue):
        if task['id'] == task_id:
            return jsonify(task)
    
    # Task not found
    return jsonify({'error': 'Task not found'}), 404

def start_server(host='0.0.0.0', port=5000):
    """Start the Flask server."""
    global model, tokenizer, background_thread
    
    # Load the model
    model = SongwritingAssistant(MODEL_DIR)
    
    # Start the background worker
    background_thread = threading.Thread(target=background_worker)
    background_thread.daemon = True
    background_thread.start()
    
    # Start the Flask server
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port)

def main():
    """Main function to deploy the songwriting model."""
    parser = argparse.ArgumentParser(description="Deploy the songwriting assistant model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
