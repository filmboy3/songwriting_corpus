import os
import json
import logging
import datetime
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uuid
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.urandom(24)  # For session management

# Default model paths - can be overridden with environment variables
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'model_output'))
DEFAULT_MEDIUM_MODEL_PATH = os.environ.get('MEDIUM_MODEL_PATH', os.path.join(BASE_DIR, 'model_output_medium'))

# Global variables to store models and tokenizers
models = {}
tokenizers = {}
current_model_path = DEFAULT_MODEL_PATH

# Demo mode flag - will be set to True if no models are available
DEMO_MODE = False

# Maximum conversation history to maintain
MAX_HISTORY_LENGTH = 10  # Number of exchanges to keep
MAX_CONTEXT_LENGTH = 2048  # Maximum tokens in context

# Intent categories for router
INTENTS = {
    'LYRICS': ['write', 'lyrics', 'song', 'verse', 'chorus', 'compose', 'words'],
    'CHORDS': ['chord', 'progression', 'key', 'harmony', 'chords', 'music'],
    'STRUCTURE': ['structure', 'arrangement', 'form', 'bridge', 'section', 'intro', 'outro'],
    'STYLE': ['style', 'genre', 'like', 'similar', 'sound like', 'vibe'],
    'REVISION': ['revise', 'change', 'edit', 'rewrite', 'modify', 'improve', 'fix'],
    'ANALYSIS': ['analyze', 'explain', 'breakdown', 'understand', 'meaning', 'interpretation']
}

def load_model(model_path):
    """Load model and tokenizer from the specified path"""
    global models, tokenizers, current_model_path
    
    if model_path in models:
        logger.info(f"Using cached model from {model_path}")
        return models[model_path], tokenizers[model_path]
    
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Check if the model path exists
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Check if the path contains the necessary model files
        if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) and \
           not os.path.exists(os.path.join(model_path, 'model.safetensors')):
            logger.error(f"No model files found in {model_path}")
            raise FileNotFoundError(f"No model files found in {model_path}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Cache the loaded model and tokenizer
        models[model_path] = model
        tokenizers[model_path] = tokenizer
        current_model_path = model_path
        
        logger.info(f"Successfully loaded model from {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def detect_intent(prompt):
    """Detect the intent of the user's prompt"""
    prompt_lower = prompt.lower()
    scores = {}
    
    for intent, keywords in INTENTS.items():
        score = sum(1 for keyword in keywords if keyword in prompt_lower)
        scores[intent] = score
    
    # Get the intent with the highest score
    max_intent = max(scores, key=scores.get)
    
    # If no keywords were found, default to LYRICS
    if scores[max_intent] == 0:
        return 'LYRICS'
    
    return max_intent

def format_prompt_based_on_intent(prompt, intent, conversation_history):
    """Format the prompt based on detected intent and conversation history"""
    
    # Extract the last song or chord progression from history if it exists
    last_song = ""
    last_chords = ""
    
    for exchange in reversed(conversation_history):
        if "<|song|>" in exchange["response"]:
            last_song = exchange["response"]
            break
    
    for exchange in reversed(conversation_history):
        if "Chord Progression:" in exchange["response"]:
            last_chords = exchange["response"]
            break
    
    # Format based on intent
    if intent == 'LYRICS':
        if "counting crows" in prompt.lower():
            formatted_prompt = f"<|song|>\nTitle: Counting Crows Style Song\nArtist: AI Songwriter\n\n<|verse|>\n"
        else:
            formatted_prompt = f"<|song|>\nTitle: Song About {prompt.split('about')[-1].strip() if 'about' in prompt else prompt}\nArtist: AI Songwriter\n\n<|verse|>\n"
    
    elif intent == 'CHORDS':
        if last_song:
            formatted_prompt = f"Based on these lyrics:\n{last_song}\n\nHere are suitable chord progressions:\nChord Progression:"
        else:
            formatted_prompt = f"Suggest chord progressions for a song about {prompt}:\nChord Progression:"
    
    elif intent == 'STRUCTURE':
        if "bridge" in prompt.lower() and last_song:
            formatted_prompt = f"Add a bridge to this song:\n{last_song}\n\n<|bridge|>\n"
        else:
            formatted_prompt = f"Create a song structure for {prompt}:\nSong Structure:"
    
    elif intent == 'REVISION':
        if last_song:
            formatted_prompt = f"Revise these lyrics based on this feedback: {prompt}\n\nOriginal lyrics:\n{last_song}\n\nRevised lyrics:\n<|song|>\n"
        else:
            formatted_prompt = f"<|song|>\nTitle: {prompt}\nArtist: AI Songwriter\n\n<|verse|>\n"
    
    elif intent == 'STYLE':
        style_match = re.search(r'like\s+([^,\.]+)', prompt.lower())
        artist = style_match.group(1) if style_match else prompt.split()[-1]
        formatted_prompt = f"<|song|>\nTitle: Song in the style of {artist}\nArtist: AI Songwriter\n\n<|verse|>\n"
    
    else:  # Default or ANALYSIS
        if last_song:
            formatted_prompt = f"Analyze these lyrics:\n{last_song}\n\nAnalysis:"
        else:
            formatted_prompt = f"Write a song about {prompt} and then analyze its themes:\n<|song|>\n"
    
    return formatted_prompt

def generate_demo_response(prompt, intent, conversation_history):
    """Generate a demo response when models are not available"""
    logger.info(f"Generating demo response for intent: {intent}")
    
    # Extract themes from the prompt
    themes = []
    for word in prompt.lower().split():
        if len(word) > 3 and word not in ['about', 'with', 'that', 'this', 'like', 'song', 'write', 'create']:
            themes.append(word)
    
    theme = themes[0] if themes else "music"
    
    # Check if we have any previous songs in the conversation
    last_song = ""
    for exchange in reversed(conversation_history):
        if "### Verse" in exchange.get("response", ""):
            last_song = exchange["response"]
            break
    
    # Generate responses based on intent
    if intent == 'LYRICS':
        return f"### Verse\nHere's where I'd generate lyrics about {theme}.\nThe trained model will create original verses\nWith rhymes and imagery based on your request.\n\n### Chorus\nThis is just a placeholder,\nUntil your model finishes training.\nSoon you'll have AI-generated songs,\nThat match your creative styling.\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"
    
    elif intent == 'CHORDS':
        if last_song:
            return f"Based on the lyrics you provided, here are some chord suggestions:\n\nVerse: `C` `Am` `F` `G`\nChorus: `F` `C` `G` `Am`\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"
        else:
            return f"For a song about {theme}, I suggest these chord progressions:\n\nVerse: `C` `G` `Am` `F`\nChorus: `F` `C` `G` `G`\nBridge: `Am` `F` `C` `G`\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"
    
    elif intent == 'STRUCTURE':
        return f"Here's a suggested structure for a song about {theme}:\n\n1. Intro (4 bars)\n2. Verse 1 (8 bars)\n3. Chorus (8 bars)\n4. Verse 2 (8 bars)\n5. Chorus (8 bars)\n6. Bridge (4 bars)\n7. Chorus (8 bars)\n8. Outro (4 bars)\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"
    
    elif intent == 'REVISION':
        if last_song:
            return f"Here's how I would revise the previous lyrics:\n\n{last_song.replace('This is just a placeholder', 'This revised chorus has more impact')}\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"
        else:
            return f"I can help revise lyrics once you've generated some. Try asking me to write lyrics first.\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"
    
    elif intent == 'STYLE':
        style_match = re.search(r'like\s+([^,\.]+)', prompt.lower())
        artist = style_match.group(1) if style_match else theme
        return f"### Verse\nThis would be a verse in the style of {artist}\nWith characteristic phrasing and themes\nThat match their unique approach\nTo songwriting and expression\n\n### Chorus\nAnd here's where the chorus would go\nIn that distinctive {artist} style\nWith their typical chord progressions\nAnd vocal delivery\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"
    
    else:  # ANALYSIS or default
        if last_song:
            return f"Analysis of the previous lyrics:\n\nThemes: The lyrics explore concepts of {theme} and self-discovery.\nStructure: Standard verse-chorus format with a bridge that provides contrast.\nImagery: Uses natural elements as metaphors for emotional states.\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"
        else:
            return f"I can analyze lyrics once you've generated some. Try asking me to write lyrics first.\n\n*Note: This is demo mode. Your models are still training. Check back soon!*"

def generate_response(prompt, conversation_history, model_path=None):
    """Generate a response based on the prompt and conversation history"""
    global DEMO_MODE
    
    if not model_path:
        model_path = current_model_path
    
    # Detect intent first (this doesn't require the model)
    intent = detect_intent(prompt)
    logger.info(f"Detected intent: {intent}")
    
    # Format prompt based on intent and history
    formatted_prompt = format_prompt_based_on_intent(prompt, intent, conversation_history)
    logger.info(f"Formatted prompt: {formatted_prompt[:100]}...")
    
    # If we're in demo mode, use the demo response generator
    if DEMO_MODE:
        logger.info("Using demo mode response generator")
        return generate_demo_response(prompt, intent, conversation_history)
    
    # Try to load the model
    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        
        # Check if we have any fallback models loaded
        if models:
            # Use the first available model
            fallback_path = list(models.keys())[0]
            logger.info(f"Using fallback model from {fallback_path}")
            model = models[fallback_path]
            tokenizer = tokenizers[fallback_path]
        else:
            # No models available, switch to demo mode
            DEMO_MODE = True
            logger.warning("No models available, switching to demo mode")
            return generate_demo_response(prompt, intent, conversation_history)
    
    # Generate text
    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        
        # Set generation parameters based on intent
        if intent == 'LYRICS':
            temperature = 0.8
            max_length = 300
            top_p = 0.92
        elif intent == 'CHORDS':
            temperature = 0.5
            max_length = 150
            top_p = 0.85
        else:
            temperature = 0.7
            max_length = 250
            top_p = 0.9
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Clean up the generated text
        generated_text = generated_text.replace(formatted_prompt, "")
        generated_text = re.sub(r'<\|endoftext\|>.*', '', generated_text)
        
        return generated_text.strip()
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        
        # Provide more helpful information based on the error
        if "CUDA out of memory" in str(e):
            return ("Sorry, the model is too large for your current GPU memory. "
                   "Try using the smaller GPT-2 base model instead of the medium model.")
        elif "device-side assert" in str(e):
            return ("There was a problem with the input format. This might be due to incompatible "
                   "tokenization between your training data and the current prompt.")
        else:
            return f"Sorry, I encountered an error: {str(e)}"

@app.route('/')
def index():
    """Render the main page"""
    # Initialize session if needed
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat interactions"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model_choice = data.get('model', 'base')  # 'base' or 'medium'
        
        if not prompt.strip():
            return jsonify({
                "error": "Please provide a prompt.",
                "status": "error"
            }), 400
        
        # Get conversation history from session or initialize
        conversation_id = session.get('conversation_id', str(uuid.uuid4()))
        session['conversation_id'] = conversation_id
        
        # Load conversation history from file or initialize
        history_file = f"conversations/{conversation_id}.json"
        os.makedirs("conversations", exist_ok=True)
        
        conversation_history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    conversation_history = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in conversation history file: {history_file}")
                # Continue with empty history
                conversation_history = []
        
        # Select model path based on user choice
        if model_choice == 'medium' and os.path.exists(DEFAULT_MEDIUM_MODEL_PATH):
            model_path = DEFAULT_MEDIUM_MODEL_PATH
            logger.info(f"Using medium model from {model_path}")
        else:
            model_path = DEFAULT_MODEL_PATH
            logger.info(f"Using base model from {model_path}")
        
        # Check if model path exists
        if not os.path.exists(model_path):
            error_msg = f"Model path does not exist: {model_path}"
            logger.error(error_msg)
            return jsonify({
                "response": f"Error: {error_msg}. Please make sure your models are trained and available.",
                "conversation_id": conversation_id,
                "status": "error"
            })
        
        # Generate response
        response = generate_response(prompt, conversation_history, model_path)
        
        # Update conversation history
        conversation_history.append({
            "prompt": prompt,
            "response": response,
            "model": model_choice,
            "timestamp": str(datetime.datetime.now())
        })
        
        # Limit history length
        if len(conversation_history) > MAX_HISTORY_LENGTH:
            conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]
        
        # Save updated history
        try:
            with open(history_file, 'w') as f:
                json.dump(conversation_history, f)
        except Exception as e:
            logger.error(f"Error saving conversation history: {str(e)}")
        
        return jsonify({
            "response": response,
            "conversation_id": conversation_id,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "response": f"Sorry, I encountered an error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/new_conversation', methods=['POST'])
def new_conversation():
    """Start a new conversation"""
    session['conversation_id'] = str(uuid.uuid4())
    return jsonify({"conversation_id": session['conversation_id']})

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models"""
    available_models = []
    
    if os.path.exists(DEFAULT_MODEL_PATH):
        available_models.append({
            "id": "base",
            "name": "GPT-2 Base",
            "path": DEFAULT_MODEL_PATH
        })
    
    if os.path.exists(DEFAULT_MEDIUM_MODEL_PATH):
        available_models.append({
            "id": "medium",
            "name": "GPT-2 Medium",
            "path": DEFAULT_MEDIUM_MODEL_PATH
        })
    
    return jsonify({"models": available_models})

if __name__ == '__main__':
    # Preload the base model
    try:
        load_model(DEFAULT_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Could not preload base model: {str(e)}")
        # Set demo mode since models are not available
        DEMO_MODE = True
        logger.info("Starting in demo mode since models are not available")
    
    # Update the welcome message to indicate demo mode if active
    if DEMO_MODE:
        logger.info("Running in demo mode - models are not available")
    
    app.run(debug=True, host='0.0.0.0', port=8889)
