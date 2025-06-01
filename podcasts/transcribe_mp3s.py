#!/usr/bin/env python3

import os
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
import whisper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("transcribe")

def transcribe_audio(audio_path, model_name="small", output_dir=None):
    """
    Transcribe an audio file using Whisper.
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model to use (tiny, base, small, medium, large)
        output_dir: Directory to save the transcript
        
    Returns:
        Path to the saved transcript JSON file
    """
    logger.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    logger.info(f"Transcribing: {audio_path}")
    result = model.transcribe(str(audio_path))
    
    # Create output filename
    audio_filename = Path(audio_path).stem
    if output_dir is None:
        output_dir = Path("transcripts")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{audio_filename}.json"
    
    # Save transcript
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved transcript to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3 files using Whisper")
    parser.add_argument("--input_dir", default="mp3s", help="Directory containing MP3 files")
    parser.add_argument("--output_dir", default="transcripts", help="Directory to save transcripts")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model to use")
    parser.add_argument("--files", help="Comma-separated list of specific files to transcribe (optional)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of MP3 files
    if args.files:
        mp3_files = [input_dir / filename for filename in args.files.split(",")]
    else:
        mp3_files = list(input_dir.glob("*.mp3"))
    
    logger.info(f"Found {len(mp3_files)} MP3 files to transcribe")
    
    # Process each file
    for mp3_path in tqdm(mp3_files, desc="Transcribing files"):
        if not mp3_path.exists():
            logger.warning(f"File not found: {mp3_path}")
            continue
            
        output_path = output_dir / f"{mp3_path.stem}.json"
        if output_path.exists():
            logger.info(f"Transcript already exists for {mp3_path.name}, skipping")
            continue
            
        try:
            transcribe_audio(mp3_path, args.model, output_dir)
        except Exception as e:
            logger.error(f"Error transcribing {mp3_path.name}: {e}")
    
    logger.info("Transcription complete!")

if __name__ == "__main__":
    main()
