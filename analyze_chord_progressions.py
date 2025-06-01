#!/usr/bin/env python3
"""
Script to analyze chord progressions in songs and identify their relationship
to common chord progressions and the Nashville Number System.
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

# Constants
CORPUS_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_CHORDS_DIR = os.path.join(CORPUS_DIR, "clean_chords")
REFERENCE_DIR = os.path.join(CORPUS_DIR, "music_theory_reference")
ANALYSIS_DIR = os.path.join(CORPUS_DIR, "chord_analysis")

def log_message(message):
    """Print a log message with timestamp."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {message}")

def ensure_directory(directory):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory

def load_reference_data():
    """Load music theory reference data."""
    reference_data = {}
    reference_files = [
        "key_chord_progressions.json",
        "nashville_number_system.json",
        "circle_of_fifths.json",
        "chord_formulas.json",
        "common_progressions.json"
    ]
    
    for filename in reference_files:
        file_path = os.path.join(REFERENCE_DIR, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                reference_data[os.path.splitext(filename)[0]] = json.load(f)
        else:
            log_message(f"Warning: Reference file not found: {file_path}")
    
    return reference_data

def detect_key(chords):
    """
    Detect the most likely key of a song based on its chords.
    Returns a tuple of (key, is_major).
    """
    reference_data = load_reference_data()
    key_progressions = reference_data.get("key_chord_progressions", {})
    
    if not key_progressions:
        return None, None
    
    # Count occurrences of each chord
    chord_counts = {}
    for chord in chords:
        base_chord = re.sub(r'(maj|min|dim|aug|sus|add|\d+|\+|-)', '', chord).strip()
        chord_counts[base_chord] = chord_counts.get(base_chord, 0) + 1
    
    # Score each possible key
    key_scores = {}
    
    # Check major keys
    for key, progression in key_progressions.get("major_keys", {}).items():
        score = 0
        for chord in progression:
            base_chord = re.sub(r'°', '', chord).strip()  # Remove diminished symbol
            if base_chord in chord_counts:
                # Weight I, IV, V chords more heavily
                if chord == progression[0]:  # I chord
                    score += chord_counts[base_chord] * 3
                elif chord == progression[3]:  # IV chord
                    score += chord_counts[base_chord] * 2
                elif chord == progression[4]:  # V chord
                    score += chord_counts[base_chord] * 2
                else:
                    score += chord_counts[base_chord]
        key_scores[(key, True)] = score
    
    # Check minor keys
    for key, progression in key_progressions.get("minor_keys", {}).items():
        score = 0
        for chord in progression:
            base_chord = re.sub(r'°', '', chord).strip()  # Remove diminished symbol
            if base_chord in chord_counts:
                # Weight i, iv, v chords more heavily
                if chord == progression[0]:  # i chord
                    score += chord_counts[base_chord] * 3
                elif chord == progression[3]:  # iv chord
                    score += chord_counts[base_chord] * 2
                elif chord == progression[4]:  # v chord
                    score += chord_counts[base_chord] * 2
                else:
                    score += chord_counts[base_chord]
        key_scores[(key, False)] = score
    
    # Find the key with the highest score
    if not key_scores:
        return None, None
    
    best_key, is_major = max(key_scores.items(), key=lambda x: x[1])[0]
    return best_key, is_major

def get_chord_degree(chord, key, is_major):
    """
    Get the scale degree (1-7) of a chord in a given key.
    Returns a tuple of (degree, accidental, chord_type).
    """
    # Define the major and minor scales
    major_scale_notes = {
        'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
        'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
        'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
        'A': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
        'E': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
        'B': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],
        'F#': ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'E#'],
        'Gb': ['Gb', 'Ab', 'Bb', 'Cb', 'Db', 'Eb', 'F'],
        'Db': ['Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb', 'C'],
        'Ab': ['Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'],
        'Eb': ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
        'Bb': ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
        'F': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E']
    }
    
    minor_scale_notes = {
        'Am': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'Em': ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
        'Bm': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],
        'F#m': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'],
        'C#m': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'],
        'G#m': ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#'],
        'D#m': ['D#', 'E#', 'F#', 'G#', 'A#', 'B', 'C#'],
        'Ebm': ['Eb', 'F', 'Gb', 'Ab', 'Bb', 'Cb', 'Db'],
        'Bbm': ['Bb', 'C', 'Db', 'Eb', 'F', 'Gb', 'Ab'],
        'Fm': ['F', 'G', 'Ab', 'Bb', 'C', 'Db', 'Eb'],
        'Cm': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],
        'Gm': ['G', 'A', 'Bb', 'C', 'D', 'Eb', 'F'],
        'Dm': ['D', 'E', 'F', 'G', 'A', 'Bb', 'C']
    }
    
    # Extract the root note of the chord
    root = re.match(r'([A-G][b#]?)', chord)
    if not root:
        return None, None, None
    
    root = root.group(1)
    
    # Determine chord type
    if 'dim' in chord or '°' in chord:
        chord_type = 'dim'
    elif 'm' in chord or 'min' in chord:
        chord_type = 'min'
    elif 'aug' in chord or '+' in chord:
        chord_type = 'aug'
    elif 'sus' in chord:
        chord_type = 'sus'
    else:
        chord_type = 'maj'
    
    # Get the scale notes for the key
    scale_notes = minor_scale_notes.get(key, []) if not is_major else major_scale_notes.get(key, [])
    if not scale_notes:
        return None, None, None
    
    # Find the degree of the root note in the scale
    degree = None
    accidental = None
    
    # First check for exact match
    if root in scale_notes:
        degree = scale_notes.index(root) + 1
    else:
        # Check for enharmonic equivalents
        enharmonics = {
            'C#': 'Db', 'Db': 'C#',
            'D#': 'Eb', 'Eb': 'D#',
            'F#': 'Gb', 'Gb': 'F#',
            'G#': 'Ab', 'Ab': 'G#',
            'A#': 'Bb', 'Bb': 'A#',
            'E#': 'F', 'F': 'E#',
            'B#': 'C', 'C': 'B#',
            'Cb': 'B', 'B': 'Cb'
        }
        
        enharmonic = enharmonics.get(root)
        if enharmonic and enharmonic in scale_notes:
            degree = scale_notes.index(enharmonic) + 1
        else:
            # Check for flattened or sharpened notes
            for i, note in enumerate(scale_notes):
                if note[0] == root[0]:  # Same letter
                    degree = i + 1
                    if '#' in root and '#' not in note:
                        accidental = '#'
                    elif 'b' in root and 'b' not in note:
                        accidental = 'b'
                    elif '#' not in root and '#' in note:
                        accidental = 'b'  # Root is natural, scale note is sharp
                    elif 'b' not in root and 'b' in note:
                        accidental = '#'  # Root is natural, scale note is flat
                    break
    
    return degree, accidental, chord_type

def convert_to_roman_numerals(chords, key, is_major):
    """
    Convert a chord progression to Roman numerals based on the detected key.
    Handles non-diatonic chords better by analyzing the relationship to the key.
    """
    if not key or not chords:
        return []
    
    # Define the Roman numeral templates
    major_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    minor_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
    
    # Convert each chord to its Roman numeral
    roman_numerals = []
    for chord in chords:
        # Skip non-chord entries
        if not re.match(r'[A-G][b#]?.*', chord):
            roman_numerals.append(chord)  # Keep as is (might be a section marker)
            continue
        
        # Get the degree, accidental, and type of the chord
        degree, accidental, chord_type = get_chord_degree(chord, key, is_major)
        
        if degree is None:
            roman_numerals.append(chord)  # Keep the original chord if we can't analyze it
            continue
        
        # Get the base Roman numeral
        if is_major:
            base_numeral = major_numerals[degree - 1]
            # Adjust for minor, diminished, augmented
            if chord_type == 'min':
                base_numeral = base_numeral.lower()
            elif chord_type == 'dim':
                base_numeral = base_numeral.lower() + '°'
            elif chord_type == 'aug':
                base_numeral = base_numeral + '+'
        else:  # Minor key
            base_numeral = minor_numerals[degree - 1]
            # Adjust for major, diminished, augmented
            if chord_type == 'maj' and degree not in [3, 6, 7]:  # III, VI, VII are naturally major
                base_numeral = base_numeral.upper()
            elif chord_type == 'dim':
                base_numeral = base_numeral + '°'
            elif chord_type == 'aug':
                base_numeral = base_numeral + '+'
        
        # Add accidental if present
        if accidental:
            base_numeral = accidental + base_numeral
        
        # Add extensions
        if '7' in chord:
            base_numeral += '7'
        elif '9' in chord:
            base_numeral += '9'
        elif 'sus4' in chord:
            base_numeral += 'sus4'
        elif 'sus2' in chord:
            base_numeral += 'sus2'
        elif 'sus' in chord:
            base_numeral += 'sus'
        
        roman_numerals.append(base_numeral)
    
    return roman_numerals

def identify_common_progressions(roman_numerals):
    """
    Identify common chord progressions within a sequence of Roman numerals.
    """
    reference_data = load_reference_data()
    common_progs = reference_data.get("common_progressions", {})
    
    if not common_progs or not roman_numerals:
        return []
    
    # Join the Roman numerals into a string for easier pattern matching
    progression_str = "-".join([re.sub(r'\d+|sus', '', numeral) for numeral in roman_numerals])
    
    # Check for common progressions
    found_progressions = []
    
    # Check major key progressions
    for prog, desc in common_progs.get("major_key", {}).items():
        if prog in progression_str:
            found_progressions.append((prog, desc, "major"))
    
    # Check minor key progressions
    for prog, desc in common_progs.get("minor_key", {}).items():
        if prog in progression_str:
            found_progressions.append((prog, desc, "minor"))
    
    return found_progressions

def analyze_chord_file(chord_file):
    """
    Analyze a chord file and identify key, chord progressions, and Roman numerals.
    """
    try:
        # Check if there's a JSON file with extracted chord progressions
        json_file = os.path.splitext(chord_file)[0] + '.json'
        
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                chord_data = json.load(f)
            
            song_name = chord_data.get("song", os.path.basename(os.path.splitext(chord_file)[0]))
            sections = chord_data.get("sections", {})
            full_progression = chord_data.get("full_progression", "").split()
            
            # Detect key based on full progression
            key, is_major = detect_key(full_progression)
            
            # Convert to Roman numerals
            roman_numerals = convert_to_roman_numerals(full_progression, key, is_major)
            
            # Identify common progressions
            common_progs = identify_common_progressions(roman_numerals)
            
            # Analyze each section
            section_analysis = {}
            for section_name, section_chords in sections.items():
                section_chords_list = section_chords.split()
                section_key, section_is_major = detect_key(section_chords_list)
                section_numerals = convert_to_roman_numerals(section_chords_list, section_key or key, section_is_major if section_key else is_major)
                section_common_progs = identify_common_progressions(section_numerals)
                
                section_analysis[section_name] = {
                    "chords": section_chords_list,
                    "key": section_key or key,
                    "is_major": section_is_major if section_key else is_major,
                    "roman_numerals": section_numerals,
                    "common_progressions": section_common_progs
                }
            
            analysis = {
                "song": song_name,
                "key": key,
                "is_major": is_major,
                "full_progression": full_progression,
                "roman_numerals": roman_numerals,
                "common_progressions": common_progs,
                "sections": section_analysis
            }
            
            return analysis
        
        # If no JSON file, analyze the text file directly
        with open(chord_file, 'r', encoding='utf-8') as f:
            chord_content = f.read()
        
        # Extract chords using regex
        chords = re.findall(r'\[([A-Ga-g][^\]]*?)\]', chord_content)
        
        # Remove duplicates while preserving order
        unique_chords = []
        for chord in chords:
            if chord not in unique_chords:
                unique_chords.append(chord)
        
        # Detect key
        key, is_major = detect_key(unique_chords)
        
        # Convert to Roman numerals
        roman_numerals = convert_to_roman_numerals(unique_chords, key, is_major)
        
        # Identify common progressions
        common_progs = identify_common_progressions(roman_numerals)
        
        # Extract sections if possible
        sections = {}
        section_matches = re.finditer(r'\[(Verse|Chorus|Bridge|Intro|Outro|Pre-Chorus|Solo|Instrumental|Interlude)[^\]]*?\](.*?)(?=\[(?:Verse|Chorus|Bridge|Intro|Outro|Pre-Chorus|Solo|Instrumental|Interlude)|$)', chord_content, re.DOTALL)
        
        for match in section_matches:
            section_name = match.group(1)
            section_content = match.group(2)
            section_chords = re.findall(r'\[([A-Ga-g][^\]]*?)\]', section_content)
            
            # Remove duplicates while preserving order
            section_unique_chords = []
            for chord in section_chords:
                if chord not in section_unique_chords:
                    section_unique_chords.append(chord)
            
            sections[section_name] = section_unique_chords
        
        # Analyze each section
        section_analysis = {}
        for section_name, section_chords in sections.items():
            section_key, section_is_major = detect_key(section_chords)
            section_numerals = convert_to_roman_numerals(section_chords, section_key or key, section_is_major if section_key else is_major)
            section_common_progs = identify_common_progressions(section_numerals)
            
            section_analysis[section_name] = {
                "chords": section_chords,
                "key": section_key or key,
                "is_major": section_is_major if section_key else is_major,
                "roman_numerals": section_numerals,
                "common_progressions": section_common_progs
            }
        
        song_name = os.path.basename(os.path.splitext(chord_file)[0])
        
        analysis = {
            "song": song_name,
            "key": key,
            "is_major": is_major,
            "full_progression": unique_chords,
            "roman_numerals": roman_numerals,
            "common_progressions": common_progs,
            "sections": section_analysis
        }
        
        return analysis
    
    except Exception as e:
        log_message(f"Error analyzing chord file {chord_file}: {e}")
        return None

def extract_lyrics_with_chords(chord_file):
    """
    Extract lyrics with chords from a chord file.
    Returns a dictionary of sections with their lyrics and chords.
    """
    try:
        with open(chord_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract sections with their content
        sections = {}
        current_section = "Intro"
        section_content = []
        
        for line in content.split('\n'):
            # Check if this is a section header
            section_match = re.match(r'\[(Verse|Chorus|Bridge|Intro|Outro|Pre-Chorus|Solo|Instrumental|Interlude)[^\]]*\]', line)
            if section_match:
                # Save the previous section
                if section_content:
                    sections[current_section] = '\n'.join(section_content)
                
                # Start a new section
                current_section = section_match.group(0).strip('[]')
                section_content = []
            else:
                section_content.append(line)
        
        # Save the last section
        if section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    except Exception as e:
        log_message(f"Error extracting lyrics with chords: {e}")
        return {}

def generate_analysis_report(analysis, output_file=None, chord_file=None):
    """
    Generate a human-readable analysis report from the analysis data.
    Includes lyrics with chords if chord_file is provided.
    """
    if not analysis:
        return "No analysis data available."
    
    report = f"# Chord Progression Analysis for '{analysis['song']}'\n\n"
    
    # Key information
    key_type = "Major" if analysis.get("is_major") else "Minor"
    report += f"## Key: {analysis.get('key', 'Unknown')} {key_type}\n\n"
    
    # Full progression
    report += "## Full Chord Progression\n\n"
    report += " - ".join(analysis.get("full_progression", [])) + "\n\n"
    
    # Roman numerals
    report += "## Roman Numeral Analysis\n\n"
    report += " - ".join(analysis.get("roman_numerals", [])) + "\n\n"
    
    # Common progressions
    report += "## Common Progressions Identified\n\n"
    common_progs = analysis.get("common_progressions", [])
    if common_progs:
        for prog, desc, key_type in common_progs:
            report += f"- {prog} ({key_type}): {desc}\n"
    else:
        report += "No common progressions identified.\n"
    
    report += "\n"
    
    # Extract lyrics with chords if chord_file is provided
    lyrics_with_chords = {}
    if chord_file and os.path.exists(chord_file):
        lyrics_with_chords = extract_lyrics_with_chords(chord_file)
    
    # Section analysis
    report += "## Section Analysis\n\n"
    sections = analysis.get("sections", {})
    if sections:
        for section_name, section_data in sections.items():
            section_key_type = "Major" if section_data.get("is_major") else "Minor"
            report += f"### {section_name}\n\n"
            report += f"Key: {section_data.get('key', 'Unknown')} {section_key_type}\n\n"
            report += "Chords: " + " - ".join(section_data.get("chords", [])) + "\n\n"
            report += "Roman Numerals: " + " - ".join(section_data.get("roman_numerals", [])) + "\n\n"
            
            section_common_progs = section_data.get("common_progressions", [])
            if section_common_progs:
                report += "Common Progressions:\n"
                for prog, desc, key_type in section_common_progs:
                    report += f"- {prog} ({key_type}): {desc}\n"
            else:
                report += "No common progressions identified in this section.\n"
            
            # Add lyrics with chords for this section if available
            section_key = section_name
            if section_key in lyrics_with_chords:
                report += "\n### Lyrics with Chords:\n\n"
                report += "```\n" + lyrics_with_chords[section_key] + "\n```\n"
            
            report += "\n"
    else:
        report += "No section analysis available.\n\n"
    
    # Add full lyrics with chords at the end
    if lyrics_with_chords:
        report += "## Full Song with Chords\n\n"
        report += "```\n"
        for section, content in lyrics_with_chords.items():
            report += f"[{section}]\n{content}\n\n"
        report += "```\n"
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        log_message(f"Analysis report saved to {output_file}")
    
    return report

def process_artist_chords(artist_name):
    """
    Process all chord files for an artist and generate analysis reports.
    Includes lyrics with chords in the analysis.
    """
    log_message(f"Analyzing chord files for artist: {artist_name}")
    
    # Get the artist's chord directory
    artist_dir = os.path.join(CLEAN_CHORDS_DIR, artist_name)
    
    if not os.path.exists(artist_dir):
        log_message(f"Error: Artist chord directory not found: {artist_dir}")
        return 0
    
    # Create the output directory
    output_dir = ensure_directory(os.path.join(ANALYSIS_DIR, artist_name))
    
    # Get all chord files
    chord_files = [os.path.join(artist_dir, f) for f in os.listdir(artist_dir) if f.endswith('.txt')]
    
    if not chord_files:
        log_message(f"No chord files found for {artist_name}")
        return 0
    
    # Process each chord file
    successful_count = 0
    
    for chord_file in chord_files:
        song_name = os.path.basename(os.path.splitext(chord_file)[0])
        log_message(f"Analyzing chord file: {song_name}")
        
        # Analyze the chord file
        analysis = analyze_chord_file(chord_file)
        
        if analysis:
            # Generate analysis report with lyrics and chords
            output_file = os.path.join(output_dir, f"{song_name}_analysis.md")
            generate_analysis_report(analysis, output_file, chord_file)
            
            # Save analysis data as JSON
            json_output_file = os.path.join(output_dir, f"{song_name}_analysis.json")
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2)
            
            log_message(f"Analysis completed for {song_name}")
            successful_count += 1
    
    log_message(f"Analyzed {len(chord_files)} chord files for {artist_name}, successfully analyzed {successful_count}")
    return successful_count

def main():
    parser = argparse.ArgumentParser(description="Analyze chord progressions in songs")
    parser.add_argument("artist", help="Artist name to analyze")
    parser.add_argument("--all", action="store_true", help="Process all artists in the clean_chords directory")
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directory(ANALYSIS_DIR)
    
    if args.all:
        # Process all artists
        artists = [d for d in os.listdir(CLEAN_CHORDS_DIR) if os.path.isdir(os.path.join(CLEAN_CHORDS_DIR, d))]
        
        for artist in artists:
            process_artist_chords(artist)
    else:
        # Process the specified artist
        process_artist_chords(args.artist)

if __name__ == "__main__":
    main()
