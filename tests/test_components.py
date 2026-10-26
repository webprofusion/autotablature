"""
Component tests for AutoTablature.

This script tests individual components of the AutoTablature system.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import librosa

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import Config
from src.audio.processor import AudioProcessor
from src.separation.separator import InstrumentSeparator
from src.models.pitch_detector import PitchDetector
from src.models.rhythm_analyzer import RhythmAnalyzer
from src.tablature.generator import TablatureGenerator


def test_audio_processor(config, input_file):
    """Test the audio processor component."""
    print("\n--- Testing Audio Processor ---")
    processor = AudioProcessor(config)
    
    # Load audio
    try:
        audio, sr = processor.load_audio(input_file)
        print(f"Loaded audio: {input_file}")
        print(f"Duration: {len(audio) / sr:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        
        # Preprocess
        features = processor.preprocess(audio)
        print(f"Extracted features:")
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        # Guitar-specific features
        guitar_features = processor.extract_guitar_features(audio)
        print(f"Extracted guitar-specific features:")
        for key, value in guitar_features.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}")
        
        return features
        
    except Exception as e:
        print(f"Error in audio processor: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_separator(config, features):
    """Test the instrument separator component."""
    print("\n--- Testing Instrument Separator ---")
    
    if features is None:
        print("Cannot test separator: No audio features available")
        return None
    
    separator = InstrumentSeparator(config)
    
    try:
        separated_tracks = separator.separate_instruments(features)
        print(f"Separated {len(separated_tracks)} tracks:")
        
        for instrument, track_data in separated_tracks.items():
            audio = track_data['audio']
            print(f"  {instrument}: duration={len(audio) / config.get('audio.sample_rate'):.2f}s")
        
        return separated_tracks
        
    except Exception as e:
        print(f"Error in instrument separator: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pitch_detector(config, track_data):
    """Test the pitch detector component."""
    print("\n--- Testing Pitch Detector ---")
    
    if track_data is None:
        print("Cannot test pitch detector: No track data available")
        return None
    
    pitch_detector = PitchDetector(config)
    
    try:
        # Use first track for testing
        instrument, data = next(iter(track_data.items()))
        print(f"Detecting pitches for track: {instrument}")
        
        pitches = pitch_detector.detect_pitches(data)
        
        print(f"Detected {len(pitches['notes'])} notes")
        print(f"First 5 notes:")
        
        for i in range(min(5, len(pitches['notes']))):
            note = pitches['notes'][i]
            time = pitches['timing'][i]
            confidence = pitches['confidence'][i]
            string_assignment = pitches['string_assignments'][i]
            
            print(f"  Note {i+1}: MIDI={note:.1f}, Time={time:.2f}s, "
                  f"Confidence={confidence:.2f}, "
                  f"String={string_assignment['string']}, Fret={string_assignment['fret']}")
        
        return pitches
    
    except Exception as e:
        print(f"Error in pitch detector: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_rhythm_analyzer(config, track_data):
    """Test the rhythm analyzer component."""
    print("\n--- Testing Rhythm Analyzer ---")
    
    if track_data is None:
        print("Cannot test rhythm analyzer: No track data available")
        return None
    
    rhythm_analyzer = RhythmAnalyzer(config)
    
    try:
        # Use first track for testing
        instrument, data = next(iter(track_data.items()))
        print(f"Analyzing rhythm for track: {instrument}")
        
        rhythm = rhythm_analyzer.analyze_rhythm(data)
        
        print(f"Detected tempo: {rhythm['tempo']:.1f} BPM")
        print(f"Time signature: {rhythm['time_signature'][0]}/{rhythm['time_signature'][1]}")
        print(f"Detected {len(rhythm['beats'])} beats")
        print(f"Detected {len(rhythm['onsets'])} onsets")
        
        if rhythm['note_durations']:
            print(f"Note durations (first 5):")
            for i, duration in enumerate(rhythm['note_durations'][:5]):
                print(f"  Note {i+1}: {duration['duration']} ({duration['beats']:.2f} beats)")
        
        return rhythm
    
    except Exception as e:
        print(f"Error in rhythm analyzer: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tablature_generator(config, pitches, rhythm):
    """Test the tablature generator component."""
    print("\n--- Testing Tablature Generator ---")
    
    if pitches is None or rhythm is None:
        print("Cannot test tablature generator: Missing pitch or rhythm data")
        return None
    
    tablature_generator = TablatureGenerator(config)
    
    try:
        # Generate tablature
        print("Generating tablature")
        track = tablature_generator.generate_track(pitches, rhythm, instrument="guitar_test")
        
        print(f"Generated {len(track.measures)} measures")
        
        # Check first few measures
        print(f"First 3 measures:")
        for i, measure in enumerate(track.measures[:3]):
            print(f"  Measure {i+1}: {len(measure.notes)} notes, "
                  f"Time signature: {measure.time_signature[0]}/{measure.time_signature[1]}")
        
        # Combine into document
        document = tablature_generator.combine_tracks([track])
        print(f"Created tablature document with {len(document.tracks)} tracks")
        
        return document
    
    except Exception as e:
        print(f"Error in tablature generator: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run the component tests."""
    parser = argparse.ArgumentParser(description='AutoTablature Component Tests')
    parser.add_argument('-i', '--input', required=True, help='Path to input audio file')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    parser.add_argument('-o', '--output', help='Path to output test tablature file')
    parser.add_argument('-t', '--test', choices=['all', 'audio', 'separation', 'pitch', 'rhythm', 'tablature'],
                        default='all', help='Test specific component')
    
    args = parser.parse_args()
    
    print("AutoTablature Component Tests")
    print("----------------------------")
    print(f"Input file: {args.input}")
    print(f"Config file: {args.config or 'Using default'}")
    print(f"Test: {args.test}")
    print("")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return
    
    # Initialize config
    config = Config(args.config)
    
    # Run tests
    features = None
    separated_tracks = None
    pitches = None
    rhythm = None
    
    if args.test in ['all', 'audio']:
        features = test_audio_processor(config, args.input)
    
    if args.test in ['all', 'separation'] and features is not None:
        separated_tracks = test_separator(config, features)
    
    if args.test in ['all', 'pitch'] and separated_tracks is not None:
        pitches = test_pitch_detector(config, separated_tracks)
    
    if args.test in ['all', 'rhythm'] and separated_tracks is not None:
        rhythm = test_rhythm_analyzer(config, separated_tracks)
    
    if args.test in ['all', 'tablature'] and pitches is not None and rhythm is not None:
        document = test_tablature_generator(config, pitches, rhythm)
        
        # Save output if requested
        if args.output and document is not None:
            document.save(args.output)
            print(f"\nSaved test tablature to {args.output}")


if __name__ == "__main__":
    main()
