"""
AutoTablature - AI-Powered Guitar Tablature Generation

Main entry point for the AutoTablature system.
"""

from src.audio.processor import AudioProcessor
from src.separation.separator import InstrumentSeparator
from src.models.pitch_detector import PitchDetector
from src.models.rhythm_analyzer import RhythmAnalyzer
from src.tablature.generator import TablatureGenerator
from src.utils.config import Config
from src.utils.logger import setup_logger
import logging
import os


class AutoTablature:
    """
    Main class for the AutoTablature system.
    
    This class orchestrates the entire pipeline from audio input
    to Guitar Pro tablature output.
    """
    
    def __init__(self, config_path=None):
        """Initialize the AutoTablature system."""
        self.logger = setup_logger(__name__)
        self.config = Config(config_path)
        
        # Initialize components
        self.audio_processor = AudioProcessor(self.config)
        self.separator = InstrumentSeparator(self.config)
        self.pitch_detector = PitchDetector(self.config)
        self.rhythm_analyzer = RhythmAnalyzer(self.config)
        self.tablature_generator = TablatureGenerator(self.config)
        
        self.logger.info("AutoTablature system initialized")
    
    def process_audio(self, audio_path, output_path=None):
        """
        Process an audio file and generate tablature.
        
        Args:
            audio_path (str): Path to input audio file
            output_path (str, optional): Path for output file
            
        Returns:
            TablatureDocument: Generated tablature
        """
        self.logger.info(f"Processing audio file: {audio_path}")
        
        # Step 1: Load and preprocess audio
        audio_data, _ = self.audio_processor.load_audio(audio_path)
        processed_audio = self.audio_processor.preprocess(audio_data)
        
        # Step 2: Separate instruments
        separated_tracks = self.separator.separate_instruments(processed_audio)
        
        # Step 3: Analyze each track
        tablature_tracks = []
        for track_name, track_audio in separated_tracks.items():
            self.logger.info(f"Analyzing track: {track_name}")
            
            # Detect pitches and timing
            pitches = self.pitch_detector.detect_pitches(track_audio)
            rhythm = self.rhythm_analyzer.analyze_rhythm(track_audio)
            
            # Generate tablature for this track
            track_tablature = self.tablature_generator.generate_track(
                pitches, rhythm, instrument=track_name
            )
            tablature_tracks.append(track_tablature)
        
        # Step 4: Combine tracks into final tablature
        final_tablature = self.tablature_generator.combine_tracks(tablature_tracks)
        
        # Step 5: Save output if path provided
        if output_path:
            final_tablature.save(output_path)
            self.logger.info(f"Tablature saved to: {output_path}")
        
        return final_tablature
    
    def train_custom_model(self, data_path, model_type="all"):
        """
        Train custom models on your dataset.
        
        Args:
            data_path (str): Path to training data directory
            model_type (str): Type of model to train ("pitch", "rhythm", "separation", "all")
        """
        from src.models.trainer import ModelTrainer
        
        trainer = ModelTrainer(self.config)
        trainer.train(data_path, model_type)
    
    def evaluate_model(self, test_data_path):
        """
        Evaluate model performance on test data.
        
        Args:
            test_data_path (str): Path to test data directory
            
        Returns:
            dict: Evaluation metrics
        """
        from src.models.evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator(self.config)
        return evaluator.evaluate(test_data_path)


def main():
    """Example usage of AutoTablature."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoTablature - AI Guitar Tablature Generation")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("-o", "--output", help="Path to output tablature file")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Initialize system
    at = AutoTablature(args.config)
    
    # Process audio
    tablature = at.process_audio(args.input, args.output)
    
    print(f"Successfully generated tablature with {len(tablature.tracks)} tracks")


if __name__ == "__main__":
    main()
