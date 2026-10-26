"""
Pitch detection module using deep learning and signal processing.

Detects individual notes, their timing, and confidence levels.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import logging
from typing import List, Tuple, Dict


class PitchDetector:
    """Detect pitches and note timing from audio."""
    
    def __init__(self, config):
        """Initialize the pitch detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = config.get('pitch.model_path', 'models/pitch_model.pth')
        
        # Guitar-specific parameters
        self.fmin = config.get('pitch.fmin', 80)   # Low E string
        self.fmax = config.get('pitch.fmax', 1000) # High guitar frequencies
        self.n_bins = config.get('pitch.n_bins', 360)  # Semitone resolution
        
        # Load model
        self.model = self._load_model()
        
        # Guitar tuning (standard tuning in Hz)
        self.standard_tuning = {
            1: 329.63,  # E4 (high E)
            2: 246.94,  # B3
            3: 196.00,  # G3
            4: 146.83,  # D3
            5: 110.00,  # A2
            6: 82.41    # E2 (low E)
        }
    
    def _load_model(self):
        """Load the pitch detection model."""
        try:
            model = PitchCNN(
                input_size=self.config.get('audio.n_fft', 2048) // 2 + 1,
                n_bins=self.n_bins
            )
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Load trained weights if available
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded pitch model from {self.model_path}")
            except FileNotFoundError:
                self.logger.warning(f"No pitch model found at {self.model_path}")
                self.logger.info("Using randomly initialized model")
            
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading pitch model: {e}")
            raise
    
    def detect_pitches(self, track_data):
        """
        Detect pitches from audio track.
        
        Args:
            track_data (dict): Audio track data with 'audio' and 'stft'
            
        Returns:
            dict: Detected pitches with timing and confidence
        """
        self.logger.info("Detecting pitches...")
        
        audio = track_data['audio']
        sample_rate = self.config.get('audio.sample_rate', 44100)
        hop_length = self.config.get('audio.hop_length', 512)
        
        # Multi-F0 detection using both classical and ML approaches
        classical_pitches = self._classical_pitch_detection(audio, sample_rate)
        ml_pitches = self._ml_pitch_detection(track_data)
        
        # Combine and refine results
        refined_pitches = self._refine_pitch_detection(classical_pitches, ml_pitches)
        
        # Convert to guitar-specific format
        guitar_notes = self._convert_to_guitar_notes(refined_pitches, sample_rate, hop_length)
        
        return guitar_notes
    
    def _classical_pitch_detection(self, audio, sample_rate):
        """Classical pitch detection using multiple algorithms."""
        # YIN algorithm for fundamental frequency
        f0_yin = librosa.yin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sample_rate,
            hop_length=self.config.get('audio.hop_length', 512)
        )
        
        # Pitch tracking using harmonic analysis
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=sample_rate,
            hop_length=self.config.get('audio.hop_length', 512),
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Combine results
        classical_results = {
            'yin_f0': f0_yin,
            'piptrack_pitches': pitches,
            'piptrack_magnitudes': magnitudes
        }
        
        return classical_results
    
    def _ml_pitch_detection(self, track_data):
        """Machine learning-based pitch detection."""
        stft = track_data.get('stft', None)
        if stft is None:
            return None
        
        # Convert STFT to magnitude spectrogram
        magnitude = np.abs(stft)
        
        # Prepare input for ML model
        input_tensor = torch.FloatTensor(magnitude.T).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        # Run inference
        with torch.no_grad():
            pitch_probabilities = self.model(input_tensor)
        
        # Convert probabilities to pitch estimates
        pitch_estimates = self._probabilities_to_pitches(
            pitch_probabilities.cpu().numpy()
        )
        
        return pitch_estimates
    
    def _probabilities_to_pitches(self, probabilities):
        """Convert pitch probabilities to frequency estimates."""
        # Find peaks in probability distribution
        peaks = []
        threshold = self.config.get('pitch.confidence_threshold', 0.3)
        
        for frame_probs in probabilities[0]:
            frame_peaks = []
            for i, prob in enumerate(frame_probs):
                if prob > threshold:
                    # Convert bin index to frequency
                    freq = self._bin_to_frequency(i)
                    frame_peaks.append((freq, prob))
            peaks.append(frame_peaks)
        
        return peaks
    
    def _bin_to_frequency(self, bin_index):
        """Convert pitch bin index to frequency in Hz."""
        # Assuming logarithmic pitch scale
        cents_per_bin = 1200 / self.n_bins  # cents per bin
        cents_from_a4 = (bin_index - self.n_bins // 2) * cents_per_bin
        freq = 440 * (2 ** (cents_from_a4 / 1200))
        return freq
    
    def _refine_pitch_detection(self, classical_results, ml_results):
        """Combine and refine classical and ML pitch detection results."""
        if ml_results is None:
            return self._process_classical_only(classical_results)
        
        # Combine both approaches
        refined_pitches = []
        
        yin_f0 = classical_results['yin_f0']
        
        for frame_idx, ml_frame_pitches in enumerate(ml_results):
            frame_pitches = []
            
            # Add YIN fundamental frequency if confident
            if frame_idx < len(yin_f0) and yin_f0[frame_idx] > 0:
                frame_pitches.append({
                    'frequency': yin_f0[frame_idx],
                    'confidence': 0.7,  # Default confidence for YIN
                    'source': 'yin'
                })
            
            # Add ML-detected pitches
            for freq, confidence in ml_frame_pitches:
                frame_pitches.append({
                    'frequency': freq,
                    'confidence': confidence,
                    'source': 'ml'
                })
            
            # Remove duplicates and sort by confidence
            frame_pitches = self._remove_duplicate_pitches(frame_pitches)
            refined_pitches.append(frame_pitches)
        
        return refined_pitches
    
    def _process_classical_only(self, classical_results):
        """Process only classical pitch detection results."""
        yin_f0 = classical_results['yin_f0']
        pitches = classical_results['piptrack_pitches']
        magnitudes = classical_results['piptrack_magnitudes']
        
        refined_pitches = []
        
        for frame_idx in range(len(yin_f0)):
            frame_pitches = []
            
            # Add YIN result
            if yin_f0[frame_idx] > 0:
                frame_pitches.append({
                    'frequency': yin_f0[frame_idx],
                    'confidence': 0.7,
                    'source': 'yin'
                })
            
            # Add piptrack results
            if frame_idx < pitches.shape[1]:
                frame_pitch_bins = pitches[:, frame_idx]
                frame_magnitudes = magnitudes[:, frame_idx]
                
                for bin_idx, (pitch, mag) in enumerate(zip(frame_pitch_bins, frame_magnitudes)):
                    if pitch > 0 and mag > 0.1:  # Threshold for magnitude
                        frame_pitches.append({
                            'frequency': pitch,
                            'confidence': min(mag * 2, 1.0),  # Scale magnitude to confidence
                            'source': 'piptrack'
                        })
            
            frame_pitches = self._remove_duplicate_pitches(frame_pitches)
            refined_pitches.append(frame_pitches)
        
        return refined_pitches
    
    def _remove_duplicate_pitches(self, pitches):
        """Remove duplicate pitches within a frequency tolerance."""
        if not pitches:
            return pitches
        
        # Sort by frequency
        pitches.sort(key=lambda x: x['frequency'])
        
        # Remove duplicates within tolerance
        tolerance = 5  # Hz
        filtered_pitches = [pitches[0]]
        
        for pitch in pitches[1:]:
            if abs(pitch['frequency'] - filtered_pitches[-1]['frequency']) > tolerance:
                filtered_pitches.append(pitch)
            elif pitch['confidence'] > filtered_pitches[-1]['confidence']:
                # Replace with higher confidence detection
                filtered_pitches[-1] = pitch
        
        # Sort by confidence (highest first)
        filtered_pitches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return filtered_pitches
    
    def _convert_to_guitar_notes(self, refined_pitches, sample_rate, hop_length):
        """Convert pitch detections to guitar-specific note format."""
        guitar_notes = {
            'notes': [],
            'timing': [],
            'confidence': [],
            'string_assignments': []
        }
        
        frame_duration = hop_length / sample_rate
        
        for frame_idx, frame_pitches in enumerate(refined_pitches):
            timestamp = frame_idx * frame_duration
            
            for pitch_info in frame_pitches:
                frequency = pitch_info['frequency']
                confidence = pitch_info['confidence']
                
                # Convert frequency to MIDI note
                midi_note = librosa.hz_to_midi(frequency)
                
                # Assign to guitar string and fret
                string_assignment = self._assign_to_guitar_string(frequency)
                
                guitar_notes['notes'].append(midi_note)
                guitar_notes['timing'].append(timestamp)
                guitar_notes['confidence'].append(confidence)
                guitar_notes['string_assignments'].append(string_assignment)
        
        return guitar_notes
    
    def _assign_to_guitar_string(self, frequency):
        """Assign a frequency to the most appropriate guitar string and fret."""
        best_string = None
        best_fret = None
        min_error = float('inf')
        
        for string_num, open_freq in self.standard_tuning.items():
            # Calculate fret number
            fret = round(12 * np.log2(frequency / open_freq))
            
            # Check if fret is playable (0-24 typically)
            if 0 <= fret <= 24:
                # Calculate the actual frequency for this string/fret combination
                actual_freq = open_freq * (2 ** (fret / 12))
                error = abs(frequency - actual_freq)
                
                if error < min_error:
                    min_error = error
                    best_string = string_num
                    best_fret = fret
        
        return {
            'string': best_string,
            'fret': best_fret,
            'frequency_error': min_error
        }


class PitchCNN(nn.Module):
    """Convolutional Neural Network for pitch detection."""
    
    def __init__(self, input_size, n_bins, chunk_size=256):
        super(PitchCNN, self).__init__()

        self.input_size = input_size
        self.n_bins = n_bins
        # Frames per chunk; bounds conv activation memory (128 x input_size per frame)
        self.chunk_size = chunk_size
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=25, padding=12)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, padding=12)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=25, padding=12)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_bins)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # Input shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, input_size = x.shape

        # Treat every frame as its own 1-channel signal so the convolutions
        # slide along the frequency axis: (batch_size * sequence_length, 1, input_size)
        frames = x.reshape(batch_size * seq_len, 1, input_size)

        outputs = []
        for chunk in torch.split(frames, self.chunk_size):
            out = self.relu(self.conv1(chunk))
            out = self.relu(self.conv2(out))
            out = self.relu(self.conv3(out))  # (chunk, 128, input_size)

            out = out.flatten(1)  # (chunk, 128 * input_size)
            out = self.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.relu(self.fc2(out))
            out = self.dropout(out)
            outputs.append(self.sigmoid(self.fc3(out)))

        # (batch_size, sequence_length, n_bins)
        return torch.cat(outputs).reshape(batch_size, seq_len, self.n_bins)
