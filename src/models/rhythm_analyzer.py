"""
Rhythm analysis module for detecting tempo, time signatures, and note durations.
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple


class RhythmAnalyzer:
    """Analyze rhythm patterns, tempo, and timing from audio."""
    
    def __init__(self, config):
        """Initialize the rhythm analyzer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Rhythm analysis parameters
        self.sample_rate = config.get('audio.sample_rate', 44100)
        self.hop_length = config.get('audio.hop_length', 512)
        self.frame_duration = self.hop_length / self.sample_rate
        
        # Time signature possibilities
        self.common_time_signatures = [
            (4, 4), (3, 4), (2, 4), (6, 8), (9, 8), (12, 8),
            (7, 8), (5, 4), (7, 4), (2, 2)
        ]
        
    def analyze_rhythm(self, track_data):
        """
        Analyze rhythm patterns from audio track.
        
        Args:
            track_data (dict): Audio track data with 'audio'
            
        Returns:
            dict: Rhythm analysis results
        """
        self.logger.info("Analyzing rhythm patterns...")
        
        audio = track_data['audio']
        
        # Basic tempo and beat detection
        tempo, beats = self._detect_tempo_and_beats(audio)
        
        # Time signature detection
        time_signature = self._detect_time_signature(audio, beats)
        
        # Onset detection
        onsets = self._detect_onsets(audio)
        
        # Note duration analysis
        note_durations = self._analyze_note_durations(onsets, tempo)
        
        # Rhythmic pattern analysis
        rhythmic_patterns = self._analyze_rhythmic_patterns(onsets, beats, time_signature)
        
        # Swing detection
        swing_ratio = self._detect_swing(onsets, beats)
        
        rhythm_analysis = {
            'tempo': tempo,
            'beats': beats,
            'time_signature': time_signature,
            'onsets': onsets,
            'note_durations': note_durations,
            'rhythmic_patterns': rhythmic_patterns,
            'swing_ratio': swing_ratio,
            'beat_grid': self._create_beat_grid(tempo, time_signature, len(audio))
        }
        
        self.logger.info(f"Rhythm analysis completed - Tempo: {tempo:.1f} BPM, Time Signature: {time_signature}")
        
        return rhythm_analysis
    
    def _detect_tempo_and_beats(self, audio):
        """Detect tempo and beat positions."""
        # Primary tempo detection
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            start_bpm=60
        )
        # librosa >= 0.10 returns tempo as a 1-element array
        tempo = float(np.atleast_1d(tempo)[0])

        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=self.hop_length)
        
        # Tempo refinement using autocorrelation
        refined_tempo = self._refine_tempo_estimation(audio, tempo, beat_times)
        
        return refined_tempo, beat_times
    
    def _refine_tempo_estimation(self, audio, initial_tempo, beat_times):
        """Refine tempo estimation using multiple methods."""
        # Calculate tempo from beat intervals
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            median_interval = np.median(beat_intervals)
            interval_tempo = 60.0 / median_interval
            
            # Weighted average with initial tempo
            refined_tempo = 0.7 * initial_tempo + 0.3 * interval_tempo
        else:
            refined_tempo = initial_tempo
        
        return refined_tempo
    
    def _detect_time_signature(self, audio, beat_times):
        """Detect time signature from beat patterns."""
        if len(beat_times) < 8:  # Need enough beats for analysis
            return (4, 4)  # Default to 4/4
        
        # Analyze beat strength patterns
        beat_strengths = self._analyze_beat_strengths(audio, beat_times)
        
        # Test different time signatures
        best_signature = (4, 4)
        best_score = 0
        
        for num, denom in self.common_time_signatures:
            score = self._score_time_signature(beat_strengths, num, denom)
            if score > best_score:
                best_score = score
                best_signature = (num, denom)
        
        return best_signature
    
    def _analyze_beat_strengths(self, audio, beat_times):
        """Analyze the strength of each beat."""
        beat_strengths = []
        
        # Convert beat times to sample indices
        beat_samples = librosa.time_to_samples(beat_times, sr=self.sample_rate)
        
        # Calculate RMS energy around each beat
        window_size = int(0.1 * self.sample_rate)  # 100ms window
        
        for beat_sample in beat_samples:
            start = max(0, beat_sample - window_size // 2)
            end = min(len(audio), beat_sample + window_size // 2)
            
            if end > start:
                beat_audio = audio[start:end]
                rms_energy = np.sqrt(np.mean(beat_audio ** 2))
                beat_strengths.append(rms_energy)
            else:
                beat_strengths.append(0)
        
        return np.array(beat_strengths)
    
    def _score_time_signature(self, beat_strengths, num_beats, denom):
        """Score how well the beat strengths match a time signature."""
        if len(beat_strengths) < num_beats * 2:
            return 0
        
        # Reshape into measures
        num_measures = len(beat_strengths) // num_beats
        if num_measures == 0:
            return 0
        
        measures = beat_strengths[:num_measures * num_beats].reshape(num_measures, num_beats)
        
        # Calculate average strength for each beat position
        avg_beat_strengths = np.mean(measures, axis=0)
        
        # Score based on expected strong/weak beat patterns
        score = 0
        if num_beats == 4:  # 4/4 time
            # Strong on 1 and 3, weaker on 2 and 4
            score = avg_beat_strengths[0] + avg_beat_strengths[2] - avg_beat_strengths[1] - avg_beat_strengths[3]
        elif num_beats == 3:  # 3/4 time
            # Strong on 1, weaker on 2 and 3
            score = avg_beat_strengths[0] - avg_beat_strengths[1] - avg_beat_strengths[2]
        elif num_beats == 2:  # 2/4 time
            # Strong on 1, weaker on 2
            score = avg_beat_strengths[0] - avg_beat_strengths[1]
        
        return max(0, score)
    
    def _detect_onsets(self, audio):
        """Detect note onsets in the audio."""
        # Multi-method onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=True,
            units='frames'
        )
        
        # Convert to time
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=self.hop_length)
        
        # Additional onset detection using spectral flux
        stft = librosa.stft(audio, hop_length=self.hop_length)
        spectral_flux = np.sum(np.diff(np.abs(stft), axis=1) > 0, axis=0)
        
        # Find peaks in spectral flux
        from scipy.signal import find_peaks
        flux_peaks, _ = find_peaks(spectral_flux, height=np.mean(spectral_flux) + np.std(spectral_flux))
        flux_onset_times = librosa.frames_to_time(flux_peaks, sr=self.sample_rate, hop_length=self.hop_length)
        
        # Combine and remove duplicates
        all_onsets = np.concatenate([onset_times, flux_onset_times])
        all_onsets = np.unique(np.round(all_onsets, 3))  # Round to 1ms precision
        
        return all_onsets
    
    def _analyze_note_durations(self, onsets, tempo):
        """Analyze note durations from onset times."""
        if len(onsets) < 2:
            return []
        
        # Calculate intervals between onsets
        intervals = np.diff(onsets)
        
        # Convert to musical note values based on tempo
        beat_duration = 60.0 / tempo  # Duration of one beat in seconds
        
        note_durations = []
        for interval in intervals:
            # Classify into standard note durations
            ratio = interval / beat_duration
            
            if ratio >= 3.5:
                duration = 'whole'
            elif ratio >= 1.5:
                duration = 'half'
            elif ratio >= 0.75:
                duration = 'quarter'
            elif ratio >= 0.375:
                duration = 'eighth'
            elif ratio >= 0.1875:
                duration = 'sixteenth'
            else:
                duration = 'thirty-second'
            
            note_durations.append({
                'duration': duration,
                'beats': ratio,
                'seconds': interval
            })
        
        return note_durations
    
    def _analyze_rhythmic_patterns(self, onsets, beats, time_signature):
        """Analyze common rhythmic patterns."""
        if len(onsets) < 4 or len(beats) < 4:
            return []
        
        num_beats, denom = time_signature
        
        # Quantize onsets to beat grid
        quantized_onsets = self._quantize_to_beat_grid(onsets, beats, num_beats)
        
        # Find recurring patterns
        patterns = self._find_recurring_patterns(quantized_onsets, num_beats)
        
        return patterns
    
    def _quantize_to_beat_grid(self, onsets, beats, num_beats):
        """Quantize onsets to the nearest beat grid positions."""
        if len(beats) < 2:
            return onsets
        
        beat_interval = np.median(np.diff(beats))
        subdivision = beat_interval / 4  # Sixteenth note subdivisions
        
        quantized = []
        for onset in onsets:
            # Find closest grid position
            grid_position = round(onset / subdivision) * subdivision
            quantized.append(grid_position)
        
        return np.array(quantized)
    
    def _find_recurring_patterns(self, quantized_onsets, num_beats):
        """Find recurring rhythmic patterns."""
        # This is a simplified implementation
        # In a full system, this would use more sophisticated pattern matching
        
        patterns = []
        # For now, just identify basic patterns like straight vs. syncopated
        
        return patterns
    
    def _detect_swing(self, onsets, beats):
        """Detect swing rhythm ratio."""
        if len(onsets) < 8 or len(beats) < 4:
            return 1.0  # No swing
        
        # Analyze timing of eighth note pairs
        beat_interval = np.median(np.diff(beats))
        eighth_note_duration = beat_interval / 2
        
        # Find onsets that could be swing eighths
        swing_ratios = []
        
        for i in range(len(onsets) - 1):
            interval = onsets[i + 1] - onsets[i]
            
            # Check if this could be an eighth note pair
            if 0.4 * eighth_note_duration < interval < 1.6 * eighth_note_duration:
                # Calculate swing ratio (should be 2:1 for triplet swing)
                ratio = interval / eighth_note_duration
                swing_ratios.append(ratio)
        
        if swing_ratios:
            median_ratio = np.median(swing_ratios)
            # Detect common swing ratios
            if 0.9 < median_ratio < 1.1:
                return 1.0  # Straight
            elif 1.4 < median_ratio < 1.6:
                return 1.5  # Light swing
            elif 1.8 < median_ratio < 2.2:
                return 2.0  # Heavy swing (triplet)
        
        return 1.0  # Default to straight
    
    def _create_beat_grid(self, tempo, time_signature, audio_length):
        """Create a beat grid for the entire audio."""
        num_beats, denom = time_signature
        
        # Calculate beat duration
        beat_duration = 60.0 / tempo
        
        # Calculate total duration
        total_duration = audio_length / self.sample_rate
        
        # Create beat grid
        beats_per_measure = num_beats
        beat_times = []
        
        current_time = 0
        while current_time < total_duration:
            beat_times.append(current_time)
            current_time += beat_duration
        
        return {
            'beat_times': np.array(beat_times),
            'beat_duration': beat_duration,
            'measure_duration': beat_duration * beats_per_measure
        }
