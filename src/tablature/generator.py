"""
Tablature generation module for creating Guitar Pro format files.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
import json
import guitarpro


class TablatureGenerator:
    """Generate guitar tablature in Guitar Pro format."""
    
    def __init__(self, config):
        """Initialize the tablature generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Guitar configuration
        self.num_strings = config.get('guitar.num_strings', 6)
        self.num_frets = config.get('guitar.num_frets', 24)
        
        # Standard tuning (MIDI note numbers)
        self.standard_tuning = [64, 59, 55, 50, 45, 40]  # E, B, G, D, A, E
        
        # Note duration mappings
        self.duration_map = {
            'whole': 1.0,
            'half': 0.5,
            'quarter': 0.25,
            'eighth': 0.125,
            'sixteenth': 0.0625,
            'thirty-second': 0.03125
        }
    
    def generate_track(self, pitches, rhythm, instrument="guitar"):
        """
        Generate tablature for a single track.
        
        Args:
            pitches (dict): Detected pitches and timing
            rhythm (dict): Rhythm analysis results
            instrument (str): Instrument type
            
        Returns:
            TablatureTrack: Generated tablature track
        """
        self.logger.info(f"Generating tablature for {instrument} track...")
        
        # Combine pitch and rhythm information
        notes = self._combine_pitch_and_rhythm(pitches, rhythm)
        
        # Optimize fingering positions
        optimized_notes = self._optimize_fingering(notes)
        
        # Create tablature measures
        measures = self._create_measures(optimized_notes, rhythm)
        
        # Create track object
        track = TablatureTrack(
            instrument=instrument,
            tuning=self.standard_tuning,
            measures=measures,
            tempo=rhythm['tempo'],
            time_signature=rhythm['time_signature']
        )
        
        self.logger.info(f"Generated {len(measures)} measures for {instrument}")
        
        return track
    
    def _combine_pitch_and_rhythm(self, pitches, rhythm):
        """Combine pitch detection with rhythm analysis."""
        notes = []
        
        pitch_notes = pitches['notes']
        pitch_times = pitches['timing']
        pitch_confidences = pitches['confidence']
        string_assignments = pitches['string_assignments']
        
        onsets = rhythm['onsets']
        note_durations = rhythm['note_durations']
        
        # Match pitches to rhythm onsets
        for i, onset_time in enumerate(onsets):
            # Find closest pitch detection
            closest_pitch_idx = self._find_closest_pitch(onset_time, pitch_times)
            
            if closest_pitch_idx is not None:
                # Get duration
                duration = 'quarter'  # Default duration
                if i < len(note_durations):
                    duration = note_durations[i]['duration']
                
                note = {
                    'time': onset_time,
                    'midi_note': pitch_notes[closest_pitch_idx],
                    'confidence': pitch_confidences[closest_pitch_idx],
                    'string_assignment': string_assignments[closest_pitch_idx],
                    'duration': duration
                }
                notes.append(note)
        
        return notes
    
    def _find_closest_pitch(self, target_time, pitch_times):
        """Find the closest pitch detection to a target time."""
        if not pitch_times:
            return None
        
        time_diffs = np.abs(np.array(pitch_times) - target_time)
        closest_idx = np.argmin(time_diffs)
        
        # Only accept if within reasonable time window (100ms)
        if time_diffs[closest_idx] < 0.1:
            return closest_idx
        
        return None
    
    def _optimize_fingering(self, notes):
        """Optimize fingering positions for playability."""
        if not notes:
            return notes
        
        optimized_notes = []
        previous_fret_positions = [0] * self.num_strings  # Track fret positions
        
        for note in notes:
            string_assignment = note['string_assignment']
            
            if string_assignment and string_assignment['string'] is not None:
                # Use the suggested string assignment
                string_num = string_assignment['string']
                fret_num = string_assignment['fret']
                
                # Validate and potentially adjust for better fingering
                adjusted_assignment = self._adjust_for_fingering(
                    note['midi_note'], 
                    string_num, 
                    fret_num, 
                    previous_fret_positions
                )
                
                optimized_note = note.copy()
                optimized_note['string'] = adjusted_assignment['string']
                optimized_note['fret'] = adjusted_assignment['fret']
                
                # Update position tracking
                if adjusted_assignment['string'] <= len(previous_fret_positions):
                    previous_fret_positions[adjusted_assignment['string'] - 1] = adjusted_assignment['fret']
                
                optimized_notes.append(optimized_note)
            else:
                # Find best fingering from scratch
                best_assignment = self._find_best_fingering(
                    note['midi_note'], 
                    previous_fret_positions
                )
                
                if best_assignment:
                    optimized_note = note.copy()
                    optimized_note['string'] = best_assignment['string']
                    optimized_note['fret'] = best_assignment['fret']
                    
                    # Update position tracking
                    previous_fret_positions[best_assignment['string'] - 1] = best_assignment['fret']
                    
                    optimized_notes.append(optimized_note)
        
        return optimized_notes
    
    def _adjust_for_fingering(self, midi_note, suggested_string, suggested_fret, previous_positions):
        """Adjust string/fret assignment for better fingering."""
        # Check if suggested position is playable given previous positions
        playability_score = self._calculate_playability_score(
            suggested_string, suggested_fret, previous_positions
        )
        
        # If playability is good enough, use suggestion
        if playability_score > 0.7:
            return {'string': suggested_string, 'fret': suggested_fret}
        
        # Otherwise, find alternative positions
        alternatives = self._find_alternative_positions(midi_note)
        
        best_alternative = None
        best_score = 0
        
        for alt_string, alt_fret in alternatives:
            score = self._calculate_playability_score(alt_string, alt_fret, previous_positions)
            if score > best_score:
                best_score = score
                best_alternative = {'string': alt_string, 'fret': alt_fret}
        
        return best_alternative or {'string': suggested_string, 'fret': suggested_fret}
    
    def _find_alternative_positions(self, midi_note):
        """Find alternative string/fret positions for a MIDI note."""
        alternatives = []
        
        for string_idx, open_note in enumerate(self.standard_tuning):
            string_num = string_idx + 1
            fret = midi_note - open_note
            
            if 0 <= fret <= self.num_frets:
                alternatives.append((string_num, fret))
        
        return alternatives
    
    def _calculate_playability_score(self, string, fret, previous_positions):
        """Calculate how playable a position is given previous positions."""
        if string <= 0 or string > len(previous_positions):
            return 0
        
        # Factors affecting playability:
        # 1. Distance from previous fret positions
        # 2. Fret range (closer to nut is generally easier)
        # 3. String stretching required
        
        score = 1.0
        
        # Distance penalty
        avg_previous_fret = np.mean([pos for pos in previous_positions if pos > 0])
        if avg_previous_fret > 0:
            distance = abs(fret - avg_previous_fret)
            if distance > 4:  # More than 4 frets apart
                score *= 0.5
            elif distance > 2:
                score *= 0.8
        
        # High fret penalty
        if fret > 12:
            score *= 0.7
        elif fret > 18:
            score *= 0.4
        
        # Open string bonus
        if fret == 0:
            score *= 1.2
        
        return score
    
    def _find_best_fingering(self, midi_note, previous_positions):
        """Find the best fingering for a note."""
        alternatives = self._find_alternative_positions(midi_note)
        
        if not alternatives:
            return None
        
        best_assignment = None
        best_score = 0
        
        for string, fret in alternatives:
            score = self._calculate_playability_score(string, fret, previous_positions)
            if score > best_score:
                best_score = score
                best_assignment = {'string': string, 'fret': fret}
        
        return best_assignment
    
    def _create_measures(self, notes, rhythm):
        """Create measures from notes and rhythm information."""
        if not notes:
            return []
        
        measures = []
        
        # Get measure duration
        time_signature = rhythm['time_signature']
        tempo = rhythm['tempo']
        
        beats_per_measure = time_signature[0]
        beat_duration = 60.0 / tempo  # seconds per beat
        measure_duration = beats_per_measure * beat_duration
        
        # Group notes into measures
        current_measure_start = 0
        current_measure_notes = []
        
        for note in notes:
            note_time = note['time']
            
            # Check if note belongs to current measure
            if note_time >= current_measure_start + measure_duration:
                # Start new measure
                if current_measure_notes:
                    measure = self._create_measure(
                        current_measure_notes, 
                        current_measure_start,
                        measure_duration,
                        time_signature
                    )
                    measures.append(measure)
                
                # Update for new measure
                current_measure_start += measure_duration
                current_measure_notes = []
            
            current_measure_notes.append(note)
        
        # Add final measure if it has notes
        if current_measure_notes:
            measure = self._create_measure(
                current_measure_notes,
                current_measure_start,
                measure_duration,
                time_signature
            )
            measures.append(measure)
        
        return measures
    
    def _create_measure(self, notes, start_time, duration, time_signature):
        """Create a single measure from notes."""
        measure = TablatureMeasure(
            time_signature=time_signature,
            start_time=start_time,
            duration=duration
        )
        
        for note in notes:
            # Calculate position within measure
            position_in_measure = note['time'] - start_time
            beat_position = position_in_measure / (duration / time_signature[0])
            
            tab_note = TablatureNote(
                string=note.get('string', 1),
                fret=note.get('fret', 0),
                duration=note['duration'],
                beat_position=beat_position,
                confidence=note.get('confidence', 1.0)
            )
            
            measure.add_note(tab_note)
        
        return measure
    
    def combine_tracks(self, tracks):
        """
        Combine multiple tracks into a single tablature document.
        
        Args:
            tracks (list): List of TablatureTrack objects
            
        Returns:
            TablatureDocument: Combined tablature document
        """
        self.logger.info(f"Combining {len(tracks)} tracks...")
        
        # Create document
        document = TablatureDocument()
        
        # Add each track
        for track in tracks:
            document.add_track(track)
        
        # Synchronize timing across tracks
        document.synchronize_tracks()
        
        self.logger.info("Track combination completed")
        
        return document


class TablatureNote:
    """Represents a single note in tablature."""
    
    def __init__(self, string, fret, duration, beat_position, confidence=1.0):
        self.string = string
        self.fret = fret
        self.duration = duration
        self.beat_position = beat_position
        self.confidence = confidence
    
    def to_dict(self):
        return {
            'string': self.string,
            'fret': self.fret,
            'duration': self.duration,
            'beat_position': self.beat_position,
            'confidence': self.confidence
        }


class TablatureMeasure:
    """Represents a measure in tablature."""
    
    def __init__(self, time_signature, start_time, duration):
        self.time_signature = time_signature
        self.start_time = start_time
        self.duration = duration
        self.notes = []
    
    def add_note(self, note):
        self.notes.append(note)
    
    def to_dict(self):
        return {
            'time_signature': self.time_signature,
            'start_time': self.start_time,
            'duration': self.duration,
            'notes': [note.to_dict() for note in self.notes]
        }


class TablatureTrack:
    """Represents a single instrument track in tablature."""
    
    def __init__(self, instrument, tuning, measures, tempo, time_signature):
        self.instrument = instrument
        self.tuning = tuning
        self.measures = measures
        self.tempo = tempo
        self.time_signature = time_signature
    
    def to_dict(self):
        return {
            'instrument': self.instrument,
            'tuning': self.tuning,
            'tempo': self.tempo,
            'time_signature': self.time_signature,
            'measures': [measure.to_dict() for measure in self.measures]
        }


class TablatureDocument:
    """Represents a complete tablature document."""

    # Export grid resolution: note onsets are quantized to sixteenth notes
    SLOTS_PER_WHOLE = 16
    # (channel, effect channel) MIDI pairs per track; channel 9 is reserved for percussion
    GP_CHANNEL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (10, 11), (12, 13), (14, 15)]
    # ASCII tab layout: characters per grid slot, and measures per line of tab
    ASCII_SLOT_WIDTH = 3
    ASCII_MEASURES_PER_LINE = 2
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self):
        self.tracks = []
        self.title = ""
        self.artist = ""
        self.album = ""
        self.copyright = ""
        self.logger = logging.getLogger(__name__)

    def add_track(self, track):
        self.tracks.append(track)
    
    def synchronize_tracks(self):
        """Synchronize timing across all tracks."""
        # This would implement timing synchronization logic
        # For now, tracks are kept as separate entities
        pass
    
    def save(self, filename):
        """Save tablature to file."""
        if filename.endswith('.json'):
            self._save_json(filename)
        elif filename.endswith('.gp5'):
            self._save_guitar_pro(filename)
        elif filename.endswith('.txt'):
            self._save_ascii(filename)
        elif filename.endswith('.gpx'):
            raise ValueError("PyGuitarPro cannot write Guitar Pro 6 (.gpx) files; use .gp5")
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    def _save_json(self, filename):
        """Save as JSON format for debugging/intermediate use."""
        data = {
            'title': self.title,
            'artist': self.artist,
            'album': self.album,
            'copyright': self.copyright,
            'tracks': [track.to_dict() for track in self.tracks]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_ascii(self, filename):
        """Save as plain-text ASCII tab."""
        with open(filename, 'w') as f:
            f.write(self.to_ascii())

    def to_ascii(self):
        """
        Render the document as ASCII tab on the sixteenth-note export grid.

        Each grid slot shows the fret of a note starting there, or dashes.
        """
        lines = [f"{self.title or 'AutoTablature Generated'} - {self.artist or 'AutoTablature'}"]

        for track_number, track in enumerate(self.tracks, start=1):
            # Tuning is listed from string 1 (highest) down
            labels = [self.NOTE_NAMES[midi_note % 12] for midi_note in track.tuning]
            labels[0] = labels[0].lower()  # conventional lowercase high e
            lines += [
                "",
                f"Track {track_number}: {track.instrument} ({track.tempo:.0f} BPM, "
                f"tuning {' '.join(reversed(labels))}, low to high)",
                "",
            ]

            measures = [
                self._ascii_measure(measure, len(track.tuning)) for measure in track.measures
            ]
            for start in range(0, len(measures), self.ASCII_MEASURES_PER_LINE):
                group = measures[start:start + self.ASCII_MEASURES_PER_LINE]
                header = "   "
                for offset, (time_signature, rows) in enumerate(group):
                    title = f"m{start + offset + 1} ({time_signature[0]}/{time_signature[1]})"
                    header += title.ljust(len(rows[0]) + 1)
                lines.append(header.rstrip())
                for string_index, label in enumerate(labels):
                    lines.append(f"{label:<2}|" + "".join(rows[string_index] + "|" for _, rows in group))
                lines.append("")

        return "\n".join(lines) + "\n"

    def _ascii_measure(self, measure, num_strings):
        """Render one measure as a row of text per string."""
        slots_per_measure, notes_by_slot = self._quantize_notes(measure.notes, measure.time_signature)
        empty = "-" * self.ASCII_SLOT_WIDTH
        rows = [[empty] * slots_per_measure for _ in range(num_strings)]

        for slot, by_string in notes_by_slot.items():
            for string, note in by_string.items():
                if 1 <= string <= num_strings:
                    rows[string - 1][slot] = str(self._fret_number(note)).ljust(self.ASCII_SLOT_WIDTH, "-")

        return measure.time_signature, ["".join(row) for row in rows]

    def _quantize_notes(self, notes, time_signature):
        """
        Quantize note onsets in a measure to the export grid.

        Returns:
            tuple: (slots_per_measure, {slot: {string: note}}), keeping the most
            confident note when several land on the same string and slot
        """
        numerator, denominator = time_signature
        slots_per_beat = self.SLOTS_PER_WHOLE // denominator
        slots_per_measure = numerator * slots_per_beat

        notes_by_slot = {}
        for note in notes:
            slot = int(round(note.beat_position * slots_per_beat))
            slot = min(max(slot, 0), slots_per_measure - 1)
            by_string = notes_by_slot.setdefault(slot, {})
            current = by_string.get(note.string)
            if current is None or note.confidence > current.confidence:
                by_string[note.string] = note

        return slots_per_measure, notes_by_slot

    def _fret_number(self, tab_note):
        """Fret as a non-negative integer (the generator can produce fractional frets)."""
        return max(0, int(round(tab_note.fret)))

    def _save_guitar_pro(self, filename):
        """Save in Guitar Pro 5 format."""
        try:
            self.logger.info(f"Creating Guitar Pro file: {filename}")

            if not self.tracks:
                raise ValueError("No tracks to save")

            gp_song = guitarpro.models.Song(
                title=self.title or "AutoTablature Generated",
                artist=self.artist or "AutoTablature",
                album=self.album or "",
                copyright=self.copyright or "",
                tempo=int(round(self.tracks[0].tempo)),
            )
            # Song() starts with one default track and measure; build our own
            gp_song.tracks = []
            gp_song.measureHeaders = []

            # Measure headers are shared by every track, so size them to the longest track
            longest = max(self.tracks, key=lambda track: len(track.measures))
            start = guitarpro.models.Duration.quarterTime
            for index in range(max(len(longest.measures), 1)):
                numerator, denominator = (
                    longest.measures[index].time_signature if longest.measures else (4, 4)
                )
                header = guitarpro.models.MeasureHeader(number=index + 1, start=start)
                header.timeSignature.numerator = numerator
                header.timeSignature.denominator.value = denominator
                gp_song.addMeasureHeader(header)
                start += header.length

            for track_index, track in enumerate(self.tracks):
                channel, effect_channel = self.GP_CHANNEL_PAIRS[track_index % len(self.GP_CHANNEL_PAIRS)]
                gp_track = guitarpro.models.Track(
                    gp_song,
                    number=track_index + 1,
                    name=track.instrument,
                    strings=self._create_gp_strings(track.tuning),
                    channel=guitarpro.models.MidiChannel(
                        channel=channel,
                        effectChannel=effect_channel,
                        instrument=self._get_gp_instrument_id(track.instrument),
                        volume=100,
                    ),
                )

                # Track() creates one empty measure per song header; fill those in.
                # Tracks shorter than the longest one are padded with rest measures
                for index, gp_measure in enumerate(gp_track.measures):
                    notes = track.measures[index].notes if index < len(track.measures) else []
                    self._fill_gp_voice(gp_measure.voices[0], notes, gp_measure.header)

                gp_song.tracks.append(gp_track)

            guitarpro.write(gp_song, filename)
            self.logger.info(f"Successfully saved Guitar Pro file to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving Guitar Pro file: {e}")
            # Fallback to JSON
            json_filename = filename.replace('.gp5', '.json')
            self.logger.warning(f"Falling back to JSON format: {json_filename}")
            self._save_json(json_filename)
    
    def _get_gp_instrument_id(self, instrument_name):
        """Map instrument name to Guitar Pro instrument ID."""
        instrument_map = {
            "guitar_lead": 25,     # Distortion Guitar
            "guitar_rhythm": 24,   # Overdriven Guitar
            "guitar_clean": 27,    # Clean Guitar
            "guitar_acoustic": 0,  # Acoustic Guitar (Steel)
            "bass": 33,            # Electric Bass (Finger)
            "piano": 0,            # Grand Piano
            "drums": 0,            # Standard Drum Kit
            "vocals": 53           # Voice Oohs
        }
        
        # Default to electric guitar if not found
        return instrument_map.get(instrument_name.lower(), 25)
    
    def _create_gp_strings(self, tuning):
        """Create Guitar Pro string tuning."""
        gp_strings = []
        
        # Convert MIDI note numbers to Guitar Pro string objects
        for midi_note in tuning:
            gp_string = guitarpro.models.GuitarString(number=len(gp_strings) + 1, value=midi_note)
            gp_strings.append(gp_string)
            
        return gp_strings
    
    def _fill_gp_voice(self, gp_voice, notes, header):
        """Fill a Guitar Pro voice with beats that cover the whole measure."""
        time_signature = (header.timeSignature.numerator, header.timeSignature.denominator.value)
        slots_per_measure, notes_by_slot = self._quantize_notes(notes, time_signature)
        onsets = sorted(notes_by_slot)

        # Leading rest before the first note (or a whole-measure rest if empty)
        first_onset = onsets[0] if onsets else slots_per_measure
        self._append_gp_beats(gp_voice, first_onset, [])

        # Each chord lasts until the next onset or the end of the measure
        for onset, next_onset in zip(onsets, onsets[1:] + [slots_per_measure]):
            self._append_gp_beats(gp_voice, next_onset - onset, list(notes_by_slot[onset].values()))

    def _append_gp_beats(self, gp_voice, length, tab_notes):
        """
        Append beats spanning `length` grid slots to a voice.

        Lengths that are not a single note value are split into several beats;
        the notes are tied across them, and empty spans become rests.
        """
        tie = False
        while length > 0:
            piece = next(size for size in (16, 8, 4, 2, 1) if size <= length)
            length -= piece

            gp_beat = guitarpro.models.Beat(
                gp_voice,
                duration=guitarpro.models.Duration(value=self.SLOTS_PER_WHOLE // piece),
            )
            if tab_notes:
                gp_beat.status = guitarpro.models.BeatStatus.normal
                for tab_note in tab_notes:
                    gp_beat.notes.append(self._create_gp_note(gp_beat, tab_note, tie))
            else:
                gp_beat.status = guitarpro.models.BeatStatus.rest

            gp_voice.beats.append(gp_beat)
            tie = True

    def _create_gp_note(self, gp_beat, tab_note, tie=False):
        """Create a Guitar Pro note from a tablature note."""
        return guitarpro.models.Note(
            gp_beat,
            value=self._fret_number(tab_note),
            string=tab_note.string,
            type=guitarpro.models.NoteType.tie if tie else guitarpro.models.NoteType.normal,
        )
