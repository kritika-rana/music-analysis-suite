import numpy as np
from typing import Dict
from models.inference import MusicAnalyzer
import pretty_midi
from midi2audio import FluidSynth

GENERATION_PATH = "data/outputs/"
SOUNDFONT_PATH = "data/soundfonts/FluidR3_GM/FluidR3_GM.sf2"
# Define available instruments for each style
STYLE_INSTRUMENTS = {
    'classical': {
        'melody': {'name': 'violin', 'program': 40},
        'harmony': {'name': 'string_ensemble', 'program': 48},
        'bass': {'name': 'cello', 'program': 42}
    },
    'jazz': {
        'melody': {'name': 'trumpet', 'program': 56},
        'harmony': {'name': 'piano', 'program': 0},
        'bass': {'name': 'acoustic_bass', 'program': 32}
    },
    'electronic': {
        'melody': {'name': 'synth_lead', 'program': 80},
        'harmony': {'name': 'synth_pad', 'program': 89},
        'bass': {'name': 'synth_bass', 'program': 38}
    },
    'rock': {
        'melody': {'name': 'electric_guitar', 'program': 27},
        'harmony': {'name': 'electric_piano', 'program': 4},
        'bass': {'name': 'electric_bass', 'program': 33}
    },
    'acoustic': {
        'melody': {'name': 'acoustic_guitar', 'program': 25},
        'harmony': {'name': 'piano', 'program': 0},
        'bass': {'name': 'acoustic_bass', 'program': 32}
    }
}

class GenerationAgent:
    def __init__(self, checkpoint_path: str, data_dir: str = 'data/midi'):
        self.analyzer = MusicAnalyzer(checkpoint_path, data_dir)

    def convert_midi_to_wav(self, midi_path, wav_path):
        fs = FluidSynth(sound_font=SOUNDFONT_PATH)
        fs.midi_to_audio(midi_path, wav_path)
    
    def _create_chord_progression(self, root_note: int, scale_type: str = 'major') -> list:
        """Create a basic chord progression based on a root note."""
        if scale_type == 'major':
            # I-IV-V-I progression
            intervals = [0, 5, 7, 0]  # Basic chord progression
            chord_intervals = [(0, 4, 7),  # Major triad
                             (5, 9, 12),   # Fourth chord
                             (7, 11, 14),  # Fifth chord
                             (0, 4, 7)]    # Back to root
        else:
            # i-iv-v-i progression for minor
            intervals = [0, 5, 7, 0]
            chord_intervals = [(0, 3, 7),  # Minor triad
                             (5, 8, 12),   # Minor fourth
                             (7, 10, 14),  # Minor fifth
                             (0, 3, 7)]    # Back to root

        return [(root_note + interval, chord_ints) 
                for interval, chord_ints in zip(intervals, chord_intervals)]

    def _add_melody(self, pm: pretty_midi.PrettyMIDI, features: np.ndarray, 
                   style: str, duration: float = 16.0) -> pretty_midi.PrettyMIDI:
        """Add a melody line based on the features."""
        instrument_config = STYLE_INSTRUMENTS[style]['melody']
        melody = pretty_midi.Instrument(program=instrument_config['program'])
        
        # Extract musical parameters from features
        base_pitch = int(features[4] * 24 + 60)  # Center around middle C
        note_density = max(4, int(features[3] * 16))  # More reasonable note density
        rhythm_variety = features[2]  # Use for varying note lengths
        
        # Create a basic scale (major or minor based on features)
        scale_type = 'major' if features[1] > 0.5 else 'minor'
        if scale_type == 'major':
            scale_intervals = [0, 2, 4, 5, 7, 9, 11, 12]
        else:
            scale_intervals = [0, 2, 3, 5, 7, 8, 10, 12]
        
        # Generate melody
        time = 0
        while time < duration:
            # Vary note length based on rhythm_variety
            note_length = max(0.25, min(1.0, 0.5 + rhythm_variety * np.random.uniform(-0.3, 0.3)))
            
            # Choose a pitch from the scale
            scale_step = np.random.choice(len(scale_intervals))
            pitch = base_pitch + scale_intervals[scale_step]
            
            # Add some melodic motion
            if np.random.random() < 0.2:  # 20% chance of larger interval
                pitch += np.random.choice([-12, -7, -5, 5, 7, 12])
            
            # Keep pitch in reasonable range
            pitch = max(48, min(84, pitch))
            
            # Add the note
            velocity = int(np.random.uniform(80, 100))
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=time,
                end=time + note_length * 0.95  # Slight separation between notes
            )
            melody.notes.append(note)
            
            time += note_length
        
        pm.instruments.append(melody)
        return pm

    def _add_accompaniment(self, pm: pretty_midi.PrettyMIDI, features: np.ndarray,
                          style: str, duration: float = 16.0) -> pretty_midi.PrettyMIDI:
        """Add accompaniment chords based on the features."""
        instrument_config = STYLE_INSTRUMENTS[style]['harmony']
        accompaniment = pretty_midi.Instrument(program=instrument_config['program'])
        
        # Determine root note and scale type from features
        root_note = int(features[4] * 12 + 48)  # Base octave for accompaniment
        scale_type = 'major' if features[1] > 0.5 else 'minor'
        
        # Get chord progression
        chord_progression = self._create_chord_progression(root_note, scale_type)
        
        # Add chords
        time = 0
        chord_duration = duration / len(chord_progression)
        for root, intervals in chord_progression:
            # Create chord
            for interval in intervals:
                note = pretty_midi.Note(
                    velocity=70,  # Softer velocity for accompaniment
                    pitch=root + interval,
                    start=time,
                    end=time + chord_duration * 0.9  # Slight separation between chords
                )
                accompaniment.notes.append(note)
            time += chord_duration
        
        pm.instruments.append(accompaniment)
        return pm
    
    def _add_bass(self, pm: pretty_midi.PrettyMIDI, features: np.ndarray,
                 style: str, duration: float = 16.0) -> pretty_midi.PrettyMIDI:
        """Add bass line based on the features and style."""
        instrument_config = STYLE_INSTRUMENTS[style]['bass']
        bass = pretty_midi.Instrument(program=instrument_config['program'])
        
        # Determine root note and scale type from features
        root_note = int(features[4] * 12 + 36)  # Lower octave for bass
        scale_type = 'major' if features[1] > 0.5 else 'minor'
        chord_progression = self._create_chord_progression(root_note, scale_type)
        
        # Add bass notes
        time = 0
        chord_duration = duration / len(chord_progression)
        for root, _ in chord_progression:
            # Add bass note
            note = pretty_midi.Note(
                velocity=85,
                pitch=root,
                start=time,
                end=time + chord_duration * 0.9
            )
            bass.notes.append(note)
            time += chord_duration
        
        pm.instruments.append(bass)
        return pm

    def convert_features_to_midi(self, features: np.ndarray, style: str) -> pretty_midi.PrettyMIDI:
        """Convert generated features to MIDI with style-specific instrumentation."""
        if style not in STYLE_INSTRUMENTS:
            raise ValueError(f"Unknown style: {style}. Available styles: {list(STYLE_INSTRUMENTS.keys())}")
            
        pm = pretty_midi.PrettyMIDI()
        duration = 16.0
        
        # Set tempo and time signature
        tempo = features[0] * 60 + 90
        pm.tempo_changes = [(0, tempo)]
        pm.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0)]
        
        # Add layers with style-specific instruments
        pm = self._add_bass(pm, features, style, duration)
        pm = self._add_accompaniment(pm, features, style, duration)
        pm = self._add_melody(pm, features, style, duration)
        
        return pm
    
    def process(self, state: Dict) -> Dict:
        """
        Process the generation request and add results to the state.
        
        Args:
            state (Dict): The state containing the input parameters.
        
        Returns:
            Dict: Updated state with the generation results.
        """
        generation_params = state.get("generation_parameters", {})
        style = generation_params.get("music_style", "classical")

        try:
            # Generate music
            generated_features = self.analyzer.generate_random_music()[0]

            # Convert generated features to MIDI format
            generated_midi_object = self.convert_features_to_midi(generated_features, style)

            # Save the MIDI object to a file
            midi_file_path = GENERATION_PATH + "generated_midi_output.mid"
            generated_midi_object.write(midi_file_path)  # Save PrettyMIDI object as a .mid file

            # save wav object to file
            wav_file_path = GENERATION_PATH + "generated_wav_output.wav"
            self.convert_midi_to_wav(midi_file_path, wav_file_path)
            
             # Update state with the generated music path
            state["generation_results"] = {
                "generated_wav_path": wav_file_path,
                "generated_midi_path": midi_file_path,
            }
            state["generation_complete"] = True
        
        except Exception as e:
            state["generation_error"] = str(e)
        
        return state
