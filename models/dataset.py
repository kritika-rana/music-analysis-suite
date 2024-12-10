import torch
from torch.utils.data import Dataset
import pretty_midi
import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings
from typing import Tuple, Optional
import logging
from pathlib import Path

# Set up logging
log_file = "midi_validation.log"
logging.basicConfig(
    filename=log_file,
    filemode='a',  # Append to the log file
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class MIDIDataset(Dataset):
    def __init__(self, data_dir: str, subset: str = 'train', min_samples: int = 100):
        self.data_dir = Path(data_dir)
        
        # Load matched files
        self.data_df = pd.read_csv(self.data_dir / 'matched_files.csv')
        
        # Load genre mapping
        with open(self.data_dir / 'genre_mapping.json', 'r') as f:
            self.genre_mapping = json.load(f)

        # Filter out genres with insufficient samples
        genre_counts = self.data_df['Genre'].value_counts()
        valid_genres = genre_counts[genre_counts >= min_samples].index
        self.data_df = self.data_df[self.data_df['Genre'].isin(valid_genres)]
        
        # Update genre mapping to only include remaining genres
        original_mapping = self.genre_mapping.copy()
        self.genre_mapping = {}
        for genre in valid_genres:
            if genre in original_mapping:
                self.genre_mapping[genre] = len(self.genre_mapping)
                
        # Save updated genre mapping
        with open(self.data_dir / 'genre_mapping_filtered.json', 'w') as f:
            json.dump(self.genre_mapping, f, indent=2)

        # Split train/val/test (80/10/10)
        np.random.seed(42)
        indices = np.random.permutation(len(self.data_df))
        if subset == 'train':
            idx = indices[:int(0.8 * len(indices))]
        elif subset == 'val':
            idx = indices[int(0.8 * len(indices)):int(0.9 * len(indices))]
        else:  # test
            idx = indices[int(0.9 * len(indices)):]
        
        self.data_df = self.data_df.iloc[idx].reset_index(drop=True)

        # Calculate sample weights for training
        if subset == 'train':
            self.sample_weights = self._compute_sample_weights()
        else:
            self.sample_weights = None
            
        self._print_stats()

    def _compute_sample_weights(self) -> torch.Tensor:
        """
        Compute weights for each sample to balance classes
        Returns weights tensor for WeightedRandomSampler
        """
        genre_counts = self.data_df['Genre'].value_counts()
        total_samples = len(self.data_df)
        
        # Weight for each genre is inversely proportional to its frequency
        weights = {genre: total_samples / (len(self.genre_mapping) * count) 
                  for genre, count in genre_counts.items()}
        
        # Assign weights to each sample
        sample_weights = torch.tensor([
            weights[genre] for genre in self.data_df['Genre']
        ])
        
        return sample_weights
    
    def get_sample_weights(self) -> Optional[torch.Tensor]:
        """
        Get sample weights for WeightedRandomSampler.
        Returns None if not in training mode.
        """
        return self.sample_weights

    def _print_stats(self):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        genre_counts = self.data_df['Genre'].value_counts()
        print("\nSamples per genre:")
        for genre, count in genre_counts.items():
            print(f"{genre}: {count}")
        print(f"\nTotal samples: {len(self.data_df)}")
        print(f"Number of genres: {len(self.genre_mapping)}")
        
        if self.sample_weights is not None:
            print("\nEffective samples per genre after weighting:")
            genre_weights = {}
            for genre, count in genre_counts.items():
                mask = self.data_df['Genre'] == genre
                avg_weight = self.sample_weights[mask].mean().item()
                effective_samples = count * avg_weight
                genre_weights[genre] = effective_samples
                print(f"{genre}: {effective_samples:.1f}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data_df.iloc[idx]
        
        # Extract features
        features = self.extract_features(row.Path)
        if features is None:
            features = np.zeros(29, dtype=np.float32)
        
        # Get genre label
        genre_idx = self.genre_mapping[row.Genre]
        
        return torch.FloatTensor(features), torch.tensor(genre_idx)
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def extract_features(self, midi_path: str) -> Optional[np.ndarray]:
        """Extract comprehensive musical features from MIDI file"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    midi_data = pretty_midi.PrettyMIDI(midi_path)
                    logging.info(f"Valid MIDI file: {midi_path}")
                except Exception as e:
                    logging.error(f"Invalid MIDI file: {midi_path} - {e}")
                    return None
                
                features = []
                
                # Basic Tempo Features
                tempo = midi_data.estimate_tempo()
                features.append((tempo - 150) / 300)  # Normalized tempo
                
                # Time Signature
                if len(midi_data.time_signature_changes) > 0:
                    ts = midi_data.time_signature_changes[0]
                    numerator = (ts.numerator - 4) / 4
                    denominator = (ts.denominator - 4) / 4
                else:
                    numerator = 0
                    denominator = 0
                features.extend([numerator, denominator])
                
                # Note Density and Duration Features
                total_notes = 0
                note_durations = []
                velocity_values = []
                
                for instrument in midi_data.instruments:
                    total_notes += len(instrument.notes)
                    note_durations.extend([note.end - note.start for note in instrument.notes])
                    velocity_values.extend([note.velocity for note in instrument.notes])
                
                duration = midi_data.get_end_time()
                features.append((total_notes / duration - 50) / 100)  # Density
                
                if note_durations:
                    features.extend([
                        np.mean(note_durations),  # Average note duration
                        np.std(note_durations),   # Variation in note duration
                        np.mean(velocity_values) / 127,  # Average velocity
                        np.std(velocity_values) / 127    # Velocity dynamics
                    ])
                else:
                    features.extend([0, 0, 0, 0])
                
                # Rhythm Features
                beat_times = midi_data.get_beats()
                if len(beat_times) > 1:
                    beat_intervals = np.diff(beat_times)
                    features.extend([
                        np.mean(beat_intervals),  # Average beat interval
                        np.std(beat_intervals),   # Rhythm regularity
                    ])
                else:
                    features.extend([0, 0])
                
                # Pitch Features
                all_notes = [note.pitch for instrument in midi_data.instruments 
                            for note in instrument.notes]
                if all_notes:
                    features.extend([
                        (max(all_notes) - min(all_notes)) / 127,  # Pitch range
                        np.mean(all_notes) / 127,  # Average pitch
                        np.std(all_notes) / 127    # Pitch variation
                    ])
                else:
                    features.extend([0, 0, 0])
                
                # Harmonic Features
                if len(midi_data.instruments) > 0:
                    # Get piano roll with higher resolution
                    piano_roll = midi_data.get_piano_roll(fs=50)
                    # Compute average harmony density
                    harmony_density = np.mean(np.count_nonzero(piano_roll, axis=0)) / piano_roll.shape[0]
                    features.append(harmony_density)
                    
                    # Compute chroma features - fixed reshaping logic
                    try:
                        # Fold piano roll into 12 pitch classes
                        chroma = np.zeros((12, piano_roll.shape[1]))
                        for pitch in range(piano_roll.shape[0]):
                            pitch_class = pitch % 12
                            chroma[pitch_class] += piano_roll[pitch]
                            
                        # Normalize
                        chroma_sum = np.sum(chroma, axis=0) + 1e-8
                        chroma = chroma / chroma_sum[np.newaxis, :]
                        
                        # Calculate mean for each pitch class
                        chroma_means = np.mean(chroma, axis=1)
                        features.extend(chroma_means)
                    except Exception as e:
                        # If chroma computation fails, add zeros
                        features.extend([0] * 12)
                else:
                    features.extend([0] * 13)  # Harmony density + 12 chroma features
                
                # Instrument Diversity Features
                num_instruments = len(midi_data.instruments)
                features.append(num_instruments / 10)  # Normalize assuming max 10 instruments
                
                # Program (instrument) variety
                programs = set(inst.program for inst in midi_data.instruments)
                features.append(len(programs) / 128)  # Normalize by max MIDI programs
                
                # Drum presence
                has_drums = any(inst.is_drum for inst in midi_data.instruments)
                features.append(float(has_drums))
                
                return np.array(features, dtype=np.float32)
                    
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None