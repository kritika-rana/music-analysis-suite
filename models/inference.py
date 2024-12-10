import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from .vae import MusicVAE
from .dataset import MIDIDataset

class MusicAnalyzer:
    def __init__(self, checkpoint_path: str, data_dir: str = 'data/midi'):
        """
        Initialize the music analyzer with a trained model
        Args:
            checkpoint_path: Path to trained model checkpoint
            data_dir: Directory containing genre mapping and data
        """
        # Load genre mapping
        with open(Path(data_dir) / 'genre_mapping_filtered.json', 'r') as f:
            self.genre_mapping = json.load(f)
        
        # Create reverse mapping
        self.idx_to_genre = {v: k for k, v in self.genre_mapping.items()}
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MusicVAE(num_genres=len(self.genre_mapping))
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize dataset for feature extraction
        self.dataset = MIDIDataset(data_dir)

    def get_latent_representation(self, midi_path: str) -> Optional[np.ndarray]:
        """
        Get the latent space representation of a MIDI file
        Args:
            midi_path: Path to MIDI file
        Returns:
            Latent vector representation
        """
        features = self.dataset.extract_features(midi_path)
        if features is None:
            return None
            
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu, _ = self.model.encode(features_tensor)
            return mu.cpu().numpy()[0]

    def classify_midi(self, midi_path: str) -> Dict[str, float]:
        """
        Classify a MIDI file and return genre probabilities
        Args:
            midi_path: Path to MIDI file
        Returns:
            Dictionary of genre probabilities
        """
        features = self.dataset.extract_features(midi_path)
        if features is None:
            raise ValueError("Could not extract features from MIDI file")
            
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, mu, _, genre_pred = self.model(features_tensor)
            probs = torch.softmax(genre_pred, dim=1)[0]
            
            return {
                self.idx_to_genre[i]: prob.item()
                for i, prob in enumerate(probs)
            }
        
    def analyze_style(self, midi_path: str, genre_probs: Dict[str, float]) -> Dict[str, str]:
        """
        Analyze musical style characteristics
        """
        features = self.dataset.extract_features(midi_path)
        if features is None:
            return {"error": "Could not analyze MIDI file"}
        
        # Get top 3 genres
        top_genres = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        analysis = {
            "primary_genre": f"{top_genres[0][0]} ({top_genres[0][1]:.1%})",
            "genre_influences": [f"{genre} ({prob:.1%})" for genre, prob in top_genres[1:]],
            "complexity": "high" if features[3] > 0.7 else "medium" if features[3] > 0.3 else "low",
            "instrumentation": "full" if features[26] * 10 > 5 else "medium" if features[26] * 10 > 2 else "minimal",
            "rhythm_character": "regular" if features[9] < 0.1 else "varied",
            "primary_tonality": self._get_primary_tonality(features[14:26])
        }
        
        return analysis

    def _get_primary_tonality(self, chroma_features: np.ndarray) -> str:
        """
        Determine primary tonality from chroma features
        """
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        max_idx = np.argmax(chroma_features)
        return notes[max_idx]

    def get_feature_description(self, midi_path: str) -> Dict[str, str]:
        """
        Get human-readable description of musical features
        Args:
            midi_path: Path to MIDI file
        Returns:
            Dictionary of feature descriptions
        """
        features = self.dataset.extract_features(midi_path)
        if features is None:
            return {"error": "Could not extract features from MIDI file"}

        # Feature indices based on our enhanced feature extraction
        descriptions = {
            # Basic Features
            'tempo': f"{features[0] * 300 + 150:.1f} BPM",
            'time_signature': f"{int(features[1] * 4 + 4)}/{int(features[2] * 4 + 4)}",
            'note_density': f"{features[3] * 100 + 50:.1f} notes per second",
            
            # Note Statistics
            'avg_note_duration': f"{features[4]:.2f} seconds",
            'duration_variety': f"{features[5]:.2f}",
            'avg_velocity': f"{features[6] * 127:.1f}",
            'dynamics_range': f"{features[7] * 127:.1f}",
            
            # Rhythm Features
            'avg_beat_interval': f"{features[8]:.2f} seconds",
            'rhythm_regularity': 'regular' if features[9] < 0.1 else 'variable',
            
            # Pitch Features
            'pitch_range': f"{features[10] * 127:.0f} semitones",
            'avg_pitch': f"{features[11] * 127:.0f}",
            'pitch_variety': f"{features[12] * 127:.0f}",
            
            # Harmonic Features
            'harmony_density': f"{features[13]:.2f}",
            'chroma_profile': {
                'C': f"{features[14]:.2f}",
                'C#': f"{features[15]:.2f}",
                'D': f"{features[16]:.2f}",
                'D#': f"{features[17]:.2f}",
                'E': f"{features[18]:.2f}",
                'F': f"{features[19]:.2f}",
                'F#': f"{features[20]:.2f}",
                'G': f"{features[21]:.2f}",
                'G#': f"{features[22]:.2f}",
                'A': f"{features[23]:.2f}",
                'A#': f"{features[24]:.2f}",
                'B': f"{features[25]:.2f}"
            },
            
            # Instrument Features
            'instrument_count': f"{int(features[26] * 10)} instruments",
            'instrument_variety': f"{int(features[27] * 128)} unique types",
            'has_drums': 'Yes' if features[28] > 0.5 else 'No'
        }

        # Add musical interpretation
        musical_character = []
        if features[3] > 0.7:  # High note density
            musical_character.append("dense and complex")
        elif features[3] < 0.3:
            musical_character.append("sparse and minimal")
            
        if features[7] > 0.5:  # High dynamics range
            musical_character.append("dynamically expressive")
        
        if features[9] < 0.1:  # Regular rhythm
            musical_character.append("rhythmically steady")
        else:
            musical_character.append("rhythmically varied")
            
        if features[26] * 10 > 5:  # Many instruments
            musical_character.append("full ensemble")
        elif features[26] * 10 <= 2:
            musical_character.append("minimal instrumentation")

        descriptions['musical_character'] = ", ".join(musical_character)

        return descriptions
    

    def generate_random_music(self, num_samples: int = 1) -> np.ndarray:
        """Generate multiple random music pieces from latent space sampling"""
        # Sample from standard normal distribution
        z_sample = torch.randn(num_samples, self.model.latent_dim).to(self.device)
        
        with torch.no_grad():
            # Decode the samples
            generated_features = self.model.decoder(z_sample)
            
        return generated_features.cpu().numpy()