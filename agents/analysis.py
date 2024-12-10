import torch
from typing import Dict
from models.inference import MusicAnalyzer

class AnalysisAgent:
    def __init__(self, checkpoint_path: str, data_dir: str = 'data/midi'):
        self.analyzer = MusicAnalyzer(checkpoint_path, data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process(self, state: Dict) -> Dict:
        """Analyze MIDI features using VAE"""
        midi_path = state.get("midi_path")
        if not midi_path:
            raise ValueError("No MIDI file path provided in state")

        try:
            # Get genre predictions
            genre_probs = self.analyzer.classify_midi(midi_path)
            
            # Get style analysis
            style_analysis = self.analyzer.analyze_style(midi_path, genre_probs)
            
            # Update state
            state.update({
                "genre_prediction": genre_probs,
                "style_analysis": style_analysis,
                "analysis_complete": True
            })
            
            return state
            
        except Exception as e:
            state["analysis_error"] = f"Analysis failed: {str(e)}"
            return state