import requests
import os
import tarfile
import zipfile
from tqdm import tqdm
from pathlib import Path
import shutil
import json
import pandas as pd
import io

class MSDMIDIDownloader:
    def __init__(self):
        self.base_dir = Path('data')
        self.midi_dir = self.base_dir / 'midi'
        self.temp_dir = self.base_dir / 'temp'
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.midi_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs
        self.urls = {
            'genre_labels': 'https://www.tagtraum.com/genres/msd_tagtraum_cd1.cls.zip',
            'midi_matched': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz'
        }

    def download_and_extract_zip(self, url: str, target_dir: Path) -> Path:
        """Download and extract zip file, returning path to .cls file"""
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        zip_content = io.BytesIO()
        with tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = zip_content.write(data)
                pbar.update(size)
        
        # Extract the zip file
        print("Extracting zip file...")
        zip_content.seek(0)
        with zipfile.ZipFile(zip_content) as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Find the .cls file
        cls_files = list(target_dir.glob('*.cls'))
        if not cls_files:
            raise FileNotFoundError("No .cls file found in zip archive")
        
        return cls_files[0]

    def download_file(self, url: str, filename: str) -> Path:
        """Download file with progress bar"""
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        filepath = self.temp_dir / filename
        with open(filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        return filepath

    def extract_file(self, filepath: Path, extract_dir: Path):
        """Extract compressed files"""
        print(f"Extracting {filepath}...")
        filepath_str = str(filepath)
        if filepath_str.endswith('.tar.gz'):
            with tarfile.open(filepath_str) as tar:
                tar.extractall(path=str(extract_dir))

    def get_genres(self, path: str) -> pd.DataFrame:
        """Read genre labels into DataFrame"""
        print("Processing genre labels...")
        ids = []
        genres = []
        with open(path) as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        ids.append(parts[0])
                        genres.append(parts[1])
        
        df = pd.DataFrame(data={"Genre": genres, "TrackID": ids})
        print(f"Found {len(df)} genre labels")
        return df

    def get_matched_midi(self, midi_folder: str, genre_df: pd.DataFrame) -> pd.DataFrame:
        """Match MIDI files with genre labels with improved matching logic"""
        print("Matching MIDI files with genres...")
        
        # Get All Midi Files
        track_ids, file_paths = [], []
        
        # Track directory structure for debugging
        dir_structure = {}
        
        for dir_path, _, files in os.walk(midi_folder):
            midi_files = [f for f in files if f.endswith('.mid')]
            if midi_files:
                # Get all parent directory names
                path_parts = Path(dir_path).parts
                current_path = ''
                for part in path_parts:
                    current_path = os.path.join(current_path, part)
                    if part not in dir_structure:
                        dir_structure[part] = {
                            'count': 1,
                            'sample_path': current_path
                        }
                    else:
                        dir_structure[part]['count'] += 1
                
                # Try different parts of the path to find the track ID
                potential_id = None
                for part in reversed(path_parts):  # Start from deepest directory
                    if part in genre_df['TrackID'].values:
                        potential_id = part
                        break
                
                if potential_id:
                    for midi_file in midi_files:
                        midi_path = os.path.join(dir_path, midi_file)
                        track_ids.append(potential_id)
                        file_paths.append(midi_path)
            
        all_midi_df = pd.DataFrame({"TrackID": track_ids, "Path": file_paths})
        print(f"Found {len(all_midi_df)} MIDI files")
        print(f"Unique track IDs in MIDI files: {len(all_midi_df['TrackID'].unique())}")
        print(f"Unique track IDs in genre labels: {len(genre_df['TrackID'].unique())}")
        
        # Save directory structure analysis
        analysis = {
            'directory_structure': dir_structure,
            'midi_stats': {
                'total_files': len(all_midi_df),
                'unique_ids': len(all_midi_df['TrackID'].unique()),
                'sample_tracks': all_midi_df['TrackID'].head(10).tolist(),
                'sample_paths': all_midi_df['Path'].head(10).tolist()
            },
            'genre_stats': {
                'total_labels': len(genre_df),
                'unique_ids': len(genre_df['TrackID'].unique()),
                'sample_ids': genre_df['TrackID'].head(10).tolist()
            }
        }
        
        with open(self.midi_dir / 'matching_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Inner Join with Genre Dataframe
        df = pd.merge(all_midi_df, genre_df, on='TrackID', how='inner')
        print(f"Successfully matched {len(df)} files with genres")
        
        return df.drop(["TrackID"], axis=1)

    def download_and_organize(self):
        """Main function to download and organize all data"""
        try:
            # Download and extract genre labels
            print("Downloading and extracting genre labels...")
            cls_file = self.download_and_extract_zip(
                self.urls['genre_labels'],
                self.temp_dir
            )
            
            # Get genre DataFrame
            genre_df = self.get_genres(str(cls_file))
            
            # Create genre mapping
            label_list = sorted(list(set(genre_df.Genre)))
            label_dict = {lbl: idx for idx, lbl in enumerate(label_list)}
            
            # Save genre mapping
            with open(self.midi_dir / 'genre_mapping.json', 'w') as f:
                json.dump(label_dict, f, indent=2)
            print("Saved genre mapping")
            
            # Download and extract MIDI files
            print("Downloading MIDI files...")
            midi_file = self.download_file(
                self.urls['midi_matched'],
                'lmd_matched.tar.gz'
            )
            print("Extracting MIDI files (this may take a while)...")
            self.extract_file(midi_file, self.midi_dir)
            
            # Match MIDI files with genres
            matched_df = self.get_matched_midi(
                str(self.midi_dir / 'lmd_matched'),
                genre_df
            )
            
            # Save matched DataFrame
            matched_df.to_csv(self.midi_dir / 'matched_files.csv', index=False)
            print("Saved matched files CSV")
            
            # Print statistics
            self.print_statistics(matched_df)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise
        finally:
            # Cleanup
            self.cleanup()

    def print_statistics(self, matched_df: pd.DataFrame):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        genre_counts = matched_df['Genre'].value_counts()
        print(genre_counts)
        
        stats = {
            'total_files': len(matched_df),
            'genres': genre_counts.to_dict()
        }
        
        with open(self.midi_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("Saved dataset statistics")

    def cleanup(self):
        """Remove temporary files"""
        print("Cleaning up temporary files...")
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

def main():
    downloader = MSDMIDIDownloader()
    downloader.download_and_organize()

if __name__ == "__main__":
    main()