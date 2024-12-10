# Music Analysis and Generation

The Music Analysis Suite is an AI-powered application designed to analyze MIDI music files. It combines machine learning (Variational Autoencoder - VAE), retrieval-augmented generation (RAG), and multi-agent coordination.

---

## Setup and Installation

### Clone the Repository
```bash
git clone https://github.com/kritika-rana/music-analysis-suite.git
cd music_analysis
```

### Training the model
Note: Model training is NOT REQUIRED as the model is already trained and saved.

If you need to train the model from scratch, use the following commands:

```bash
python3 -m models.download_dataset
python3 -m models.train
```

### Music Generation Setup
#### Install FluidSynth
FluidSynth is required for rendering audio:
```bash
brew install fluidsynth
```

#### Download Soundfonts
Soundfonts are to be stored in data/soundfonts. You can download them from:
[FluidR3 GM Soundfont](https://member.keymusician.com/Member/FluidR3_GM/index.html)


### Environment Variables
Create a .env file in the project directory with the following content:
```bash
OPENAI_API_KEY=<your_api_key>
```

### Run the Application

Start the Streamlit application:
```bash
streamlit run run.py
```