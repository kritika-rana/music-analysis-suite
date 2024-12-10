import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def populate_analysis_tab(state):
    genre_prediction = state.get("genre_prediction", {})

    genres = list(genre_prediction.keys())
    probabilities = list(genre_prediction.values())
    max_genre_index = np.argmax(probabilities)
    predicted_genre = genres[max_genre_index]

    st.markdown(f"### Predicted Genre: **{predicted_genre}**")

    fig, ax = plt.subplots(figsize=(4, 2.5)) 
    ax.barh(genres, probabilities, color=plt.cm.Paired.colors, edgecolor='black')
    ax.set_xlabel("Probability", fontsize=8) 
    ax.set_title("Genre Prediction", fontsize=10, pad=8)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', labelsize=7) 
    plt.tight_layout()
    

    for i, (genre, prob) in enumerate(zip(genres, probabilities)):
        ax.text(prob + 0.02, i, f"{prob * 100:.1f}%", va='center', fontsize=8)

    st.pyplot(fig, use_container_width=False)

    # Display style analysis
    style_analysis = state.get("style_analysis", {})
    if style_analysis:
        st.subheader("Musical Style Analysis")
        
        # Primary characteristics
        st.markdown(f"**Primary Genre:** {style_analysis['primary_genre']}")
        
        # Genre influences
        st.markdown("**Genre Influences:**")
        for influence in style_analysis['genre_influences']:
            st.markdown(f"- {influence}")
        
        # Musical characteristics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Musical Characteristics:**")
            st.markdown(f"- Complexity: {style_analysis['complexity'].title()}")
            st.markdown(f"- Rhythm: {style_analysis['rhythm_character'].title()}")
        
        with col2:
            st.markdown("**Structural Elements:**")
            st.markdown(f"- Instrumentation: {style_analysis['instrumentation'].title()}")
            st.markdown(f"- Primary Tonality: {style_analysis['primary_tonality']}")

def populate_theory_tab(state):
    # Get theory analysis from state
    theory_analysis = state.get("theory_results", "No analysis available.")
    
    # Display the theory analysis
    st.markdown(theory_analysis)

def populate_generation_tab(state, coordinator):
    st.subheader("Generation Parameters")

    # Style selection
    style_options = [
        "classical", "jazz", "electronic", "rock", "acoustic"
    ]
    
    # Style selector
    selected_style = st.selectbox(
        "Select Music Style",
        options=style_options,
        help="Choose the style of music you want to generate"
    )
        

    # Generate button
    generate_button = st.button("Generate Random Music")

    if generate_button:
        with st.spinner("Generating music..."):
            state["generation_complete"] = False
            # Update the state style
            state["generation_parameters"] = {
                "music_style": selected_style
            }

            # Process music generation
            state = coordinator.run(state)

            if state.get("generation_error"):
                st.error(f"Error: {state['generation_error']}")
            else:
                try:
                    # Retrieve the saved MIDI file path
                    generated_wav_path = state["generation_results"]["generated_wav_path"]

                    # Ensure the file exists before playback
                    if not Path(generated_wav_path).exists():
                        st.error("Generated file not found.")
                        return

                    # Add audio player
                    st.subheader("Generated Music")
                    st.audio(generated_wav_path, format="audio/wav")
                    
                    # Display generation info
                    st.info(f"Generated a {selected_style} style piece")

                except Exception as e:
                    st.error(f"Error playing generated music: {e}")
    else:
        st.info("Select your style preference to generate music.")