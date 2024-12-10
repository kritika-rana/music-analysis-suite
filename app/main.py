import streamlit as st
from pathlib import Path
from agents import CoordinatorAgent
from .utils import populate_analysis_tab, populate_generation_tab, populate_theory_tab

def app():
    # Initialize workflow state
    state = {
        "analysis_complete": False,
        "theory_complete": False,
        "generation_complete": False,
        "error": None
    }

    # Initialize the Coordinator Agent
    coordinator = CoordinatorAgent()

    # Set up Streamlit page config
    st.set_page_config(
        page_title="Music Analysis Suite",
        page_icon="🎵",
        layout="wide"
    )

    # Main title and file uploader
    st.title("🎵 Music Analysis Suite")
    st.write("Upload MIDI files for analysis, theory exploration, and music generation.")

    # File uploader
    uploaded_file = st.file_uploader("Upload MIDI File", type=["mid", "midi"])
    if uploaded_file is not None:
        temp_midi_path = Path("data/inputs/temp_midi.mid")
        with open(temp_midi_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        state["midi_path"] = str(temp_midi_path)
    else:
        state["midi_path"] = None

    # Tabs for different functionalities
    tabs = st.tabs(["🎧 Analysis", "📖 Music Theory", "🎼 Generation"])

    # -------------------- Analysis Tab --------------------
    with tabs[0]:
        st.header("Analysis")

        if uploaded_file is not None:
            # Run the coordinator workflow
            with st.spinner("Analyzing your music..."):
                while not state["analysis_complete"] and not state.get("analysis_error"):
                    state = coordinator.run(state)

            # Handle the results
            if state.get("analysis_error"):
                st.error(f"Error: {state['analysis_error']}")
            else:
                populate_analysis_tab(state)

        else:
            st.info("Upload a file to get analysis.")


    # -------------------- Music Theory Tab --------------------
    with tabs[1]:
        st.header("Music Theory")
        if uploaded_file is not None:
            # Run the coordinator workflow
            with st.spinner("Analyzing your music..."):
                while not state["theory_complete"] and not state.get("theory_error"):
                    state = coordinator.run(state)

            # Handle the results
            if state.get("theory_error"):
                st.error(f"Error: {state['theory_error']}")
            else:
                populate_theory_tab(state)

        else:
            st.info("Upload a file to get analysis.")

    # -------------------- Generation Tab --------------------
    with tabs[2]:
        st.header("Generation")
        populate_generation_tab(state, coordinator)

if __name__ == "__main__":
    app()