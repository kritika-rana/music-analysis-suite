from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import RegexParser
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
from typing import Dict, Tuple
from models.inference import MusicAnalyzer

class TheoryAgent:
    def __init__(self, checkpoint_path: str, data_dir: str = 'data/midi'):
        self.analyzer = MusicAnalyzer(checkpoint_path, data_dir)
        self.max_tokens = 1000  # Increased token limit for longer responses

    @st.cache_resource
    def initialize_vector_store(_self):
        """Initialize or load the vector store."""
        embeddings = OpenAIEmbeddings()

        # Load existing vector store if available, otherwise create a new one
        if os.path.exists("data/theory_docs/theory_faiss_index"):
            return FAISS.load_local("data/theory_docs/theory_faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            loader = DirectoryLoader("data/theory_docs/", glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local("data/theory_docs/theory_faiss_index")
            return vectorstore

    @st.cache_resource
    def initialize_models(_self):
        """Initialize and cache the analysis chain."""
        prompt_template = """You are a professional music theorist analyzing a piece of music. Using both the musical features provided and the following reference materials about music theory, provide an in-depth theoretical analysis.

        Reference Materials:
        ---------
        {context}
        ---------

        Musical Features:
        {question}

        Provide a comprehensive music theory analysis that covers:
        1. Harmonic Analysis:
           - Analyze the tonal center and harmonic progressions suggested by the chroma features
           - Discuss harmonic density and its implications for texture
           - Identify potential cadential patterns based on note density and duration patterns

        2. Rhythmic Analysis:
           - Examine the metric organization (time signature, beat patterns)
           - Discuss rhythmic complexity and groove based on note density and duration
           - Analyze the relationship between tempo and musical style

        3. Structural Analysis:
           - Evaluate the overall musical texture based on instrument variety and density
           - Analyze the relationship between pitch range and melodic construction
           - Discuss how the instrumentation supports the musical structure

        4. Style and Genre Implications:
           - Connect the theoretical elements to specific musical traditions
           - Discuss how the theoretical elements contribute to genre characteristics
           - Analyze any fusion of different musical styles


        Guidelines:
        - Use proper music theory terminology
        - Connect theoretical concepts to practical musical elements
        - Explain how different musical elements interact and support each other
        - Consider historical and stylistic context where relevant
        - Back up your analysis with specific features from the data
        
        Ensure to write your response in flowing paragraphs."""

        # Define the RegexParser with default values
        output_parser = RegexParser(
            regex=r"Answer:\s*(.*?)(?:\n*Score:\s*(\d+))?$",
            output_keys=["answer", "score"],
            default_output_key="answer"
        )

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
            output_parser=output_parser
        )

        chain = load_qa_chain(
            OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct", max_tokens=1000),
            chain_type="stuff",
            prompt=PROMPT
        )
        return chain

    def format_features(self, features: Dict) -> str:
        """Format musical features for theoretical analysis."""
        return f"""
        Musical Features Analysis:
        
        Tempo and Meter:
        - Tempo: {features.get('tempo', 'unknown')} BPM
        - Time Signature: {features.get('time_signature', '4/4')}
        - Beat Interval: {features.get('avg_beat_interval', 'unknown')} seconds
        - Rhythmic Regularity: {features.get('rhythm_regularity', 'unknown')}
        
        Harmonic Content:
        - Harmony Density: {features.get('harmony_density', 'unknown')}
        - Primary Tonality: {features.get('primary_tonality', 'unknown')}
        - Chroma Profile: {features.get('chroma_profile', {})}
        
        Texture and Structure:
        - Note Density: {features.get('note_density', 'unknown')} notes per second
        - Pitch Range: {features.get('pitch_range', 'unknown')} semitones
        - Average Note Duration: {features.get('avg_note_duration', 'unknown')} seconds
        - Dynamic Range: {features.get('dynamics_range', 'unknown')}
        
        Instrumentation:
        - Number of Instruments: {features.get('instrument_count', 'unknown')}
        - Instrument Variety: {features.get('instrument_variety', 'unknown')}
        - Includes Percussion: {features.get('has_drums', 'No')}
        """

    def process(self, state: Dict) -> Dict:
        """Process the current state and add music theory analysis."""
        try:
            midi_path = state.get("midi_path")
            if not midi_path:
                raise ValueError("No MIDI file path provided")

            features = self.analyzer.get_feature_description(midi_path)
            feature_desc = self.format_features(features)

            # Retrieve context from the vector store
            vector_store = self.initialize_vector_store()
            relevant_chunks = vector_store.similarity_search(feature_desc, k=3)

            # Generate theory analysis
            chain = self.initialize_models()
            results = chain({
                "input_documents": relevant_chunks,
                "question": f"Musical features:\n{feature_desc}\n\nProvide a detailed music theory analysis."
            })

            # Update state
            state["theory_results"] = results["output_text"]
            state["theory_complete"] = True
            
        except Exception as e:
            state["error"] = f"Theory analysis error: {str(e)}"
            
        return state