from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Union
from .theory import TheoryAgent
from .analysis import AnalysisAgent
from .generation import GenerationAgent

CHECKPOINT_PATH = "models/checkpoints/best_model.pt"

class State(TypedDict):
    analysis_complete: bool
    theory_complete: bool
    generation_complete: bool
    analysis_error: Union[str, None]
    theory_error: Union[str, None]
    generation_error: Union[str, None]
    next_agent: Union[str, None]
    midi_path: Union[str, None]
    genre_prediction: Union[dict, None]
    style_analysis: Union[dict, None]
    theory_results: Union[str, None]
    generation_parameters: Union[dict, None]
    generation_results: Union[dict, None]

class CoordinatorAgent:
    def __init__(self):
        self.theory_agent = TheoryAgent(CHECKPOINT_PATH)
        self.analysis_agent = AnalysisAgent(CHECKPOINT_PATH)
        self.generation_agent = GenerationAgent(CHECKPOINT_PATH)
        self.workflow = self.build_workflow()

    def process(self, state: Dict) -> Dict:
        """Process current state and decide next steps"""

        # If MIDI is uploaded, send to analysis branch
        if state.get("midi_path") and not state.get("analysis_complete"):
            state["next_agent"] = "analysis"
        # If analysis is complete, proceed to theory
        elif state.get("analysis_complete") and not state.get("theory_complete"):
            state["next_agent"] = "theory"
        # If generation parameters are set, send to generation
        elif state.get("generation_parameters") and not state.get("generation_complete"):
            state["next_agent"] = "generation"
        else:
            state["next_agent"] = END
            
        return state

    def build_workflow(self) -> StateGraph:
        """Define the workflow using StateGraph and compile it."""
        # Initialize graph
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("coordinator", self.process)
        graph_builder.add_node("analysis", self.analysis_agent.process)
        graph_builder.add_node("theory", self.theory_agent.process)
        graph_builder.add_node("generation", self.generation_agent.process)

        # Add edges for analysis branch
        graph_builder.add_edge("analysis", "coordinator")
        graph_builder.add_edge("theory", "coordinator")

        # Add edge for generation branch
        graph_builder.add_edge("generation", "coordinator")

        # Add conditional routing from coordinator
        graph_builder.add_conditional_edges(
            "coordinator",
            lambda x: x["next_agent"]
        )

        # Set entry point
        graph_builder.set_entry_point("coordinator")
        
        return graph_builder.compile()

    def run(self, initial_state: Dict) -> Dict:
        """Run the compiled workflow."""
        # Ensure next_agent is initialized
        if "next_agent" not in initial_state:
            initial_state["next_agent"] = "coordinator"
            
        result = self.workflow.invoke(initial_state)
        return result