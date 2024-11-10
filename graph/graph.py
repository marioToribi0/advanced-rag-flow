from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.state import GraphState
from graph.nodes import retrieve, grade_documents, web_search, generate

load_dotenv()

flow = StateGraph(GraphState)

flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(WEBSEARCH, web_search)
flow.add_node(GENERATE, generate)

def is_web_search_neccesary(state: GraphState):
    if (state["web_search"]==True):
        print("---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEBSEARCH---")
        return WEBSEARCH
    return GENERATE

flow.set_entry_point(RETRIEVE)
flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
flow.add_conditional_edges(GRADE_DOCUMENTS, is_web_search_neccesary, [WEBSEARCH, GENERATE])
flow.add_edge(WEBSEARCH, GENERATE)
flow.add_edge(GENERATE, END)

app = flow.compile()


