from graph.state import GraphState
from ingestion import retriever

def retrive(state: GraphState):
    documents = retriever.invoke(state["question"])
    return {"documents": documents}