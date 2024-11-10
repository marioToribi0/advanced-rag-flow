from graph.state import GraphState
from ingestion import retriever

def retrive(state: GraphState):
    print("--RETRIEVE--")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}