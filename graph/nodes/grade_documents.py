from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    documents = state["documents"]
    question = state["question"]
    
    websearch = False
    relevant_documents = []
    
    for document in documents:
        is_relevant: GradeDocuments = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )
        
        if is_relevant.binary_score=="no":
            websearch = True
        else:
            relevant_documents.append(document)
            
    return {"question": question, "documents": relevant_documents, "web_search": websearch}
    
    
    