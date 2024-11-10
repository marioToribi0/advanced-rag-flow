import sys
sys.path.append("./")

from langchain_community.tools import TavilySearchResults
from graph.state import GraphState
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    search_results = web_search_tool.invoke(question)
    
    search_documents = []
    
    for search in search_results:
        search_documents.append(Document(
            page_content=search["content"],
            metadata={"source": search["url"]}
        ))
    
    return {"question": question,
            "documents": documents+search_documents
            }

if __name__ == "__main__":
    result = web_search(state={
        "question": "agent documents",
        "documents": []
    })
    
    print(result)