from dotenv import load_dotenv

load_dotenv()

from graph.chains.nodes.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

def test_retrival_grader_answer_yes() -> None:
    question = "prompt engineering"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    
    assert res.binary_score == "yes"
    
    