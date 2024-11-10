from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from typing import Literal
from langchain_openai import ChatOpenAI
from config import BASE_URL_OLLAMA

llm = ChatOpenAI(temperature=0)
# llm = ChatOllama(
#     model="llama3.1",
#     base_url=BASE_URL_OLLAMA
# )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    
    binary_score: Literal['yes', 'no'] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question\n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader