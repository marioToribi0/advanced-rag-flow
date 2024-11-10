from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from config import BASE_URL_OLLAMA

llm = ChatOpenAI(temperature=0)
# llm = ChatOllama(
#     model="llama3.1",
#     base_url=BASE_URL_OLLAMA
# )

class GradeAnswer(BaseModel):

    binary_score: Literal['yes', 'no'] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
   
structured_llm_grader = llm.with_structured_output(GradeAnswer)
    
system = """You are a grader assessing whether an answer addresses / resolves a question \n
    Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader