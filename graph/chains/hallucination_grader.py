from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableSequence

from llm_models import llm

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    
    binary_score: Literal['yes', 'no'] = Field(
        description="Answer is supported by a set of retrieved facts, 'yes' or 'no'"
    )
   
structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
system = """You are a grader assessing whether an LLM generation is grounded in or supported by a set of retrieved facts.\n
            Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in or supported by the set of facts.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader