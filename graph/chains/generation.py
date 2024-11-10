from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from llm_models import llm

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()