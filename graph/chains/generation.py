from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from config import BASE_URL_OLLAMA

# llm = ChatOpenAI(temperature=0)
llm = ChatOllama(
    model="llama3.1",
    base_url=BASE_URL_OLLAMA
)
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()