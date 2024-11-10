from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from config import BASE_URL_OLLAMA

# llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
llm = ChatOllama(
    model="llama3.1",
    base_url=BASE_URL_OLLAMA
)