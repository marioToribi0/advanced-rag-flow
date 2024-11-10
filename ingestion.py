from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import BASE_URL_OLLAMA

load_dotenv()

embeddings = OllamaEmbeddings(
    model="llama3.1",
    base_url=BASE_URL_OLLAMA
)

urls = [
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma", 
#     embedding=embeddings,
#     persist_directory="./.chroma"
# )

retriever = Chroma(
    collection_name="rag-chroma", 
    embedding_function=embeddings,
    persist_directory="./.chroma"
).as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is prompt engineering?")

# show the retrieved document's content
print([doc.page_content for doc in retrieved_documents[:3]])