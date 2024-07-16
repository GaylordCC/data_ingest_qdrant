from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
# from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

from langchain_qdrant import RetrievalMode

import os

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
QDRANT_URL_PROTOCOL = os.getenv("QDRANT_URL_PROTOCOL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_HOST_PORT = os.getenv("QDRANT_HOST_PORT")
FAST_API_ENV = os.getenv("FAST_API_ENV")

collection = "draft_strategy_data_v2"

openai_api_jey = os.getenv("OPENAI_API_KEY")

def azure_openai_embeddings():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="rotoembed",
        openai_api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    return embeddings 


def create_chunk_document():
    file="./nfl_description.txt"
    if file.endswith(".txt"):
        loader = TextLoader(file, encoding="utf-8")
    elif file.endswith(".pdf"):
        loader = PyMuPDFLoader(file)
    else:
        raise ValueError(
            "Unsupported file type. Currently only support .txt and .pdf files"
        )

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    return docs

def load_data_from_file():
    docs = create_chunk_document()
    # embeddings = azure_openai_embeddings()   # AzureOpenAI Embeddings   [NO FUNCIKONAAAAAAAAAAAAAA]
    embeddings = OpenAIEmbeddings()          # Open AI Embeddings
    url = f"{QDRANT_URL_PROTOCOL}{QDRANT_HOST}:{QDRANT_HOST_PORT}"
    qdrant = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=url,
        collection_name=collection,
    )
    
    
    query = "Where are the international office of the NFL?"
    # qdrant = Qdrant.from_documents(
    #     docs,
    #     url,
    #     collection,
    #     embeddings
    # )
    
    found_docs = qdrant.similarity_search_with_score(query)
    print(found_docs)
    document, score = found_docs[0]
    print(document.page_content)
    print(f"\nScore: {score}")

    
    print({ "found_docs": document.page_content, "score": score})


load_data_from_file()