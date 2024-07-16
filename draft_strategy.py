from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
# from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

from langchain_qdrant import RetrievalMode

import os

from qdrant_client import QdrantClient

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
QDRANT_URL_PROTOCOL = os.getenv("QDRANT_URL_PROTOCOL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_HOST_PORT = os.getenv("QDRANT_HOST_PORT")
FAST_API_ENV = os.getenv("FAST_API_ENV")

collection = "draft_strategy_data_v2"

# openai_api_jey = os.getenv("OPENAI_API_KEY")
openai_api_jey = "2ec6eea226eb409996e02532fb49142e"

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


def search_similar():
    url = f"{QDRANT_URL_PROTOCOL}{QDRANT_HOST}:{QDRANT_HOST_PORT}"
    client = QdrantClient(url=url)
    qdrant_client = Qdrant(collection_name=collection, client=client, embeddings=OpenAIEmbeddings())
    doc_found = qdrant_client.similarity_search_with_score("Where are the international office of the NFL?")
    print(doc_found)

# load_data_from_file()

def create_chunk():
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
    return text_splitter, documents


def add_document_with_metadata(text_splitter, documents):
    from langchain.schema import Document

    oppponent_team_stats_docs  = []

    for doc in text_splitter.split_documents(documents):
        print(doc)
        breakpoint()
        oppponent_team_stats_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "metadata_type": "oppponent_team_stats",
                    "source": doc.metadata['source'],
                },
            )
        )
        
        print("*****************")
        print("\n")

    url=f"{QDRANT_URL_PROTOCOL}{QDRANT_HOST}:{QDRANT_HOST_PORT}"
    embeddings = OpenAIEmbeddings()

    qdrant = QdrantVectorStore.from_documents(
        oppponent_team_stats_docs,
        embedding=embeddings,
        url=url,
        collection_name='garyn_collection',
    )

def meta_search():
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from qdrant_client import QdrantClient, models

    url = f"{QDRANT_URL_PROTOCOL}{QDRANT_HOST}:{QDRANT_HOST_PORT}"
    client = QdrantClient(url=url)
    qdrant_client = Qdrant(collection_name='garyn_collection', client=client, embeddings=OpenAIEmbeddings())

    # Define the metadata filter
    metadata_filter = Filter(
        must=[
            FieldCondition(
                key="metadata_type",
                match=MatchValue(value="oppponent_team_stats")
            )
        ]
    )

    query = "What are some key elements of the NFL's model for a successful modern sports league, and how has the league expanded both nationally and internationally?"
    # doc_found = qdrant_client.similarity_search_with_score(query=query, metadata_filter=metadata_filter)
    found_docs = qdrant_client.similarity_search_with_score(
        query=query,
        filter=metadata_filter,
    )
    print(found_docs)

# text_splitter, documents = create_chunk()
# add_document_with_metadata(text_splitter, documents)
meta_search()
# search_similar()