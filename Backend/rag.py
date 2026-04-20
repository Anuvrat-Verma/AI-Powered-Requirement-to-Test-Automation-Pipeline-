# rag.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# Initialize the local embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
DB_DIR = "./chroma_db"


def ingest_documents(folder_path="docs"):
    """Loads all .txt files in a folder and saves them to ChromaDB."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created '{folder_path}' directory. Put your reference .txt files here!")
        return

    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())

    if not docs:
        print("⚠️ No documents found to ingest.")
        return

    # Chunk the documents into smaller pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Store them in the local ChromaDB
    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
    print("✅ Documents successfully ingested into vector database!")


def retrieve_context(query: str) -> str:
    """Searches the database for info related to the user's requirements."""
    if not os.path.exists(DB_DIR):
        return ""  # No database exists yet

    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Retrieve top 3 most relevant chunks
    docs = db.similarity_search(query, k=3)

    # Combine them into a single string
    context = "\n".join([doc.page_content for doc in docs])
    return context