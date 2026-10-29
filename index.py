from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).parent/"The_stranger.pdf"
#Load this file in python program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

#Split the doc into smaller chunnk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)

chunks = text_splitter.split_documents(documents=docs)

# print("Total PDF pages:", len(docs))
# print("Total chunks:", len(chunks))

# print("\nSearching chunks for Salamano...")

# for i, chunk in enumerate(chunks):
#     if "Salamano" in chunk.page_content:
#         print(f"\nChunk index: {i}")
#         print("Metadata:", chunk.metadata)
#         print("Content:", chunk.page_content)

#vector embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="RAG_OLLAMA"
)

print("Indexing of the documents done...")