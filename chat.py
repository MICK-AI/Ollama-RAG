from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

#vector embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
openai_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="RAG_OLLAMA"
)

#  Take the user input
user_query = input("Hey! ask something!\n")

search_results = vector_db.similarity_search(query=user_query,k=6)


context = "\n\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata.get('Page_label', 'Unknown')}\n"
    f"File Location: {result.metadata.get('source', 'Unknown')}"
    for result in search_results
])

SYSTEM_PROMPT = f"""You are an helpfull assistant who answers the user
quesry based on the available context retrieved from a PDF file along with page_contents
and page number.

you should only answer the user based on the following context and navigate the
user query based on available context retrieved from a PDF file along with page_contents
and page number.

You should  only ans the user based on the following context and  navigate the user to 
open the right page number to know more.

Context:
{context}
"""


response = openai_client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_query}
        ]
)

print(f"🤖: {response.choices[0].message.content}")