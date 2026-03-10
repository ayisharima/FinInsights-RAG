import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

def ingest_data():
    # 1. Setup Free Embedding Model (Runs on your CPU)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Setup Free LLM (Groq - Llama 3)
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    # 3. Initialize Parser
    parser = LlamaParse(result_type="markdown", verbose=True)
    file_extractor = {".pdf": parser}

    # 4. Load Documents
    print("Parsing documents (Free via LlamaParse/HuggingFace)...")
    documents = SimpleDirectoryReader("./dataset", file_extractor=file_extractor).load_data()

    # 5. Create and Save Index
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="./storage")
    
    print("SUCCESS: Your local brain is ready in the 'storage' folder.")
    return index

if __name__ == "__main__":
    ingest_data()