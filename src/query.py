import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

# Setup same free models
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

def ask_financial_question(question):
    # Load the brain from the storage folder
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    
    # Create the query engine with financial constraints
    query_engine = index.as_query_engine(
        response_mode="compact",
        streaming=False
    )
    
    # Strict prompt for professionalism
    full_query = f"""
    You are a professional financial analyst. 
    Use the provided context to answer the question. 
    If the data is not in the context, say you don't know. 
    Question: {question}
    """
    
    response = query_engine.query(full_query)
    return response

if __name__ == "__main__":
    user_q = input("Ask a financial question (e.g., 'What are Nvidia's risks?'): ")
    result = ask_financial_question(user_q)
    print("\n--- ANALYST REPORT ---")
    print(result)