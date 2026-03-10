import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# 1. Page Configuration
st.set_page_config(page_title="FinInsights RAG", layout="wide")
st.title("📊 FinInsights: Intelligent Financial Analysis")
st.markdown("---")

load_dotenv()

# 2. Setup Models (Same as your query.py)
@st.cache_resource
def load_rag_engine():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=3)

query_engine = load_rag_engine()

# 3. User Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ask a Deep Financial Question")
    user_input = st.text_input("Example: 'Compare the risk factors of Nvidia vs Apple'", placeholder="Type here...")
    
    if st.button("Analyze Reports"):
        if user_input:
            with st.spinner("Analyzing filings..."):
                response = query_engine.query(user_input)
                
                st.markdown("### 🤖 Analyst Response")
                st.write(response.response)
        else:
            st.warning("Please enter a question.")

with col2:
    st.subheader("📌 Source Evidence (Citations)")
    if 'response' in locals():
        # This part shows exactly WHERE the AI found the information
        for i, node in enumerate(response.source_nodes):
            file_name = node.metadata.get('file_name', 'Unknown Source')
            
            with st.expander(f"Source {i+1} - Relevance Score: {node.score:.2f}"):
                st.write(node.text)
                st.json(node.metadata) # Shows filename and page number
    else:
        st.info("Ask a question to see the supporting evidence from the documents.")