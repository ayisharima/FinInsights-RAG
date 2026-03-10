# FinInsights-RAG 📈
An advanced RAG system for deep financial analysis of SEC/NSE filings.

## Key Features
- **High-Fidelity Parsing:** Uses LlamaParse to accurately interpret complex financial tables.
- **Hybrid Data Support:** Processes both PDF annual reports and Text earnings transcripts.
- **Evidence-Based:** Every answer includes citations and relevance scores from the source files.
- **Cost-Effective:** Uses local embeddings (BGE-Small) and Groq (Llama 3.3) for $0 operational cost.

## Tech Stack
- **Framework:** LlamaIndex
- **LLM:** Groq (Llama 3.3 70B)
- **Embedding:** HuggingFace BGE
- **UI:** Streamlit

  ## Sample Questions
- What was NVIDIA's total revenue for fiscal year 2025, and what percentage of that came from Data Center sales?
- Compare the Research and Development (R&D) expenses of Tata Motors between 2023 and 2024
- How does Apple describe the impact of global supply chain constraints on its product margins?
