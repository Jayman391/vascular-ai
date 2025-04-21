import sys
from langchain_core.tools import tool
from src.api.preprocess import *

@tool
def pubmed_rag(state):
    """Searches PubMed using RAG"""
    question = state['query']
    try:
        vectorstore = ingest_and_prepare_vector_store()
    except Exception as e:
        sys.stdout.write(f"\nError preparing vector store: {e}\n")
        return str(e)
    
    if vectorstore:
        try:
            sys.stdout.write("\nSearching The Pubmed Database\n")
            return vectorstore.similarity_search(question, k=5)
        except Exception as e:
            sys.stdout.write(f"\nError searching PubMed: {e}\n")
            return str(e)