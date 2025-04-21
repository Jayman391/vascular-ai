import os
import sys
from langchain_core.tools import tool
from src.api.preprocess import *

@tool
def pubmed_rag(state):
    """Searches PubMed using RAG"""
    question = state
    vectorstore = ingest_and_prepare_vector_store()
    sys.stdout.write("\nSearching The Pubmed Database\n")
    return vectorstore.similarity_search(question, k=5)