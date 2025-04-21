import os
import sys
from langchain_core.tools import tool

from langchain_community.utilities import ArxivAPIWrapper

@tool
def arxiv_search(state):
    """Searches arxiv"""
    question = state
    search = ArxivAPIWrapper()
    response = search.run(question)
    sys.stdout.write("\nSearching Arxiv\n")
    return response