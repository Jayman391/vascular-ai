import os
import sys
from langchain_core.tools import tool

from langchain_google_community import GoogleSearchAPIWrapper  

@tool
def google_search(state):
    """Searches Google"""
    question = state
    search = GoogleSearchAPIWrapper(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        google_cse_id=os.environ["GOOGLE_CSE_ID"],
    )
    response = search.run(question)
    sys.stdout.write("\nSearching Google\n")
    return response

