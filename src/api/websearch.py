import os
import sys
from langchain_core.tools import tool

from langchain_google_community import GoogleSearchAPIWrapper  

@tool
def google_search(state):
    """Searches Google"""
    question = state
    try:
        search = GoogleSearchAPIWrapper(
            google_api_key=os.environ["GOOGLE_API_KEY"],
            google_cse_id=os.environ["GOOGLE_CSE_ID"],
        )
    except Exception as e:
        sys.stdout.write(f"\nError initializing Google Search API: {e}\n")
        return str(e)

    try:
        response = search.run(question)
        sys.stdout.write("\nSearching Google\n")
        return response
    except Exception as e:
        sys.stdout.write(f"\nError searching Google: {e}\n")
        return str(e)

