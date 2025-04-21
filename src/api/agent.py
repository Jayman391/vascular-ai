import os
from langgraph.prebuilt import create_react_agent
from src.api.websearch import *
from src.api.dbsearch import *
from src.api.rag import *
from langchain_openai import ChatOpenAI  

# Setup the LLM and tools

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
tools = [arxiv_search, google_search, pubmed_rag]

def get_response(user_query):
    prompt = f"""
    You are a medical expert. Your job is to provide a detailed and accurate answer to a user's question based on the context provided from multiple web search and web journal database results.
    Please provide a comprehensive answer to the question, incorporating information from both sources.

    Please have a separate section to explicitly state each citation and respective authors as well as including the sources of information used in your answer.
    Cite each paper with its respective authors as one chunk. Have all papers be assigned a chunk. Note that a list of authors will appear at the end of each document retrieved from the RAG pipeline.
    By default, use the RAG database. If that yields no results, search on ArXiv. If that yields no results, search on Google.

    Here is the question: {user_query}
    """

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )

    try : 
        final_stream = agent.stream({"query": user_query}, stream_mode="values")

        final_message = None
        for s in final_stream:
            final_message = s["messages"][-1]
        
        if final_message:
            return str(final_message) if isinstance(final_message, tuple) else str(final_message)
    
    except Exception as e:
        print(f"Error during agent processing: {e}")
        return str(e)