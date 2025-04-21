import os
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI  
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

from src.api.websearch import *
from src.api.dbsearch import *
from src.api.rag import *
from src.api.preprocess import *

vectorstore = ingest_and_prepare_vector_store()

# Initialize Flask app
app = Flask(__name__)

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

    final_stream = agent.stream({"query": user_query}, stream_mode="values")

    final_message = None
    for s in final_stream:
        final_message = s["messages"][-1]
    
    if final_message:
        # Ensure we're returning a plain string instead of any complex object
        return str(final_message) if isinstance(final_message, tuple) else str(final_message)
    return "No response found."


# Define the Flask route
@app.route('/query', methods=['POST'])
def query():
    # Extract user query from the POST request
    data = request.get_json()
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Get the response based on the user query
    response = get_response(user_query)

    return jsonify({"response": response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
