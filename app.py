import os
from langchain_openai import ChatOpenAI  
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

from src.websearch import *
from src.dbsearch import *
from src.rag import *

# Define the state class
class State(MessagesState):
    question: str
    response: str

# Define available tools
tools = [arxiv_search, google_search, pubmed_rag]

# Function to prompt the user for input
def prompt_user(prompt: str):
    return input(prompt)

# Entry point for the agent
def entry_point(state: State):
    new_message = AIMessage(content="""\n \n Hi! I'm PubMedBot, the Chatbot that uses PubMed Research to assist Physicians in Clinical Settings. How can I help you today? \n \n
    1. Find a Research Paper 
    2. Answer a Domain Specific Question 
    3. Help Make a Diagnosis
    """)
    return {"messages": [new_message]}

# Function for querying the user
def user_question(question="\nPlease enter your question:\n\n"):
    return {"query": prompt_user(question)}

# New helper function: Only display the final response from the stream.
def print_final_response(stream):
    final_message = None
    for s in stream:
        # Keep updating the final_message with the last message of each stream output.
        final_message = s["messages"][-1]
    # Print only the final message if available.
    if final_message:
        if isinstance(final_message, tuple):
            print(final_message)
        else:
            final_message.pretty_print()

def main():
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])

    # Present initial welcome message.
    state = State(question="", response="")
    entry_response = entry_point(state)
    print(entry_response['messages'][0].content)
    user_query = user_question()

    while True:        
        # Dynamically construct the prompt using the fresh query.
        prompt = f"""
        You are a medical expert. Your job is to provide a detailed and accurate answer to a user's question based on the context provided from multiple web search and web journal database results.
        Please provide a comprehensive answer to the question, incorporating information from both sources.

        Please have a separate section to explicitly state each citation and respective authors as well as including the sources of information used in your answer.
        Cite each paper with its respective authors as one chunk. Have all papers be assigned a chunk. Note that a list of authors will appear at the end of each document retrieved from the RAG pipeline.
        By default, use the RAG database. If that yields no results, search on ArXiv. If that yields no results, search on Google.

        Here is the question: {user_query['query']}
        """

        # Reinitialize the agent with the updated prompt.
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=prompt,
        )

        print("\nProcessing your query... Please wait.")
        # Instead of printing incremental results, capture and print only the final response.
        final_stream = agent.stream(user_query, stream_mode="values")
        print_final_response(final_stream)

        user_query = user_question("\nAsk another question or leave blank to end the program:\n")
        # The loop will re-prompt for a new question.
        if not user_query['query']:
            print("Ending the program.")
            break

# Run the program
if __name__ == "__main__":
    main()
