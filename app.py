import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from langchain_openai import ChatOpenAI  # Updated import from langchain_openai
from langchain_chroma import Chroma
from langchain_google_community import GoogleSearchAPIWrapper  # Updated import from langchain_google_community

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from src.preprocess import ingest_and_prepare_vector_store

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class State(MessagesState):
    websearch:bool = False
    rag_response:str
    websearch_response:str
    full_response:str

def entry_point(state: State):
    new_message = AIMessage(content="""\n \n Hi! I'm PubMedBot, the Chatbot that uses Pubmed Research to assist Physicians in Clinical Settings. How can I help you today? \n \n
    1. Find a Research Paper \n
    2. Answer a Domain Specific Question \n
    3. Help Make a Diagnosis
    """)
    return {"messages": [new_message], "extra_field": 10}

def initial_request(state: State):
    feedback = interrupt("Please ask your question and provide any additional information that may help me assist you better : \n")
    return {"messages": feedback}

def websearch_request(state: State):
    feedback = interrupt("Would you like to search the web? \n")
    if feedback.lower()[0] == "y":
        return {"websearch": True}
    else:
        return {"websearch": False}
    
def route_request(state: State):
    if state["websearch"] == True:
        return ["rag_pipeline", "websearch_pipeline"]
    else:
        return ["rag_pipeline"]

def rag(state: State):
    question = state['messages'][2].content
    vectorstore = ingest_and_prepare_vector_store()
    return {"rag_response": vectorstore.similarity_search(question, k=5)}

def websearch(state : State):
    question = state['messages'][2].content
    search = GoogleSearchAPIWrapper(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        google_cse_id=os.environ["GOOGLE_CSE_ID"],
    )
    return {"websearch_response": search.run(question)}

@traceable
def response(state: State):
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
    websearch_response = ""
    rag_response = state["rag_response"]
    if state.get("websearch_response") is not None:
        websearch_response = state["websearch_response"]
    question = state['messages'][2].content

    prompt = ChatPromptTemplate.from_template(
        """
        You are a medical expert. Your job is to provide a detailed and accurate answer to a user's question based on the context provided from traditional RAG and web search results.
        Please provide a comprehensive answer to the question, incorporating information from both sources. 

        Please have a separate section to explicitly state each citation and respective authors as well as including the sources of information used in your answer.
        Cite each paper with its respective authors as one chunk. Have all papers be assigned a chunk. Note that a list of authors will appear at the end of each document retreived from the RAG pipeline.

        Here is the question: {question}

        Here is the context from the web search pipeline: {websearch_response}

        Here is the context from the RAG pipeline: {rag_response}
        """
    )

    chain = RunnablePassthrough() | prompt | llm | StrOutputParser()

    return {"full_response": chain.invoke({
        "question": question,
        "rag_response": rag_response,
        "websearch_response": websearch_response,
    })}


def initialize_graph():

    graph = StateGraph(State)

    graph.add_node("initial_node",entry_point)
    graph.add_node("initial_request", initial_request)
    graph.add_node("websearch_request", websearch_request)
    graph.add_node("rag_pipeline", rag)
    graph.add_node("websearch_pipeline", websearch)
    graph.add_node("response", response)


    graph.add_edge(START, "initial_node")
    graph.add_edge("initial_node", "initial_request")
    graph.add_edge("initial_request", "websearch_request")
    graph.add_conditional_edges("websearch_request", route_request, ["rag_pipeline", "websearch_pipeline"])
    graph.add_edge("rag_pipeline", "response")
    graph.add_edge("websearch_pipeline", "response")
    graph.add_edge("response", END)


    memory = MemorySaver()

    graph = graph.compile(checkpointer=memory)

    return graph


def prompt_user(prompt : str):
    return input(prompt)

def run_program():
    graph = initialize_graph()
    user_input = {"messages": ""}
    thread = {"configurable": {"thread_id": "1"}}

    events = graph.stream(user_input, thread, stream_mode="updates")
    
    while True:
        try:
            next_event = next(events)
        except StopIteration:
            print("Thank you for using PubMedBot! If you have any more questions, rerun the program")
            break

        event_key = list(next_event.keys())[0]

        # If the event is an interrupt from a human interaction prompt.
        if event_key == "__interrupt__":
            prompt_text = next_event["__interrupt__"][0].value
            user_input_value = prompt_user(prompt_text)
            command = Command(resume=user_input_value)
            events = graph.stream(command, thread, stream_mode="updates")
        
        # Display the plain text from the entry point.
        elif event_key == "initial_node":
            messages = next_event["initial_node"].get("messages", [])
            if messages and hasattr(messages[0], "content"):
                print(messages[0].content)
        
      
        
        # When the response node returns, print only the final response text.
        elif event_key == "response":
            final_response = next_event["response"].get("full_response", "")
            print(final_response)
        
        # End the loop if an END event is received.
        if event_key == END:
            break


if __name__ == "__main__":
    run_program()