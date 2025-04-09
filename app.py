from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.utilities import GoogleSearchAPIWrapper

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from src.preprocess import ingest_and_prepare_vector_store


import os
import warnings


os.environ["TOKENIZERS_PARALLELISM"] = "true"

from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore")

class State(MessagesState):
    websearch:bool = False
    rag_response:str
    websearch_response:str
    full_response:str

def initial_node(state: State):
    new_message = AIMessage(content="""Hi! I'm PubMedBot, the Chatbot that uses Pubmed Research to assist Physicians in Clinical Settings. 
    How can I help you today? \n \n
    1. Find a Research Paper \n
    2. Answer a Domain Specific Question \n
    3. Help Make a Diagnosis \n \n
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
    if state["websearch_response"]:
        websearch_response = state["websearch_response"]
    question = state['messages'][2].content

    prompt = ChatPromptTemplate.from_template(
        """
        You are a medical expert. Your job is to provide a detailed and accurate answer to a user's question based on the context provided from traditional RAG and web search results.

        Please have a separate section to explicitly state each citation and respective authors as well as including the sources of information used in your answer.

        Note that a list of authors will appear at the end of each document retreived from the RAG pipeline.


        Here is the question: {question}

        Please provide a comprehensive answer to the question, incorporating information from both sources. 

        Here is the context from the RAG pipeline: {rag_response}

        Here is the context from the web search pipeline: {websearch_response}

        """
    )

    chain = RunnablePassthrough() | prompt | llm | StrOutputParser()

    return {"full_response": chain.invoke({
        "question": question,
        "rag_response": rag_response,
        "websearch_response": websearch_response,
    })}
   

graph = StateGraph(State)

graph.add_node("initial_node",initial_node)
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


# Input
initial_input = {"messages": ""}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print(event)
    print("\n")

# Continue the graph execution
for event in graph.stream(
    Command(resume="Find me examples of research papers on pseudoaneurysms"), thread, stream_mode="updates"
):
    print(event)
    print("\n")


# Continue the graph execution
for event in graph.stream(
    Command(resume="Yes"), thread, stream_mode="updates"
):
    print(event)
    print("\n")

