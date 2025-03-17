import streamlit as st
import json
import yaml
from yaml.loader import SafeLoader
import os
import pandas as pd
import streamlit_authenticator as stauth
import sqlite3
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import time
from typing_extensions import Annotated, TypedDict
from langsmith import Client, traceable

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "pubmed_qa"
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]

# ------------------- Page Configuration and Custom CSS ------------------- #
st.set_page_config(page_title="PubMed Research Q&A", page_icon="📚", layout="wide")

custom_css = """
<style>
/* General styling for headings */
h1, h2, h3 {
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Add padding to the main container */
.reportview-container, .main {
    padding: 2rem;
}
/* Customize sidebar appearance */
.sidebar .sidebar-content {
    background-color: #f0f2f6;
    padding: 1rem;
}
/* Style for Streamlit buttons */
.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 0.5em 1em;
    border: none;
    border-radius: 4px;
    font-size: 16px;
}
/* Additional styling for inputs and markdown */
.stTextInput label, .stMarkdown {
    font-size: 18px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------- Data Ingestion Functions ------------------- #
def process_author_list_to_string(author_list):
    return " ".join(
        " ".join(f"{key} {value}" for key, value in author.items())
        for author in author_list
    )

def preprocess_data(dir="Pubmed_Data"):
    data_as_strs = []
    for filename in os.listdir(dir):
        with open(f"{dir}/{filename}") as file:
            data = json.load(file)
        for datum in data:
            prompt = f"""\
Paper: {datum.get("title", "")}
Abstract: {datum.get("abstract", "")}
Results: {datum.get("results", "")}
Authors: {process_author_list_to_string(datum.get("authors", ""))}
Keywords: {" ".join(datum.get("keywords", ""))}
"""
            data_as_strs.append(prompt)
    return data_as_strs

def ingest_and_prepare_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    persist_dir = "./pubmed_db"
    collection_name = "pubmed_vascular"

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    else:
        data_as_strs = preprocess_data()
        df = pd.DataFrame(data_as_strs, columns=["text"])
        df.to_csv("data.csv", index=False)
        loader = CSVLoader(file_path="data.csv")
        documents = loader.load()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        vector_store.add_documents(documents=documents)
    return vector_store

# ------------------- Question Generation and DB Storage ------------------- #
def init_db():
    conn = sqlite3.connect("questions.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS subquestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_question TEXT,
            subquestion TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_subquestions(main_question, subquestions):
    conn = sqlite3.connect("questions.db")
    c = conn.cursor()
    for q in subquestions:
        c.execute(
            "INSERT INTO subquestions (main_question, subquestion) VALUES (?, ?)",
            (main_question, q),
        )
    conn.commit()
    conn.close()

def get_subquestions(main_question):
    conn = sqlite3.connect("questions.db")
    c = conn.cursor()
    c.execute(
        "SELECT subquestion FROM subquestions WHERE main_question = ?", (main_question,)
    )
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

@traceable
def generate_subquestions(main_question, llm) -> dict:
    prompt_template = ChatPromptTemplate.from_template(
        """Generate three sub-questions that break down the following research question into key aspects:

Research Question: {question}

Sub-Questions:"""
    )
    # Chain now takes a dictionary and returns the raw text output.
    chain = (
        {"question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser()
    )
    # Invoke with dictionary input
    result = chain.invoke({"question": main_question})
    # Split the output into sub-questions and return as dict.
    subquestions = [q.strip() for q in result.split("\n") if q.strip()]
    return {"subquestions": subquestions}

# ------------------- Retrieval Functions ------------------- #
def retrieve_context_for_subquestions(subquestions, vector_store):
    context_parts = []
    for subq in subquestions:
        docs = vector_store.similarity_search(subq, k=3)
        formatted_docs = "\n".join(
            f"Text: {doc.page_content}\nMetadata: {doc.metadata}" for doc in docs
        )
        context_parts.append(f"Sub-question: {subq}\nResults:\n{formatted_docs}")
    return "\n\n".join(context_parts)

def retrieve_citations(query, vector_store):
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join(
        f"Text: {doc.page_content}\nMetadata: {doc.metadata}" for doc in docs
    )

# ------------------- Sub Question for Answer Generation ------------------- #
@traceable
def sub_question_chain(main_question, sub_context, citations_context) -> dict:
    final_prompt = ChatPromptTemplate.from_template(
        """Using the context below, which includes results from sub-questions and additional citation information from the dataset, answer the original research question. Provide citations for each source used.

Sub-Questions Context:
{sub_context}

Main Query Citations:
{citations}

Research Question: {question}"""
    )
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    chain = final_prompt | llm | StrOutputParser()
    inputs = {
        "sub_context": sub_context,
        "citations": citations_context,
        "question": main_question,
    }
    # Wrap the output in a dict with key "final_answer"
    answer = chain.invoke(inputs)
    return {"final_answer": answer}

# ------------------- Direct Answer Pipeline ------------------- #
@traceable
def direct_pipeline(vector_store, question: str) -> dict:
    # For retrieval, we use a simple similarity search (or your graph retriever logic)
    docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join(
        f"Text: {doc.page_content}\nMetadata: {doc.metadata}" for doc in docs
    )
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
Return your answer as well as the papers cited and their authors.

Context: {context}

Question: {question}"""
    )
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "documents": docs}

# ------------------- Fusion of Final and Direct Answers ------------------- #
@traceable  
def fuse_results(final_ans: str, direct_ans: str) -> dict:
    fusion_prompt = ChatPromptTemplate.from_template(
        """You are given two answers to a research question:

Final Answer with Citations:
{final_ans}

Direct Answer:
{direct_ans}

Fuse these two answers into one comprehensive, clear, and well-cited final response that leverages the strengths of both.
Final Fused Answer:"""
    )
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    chain = fusion_prompt | llm | StrOutputParser()
    inputs = {"final_ans": final_ans, "direct_ans": direct_ans}
    fused = chain.invoke(inputs)
    return {"answer": fused}

# ------------------- Streamlit App ------------------- #
def main():
    # Sidebar with logo and instructions
    with st.sidebar:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_column_width=True)
        st.header("PubMed Q&A")
        st.markdown(
            """
            This app allows you to ask questions about PubMed research articles and get a comprehensive, well-cited answer.
            **How to use:**
            - Login using your credentials.
            - Enter your research question.
            - Review the direct answer, generated sub-questions, and the final synthesized answer.
            - If you like the answer, click the button to log it to our LangSmith dataset.
            """
        )

    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

    try:
        authenticator.login()
    except Exception as e:
        st.error(e)

    if st.session_state.get("authentication_status"):
        time.sleep(1)
        st.write(f"Welcome *{st.session_state.get('name')}*")
        st.title("PubMed Research Question Answering")
        vector_store = ingest_and_prepare_vector_store()

        main_question = st.text_input("Ask a question about the PubMed articles:")

        if st.button("Submit"):
            with st.spinner("Processing..."):
                # Direct answer step (with retrieval)
                direct_output = direct_pipeline(vector_store, main_question)
                st.markdown("### Direct Answer:")
                st.write(direct_output["answer"])

                # Generate sub-questions using an LLM
                llm_for_questions = init_chat_model("gpt-4o-mini", model_provider="openai")
                subq_dict = generate_subquestions(main_question, llm_for_questions)
                subquestions = subq_dict["subquestions"]
                st.markdown("### Generated Sub-Questions:")
                st.write(subquestions)

                # Store and retrieve sub-questions in/from the DB
                init_db()
                store_subquestions(main_question, subquestions)
                retrieved_subquestions = get_subquestions(main_question)

                # Retrieve context for sub-questions and citations
                sub_context = retrieve_context_for_subquestions(retrieved_subquestions, vector_store)
                citations_context = retrieve_citations(main_question, vector_store)

                # Generate final answer using sub-questions and citations
                final_dict = sub_question_chain(main_question, sub_context, citations_context)
                final_answer_raw = final_dict["final_answer"]

                # Fuse both answers into one comprehensive response
                fused_dict = fuse_results(final_answer_raw, direct_output["answer"])

                st.markdown("### Full Answer with Additional Synthesis:")
                st.write(fused_dict["answer"])

                if st.button("Log to LangSmith Dataset"):

                    client = Client(api_key=st.secrets["LANGSMITH_API_KEY"])

                    # Filter runs to add to the dataset where name = fuse_results
                    runs = client.list_runs(
                        project_name="pubmed_qa",
                        filter='eq(name, "fuse_results")',
                        select=["inputs", "outputs"]
                    )

                    for run in list(runs):
                        print(run.__class__)
                        client.create_example(
                            inputs=run.inputs,
                            outputs=run.outputs,
                            dataset_name="pubmed_qa"                        
                        )

        authenticator.logout()

    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")
    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")

if __name__ == "__main__":
    main()
