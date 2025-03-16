import streamlit as st
import json
import yaml
from yaml.loader import SafeLoader
import os
import pandas as pd
import streamlit_authenticator as stauth
import sqlite3
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Set environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# ------------------- Data Ingestion ------------------- #


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

    # Check if the persistent directory exists and is not empty
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        st.write("Loading existing Chroma database...")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    else:
        st.write("No existing database found. Ingesting new data...")
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


def generate_subquestions(main_question, llm):
    prompt_template = ChatPromptTemplate.from_template(
        """Generate three sub-questions that break down the following research question into key aspects:

Research Question: {question}

Sub-Questions:"""
    )
    chain = (
        {"question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser()
    )
    response = chain.invoke(main_question)
    # Assume the model returns newline-separated sub-questions
    subquestions = [q.strip() for q in response.split("\n") if q.strip()]
    return subquestions


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


# ------------------- Final Chain for Answer Generation ------------------- #


def final_chain(main_question, sub_context, citations_context):
    final_prompt = ChatPromptTemplate.from_template(
        """Using the context below, which includes results from sub-questions and additional citation information from the dataset, answer the original research question. Provide citations for each source used.

Sub-Questions Context:
{sub_context}

Main Query Citations:
{citations}

Research Question: {question}"""
    )
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    # Compose the chain without wrapping constants in a dict composition
    chain = final_prompt | llm | StrOutputParser()
    inputs = {
        "sub_context": sub_context,
        "citations": citations_context,
        "question": main_question,
    }
    return chain.invoke(inputs)


# ------------------- Original Direct Answer Chain ------------------- #


def setup_chain(vector_store):
    traversal_retriever = GraphRetriever(
        store=vector_store,
        edges=[
            ("Has symptom", "Has symptom"),
            ("Increases risk", "Increases risk"),
            ("Treated with", "Treated with"),
            ("May require", "May require"),
            ("Can lead to", "Can lead to"),
            ("Studied by", "Studied by"),
            ("Has outcome", "Has outcome"),
        ],
        strategy=Eager(k=5, start_k=5, max_depth=2),
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
Return your answer as well as the papers cited and their authors.

Context: {context}

Question: {question}"""
    )

    def format_docs(docs):
        return "\n\n".join(
            f"Text: {doc.page_content}\nMetadata: {doc.metadata}" for doc in docs
        )

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    chain = (
        {
            "context": traversal_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ------------------- Fusion of Final and Direct Answers ------------------- #


def fuse_results(final_ans, direct_ans):
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
    return chain.invoke(inputs)


# ------------------- Streamlit App ------------------- #


def main():
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
        authenticator.logout()
        st.write(f"Welcome *{st.session_state.get('name')}*")
        st.title("PubMed Research Question Answering")
        vector_store = ingest_and_prepare_vector_store()
        # Initialize the direct chain
        direct_chain = setup_chain(vector_store)
        main_question = st.text_input("Ask a question about the PubMed articles:")
        if st.button("Submit"):
            with st.spinner("Processing..."):
                # Initialize the questions database
                init_db()

                # Generate direct answer using the original chain
                direct_answer = direct_chain.invoke(main_question)

                st.markdown("### Direct Answer:")
                st.write(direct_answer)

                # Generate sub-questions using an LLM
                llm_for_questions = init_chat_model(
                    "gpt-4o-mini", model_provider="openai"
                )
                subquestions = generate_subquestions(main_question, llm_for_questions)
                st.write("Generated Sub-Questions:")
                st.write(subquestions)

                # Store the generated sub-questions in the DB
                store_subquestions(main_question, subquestions)

                # Retrieve sub-questions from the DB
                retrieved_subquestions = get_subquestions(main_question)

                # Retrieve context for each sub-question from the vector store
                sub_context = retrieve_context_for_subquestions(
                    retrieved_subquestions, vector_store
                )

                # Retrieve additional citation information for the main query
                citations_context = retrieve_citations(main_question, vector_store)

                # Generate final answer using sub-questions and citations
                final_answer = final_chain(
                    main_question, sub_context, citations_context
                )

                # Fuse both answers into one comprehensive response
                fused_answer = fuse_results(final_answer, direct_answer)

                st.markdown("### Full Answer with Additional Synthesis:")
                st.write(fused_answer)

    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")

    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")


if __name__ == "__main__":
    main()
