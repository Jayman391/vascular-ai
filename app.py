import streamlit as st
import json
import os
import pandas as pd
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
    return ' '.join(' '.join(f'{key} {value}' for key, value in author.items()) for author in author_list)

def preprocess_data(dir="Pubmed_Data"):
    data_as_strs = []
    for filename in os.listdir(dir):
        with open(f'{dir}/{filename}') as file:
            data = json.load(file)
        for datum in data:
            prompt = f"""\
Paper: {datum.get("title", "")}
Abstract: {datum.get("abstract", "")}
Results: {datum.get("results", "")}
Authors: {process_author_list_to_string(datum.get('authors', ''))}
Keywords: {' '.join(datum.get("keywords", ''))}
"""
            data_as_strs.append(prompt)
    return data_as_strs

def ingest_and_prepare_vector_store():
    data_as_strs = preprocess_data()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    df = pd.DataFrame(data_as_strs, columns=["text"])
    df.to_csv("data.csv", index=False)

    loader = CSVLoader(file_path='data.csv')
    documents = loader.load()

    vector_store = Chroma(embedding_function=embeddings)
    vector_store.add_documents(documents=documents)

    return vector_store

# ------------------- Response Generation ------------------- #
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
        return "\n\n".join(f"text: {doc.page_content} metadata: {doc.metadata}" for doc in docs)

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    chain = (
        {"context": traversal_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ------------------- Streamlit App ------------------- #
def main():
    st.title("PubMed Research Question Answering")

    vector_store = ingest_and_prepare_vector_store()
    chain = setup_chain(vector_store)

    question = st.text_input("Ask a question about the PubMed articles:")

    if st.button("Submit"):
        with st.spinner("Generating answer..."):
            response = chain.invoke(question)
        st.markdown("### Answer:")
        st.write(response)

if __name__ == "__main__":
    main()