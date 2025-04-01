from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma

import os
import warnings

warnings.filterwarnings("ignore")

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])

from src.preprocess import format_docs


# ------------------- Langchain Pipelines ---------------------------------- #

@traceable
def graphrag_pipeline(vectorstore):
    retriever = GraphRetriever(
        store=vectorstore,
        edges=[
            ("Has symptom", "Has symptom"),
            ("Increases risk", "Increases risk"),
            ("Treated with", "Treated with"),
            ("May require", "May require"),
            ("Can lead to", "Can lead to"),
            ("Studied by", "Studied by"),
            ("Has outcome", "Has outcome"),
        ],
        strategy=Eager(start_k=5, max_depth=None),
    )
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
        Return your answer as well as the papers cited and their authors.

        Context: {context}

        Question: {question}"""
    )


    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


@traceable
def generate_subquestions_chain():
    prompt_template = ChatPromptTemplate.from_template(
        """Generate three sub-questions that break down the following research question into key aspects:

        Research Question: {question}

        Sub-Questions:"""
            )
    
    chain = (
        {"question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser()
    )
    
    return chain


@traceable
def subquestions_chain():
    prompt = ChatPromptTemplate.from_template(
        """Using the context below, which includes results from sub-questions and additional citation information from the dataset, 
        answer the original research question. Provide citations for each source used.

        Sub-Questions Context:
        {subq_context}

        Original Answer:
        {initial_ans}

        Research Question: {question}"""
        )
    
    chain = prompt | llm | StrOutputParser()

    return chain

@traceable
def final_fusion_chain():
    fusion_prompt = ChatPromptTemplate.from_template(
        """You are given two answers to a research question:

Initial Fused Answer:
{initial_ans}

Sub-Question Based Answer:
{subq_ans}

Fuse these answers into one comprehensive, clear, and well-cited final response that leverages the strengths of both.
Final Fused Answer:"""
    )
    chain = fusion_prompt | llm | StrOutputParser()
    return chain


