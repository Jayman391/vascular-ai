FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install streamlit langchain langchain-community langchain-chroma langchain-graph-retriever pandas sentence-transformers chromadb openai streamlit-authenticator

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
