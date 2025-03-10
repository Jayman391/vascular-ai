docker build -t pubmed-streamlit-app .
docker run -d -p 8501:8501 pubmed-streamlit-app
The app will be accessible at http://localhost:8501 