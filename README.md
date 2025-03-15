Run 

```sh
docker build -t pubmed-streamlit-app .
docker run -d -p 8501:8501 pubmed-streamlit-app
```

Stop

```sh
docker stop pubmed-streamlit-app
```

don't forget to have a secrets.toml in .streamlit/

The app will be accessible at http://localhost:8501 
