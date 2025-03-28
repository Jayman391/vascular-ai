import os

def make_environment_variables():
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["HF_TOKEN"] = "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"
    os.environ["LANGSMITH_API_KEY"] = "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"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "pubmed_qa"