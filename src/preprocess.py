import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader

# ------------------- Preprocessing & Setup ------------------- #

def ingest_and_prepare_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )
    persist_dir = "/Users/jason/Desktop/projects/streamlit-graphrag/pubmed_db"
    collection_name = "pubmed_vascular"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    
    if not os.path.exists(f'{persist_dir}/chroma.sqlite3'):

        # Define file paths and batch parameters
        input_file_path = "/Users/jason/Desktop/projects/streamlit-graphrag/Pubmed_Data/data.txt"
        batch_size = 100
        batch_dir = "/Users/jason/Desktop/projects/streamlit-graphrag/Pubmed_Data/batches"
        os.makedirs(batch_dir, exist_ok=True)
        batch_files = []

        # Read the original file in batches and write each batch to its own file
        with open(input_file_path, "r", encoding="utf-8") as infile:
            batch_number = 0
            batch_lines = []
            for line in infile:
                # remove commas from the line
                line = line.replace(",", "")
                batch_lines.append(line)
                if len(batch_lines) >= batch_size:
                    batch_number += 1
                    batch_file = os.path.join(batch_dir, f"data_batch_{batch_number}.csv")
                    with open(batch_file, "w", encoding="utf-8") as outfile:
                        outfile.write(",".join(batch_lines))
                    batch_files.append(batch_file)
                    print(f"Wrote batch {batch_number} with {len(batch_lines)} lines to {batch_file}")
                    batch_lines = []
            # Write any remaining lines to a final batch file
            if batch_lines:
                batch_number += 1
                batch_file = os.path.join(batch_dir, f"data_batch_{batch_number}.csv")
                with open(batch_file, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(batch_lines))
                batch_files.append(batch_file)
                print(f"Wrote final batch {batch_number} with {len(batch_lines)} lines to {batch_file}")

        # Load each batch file using TextLoader and add its documents to the vector store
        for idx, batch_file in enumerate(batch_files, start=1):
            try:
                loader = CSVLoader(file_path=batch_file)
                documents = loader.load()
                vector_store.add_documents(documents)
                print(f"Loaded and added {len(documents)} documents from {batch_file} (Batch {idx}).")
            except Exception as e:
                print(f"Error loading and adding documents from {batch_file} (Batch {idx}): {e}")
    
    return vector_store

def format_docs(docs):
    return [doc["text"] for doc in docs if isinstance(doc, dict) and "text" in doc]