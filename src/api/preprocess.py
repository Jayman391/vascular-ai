import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader

# ------------------- Preprocessing & Setup ------------------- #

def create_embeddings():
    """Creates and returns the embeddings function."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

def create_vector_store(embeddings, persist_dir, collection_name="abstract_db"):
    """Creates and returns the vector store."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

def get_txt_files_from_directory(directory):
    """Returns a list of all .txt files in the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]

def process_batch(input_file_path, batch_size=100, batch_dir="./batches"):
    """Processes the input file in batches and writes each batch to separate files."""
    os.makedirs(batch_dir, exist_ok=True)
    batch_files = []
    batch_number = 0
    batch_lines = []

    with open(input_file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.replace(",", "")  # Remove commas from line
            batch_lines.append(line)
            if len(batch_lines) >= batch_size:
                batch_number += 1
                batch_file = os.path.join(batch_dir, f"data_batch_{batch_number}.txt")
                with open(batch_file, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(batch_lines))
                batch_files.append(batch_file)
                print(f"Wrote batch {batch_number} with {len(batch_lines)} lines to {batch_file}")
                batch_lines = []
        # Write any remaining lines to a final batch file
        if batch_lines:
            batch_number += 1
            batch_file = os.path.join(batch_dir, f"data_batch_{batch_number}.txt")
            with open(batch_file, "w", encoding="utf-8") as outfile:
                outfile.write("".join(batch_lines))
            batch_files.append(batch_file)
            print(f"Wrote final batch {batch_number} with {len(batch_lines)} lines to {batch_file}")
    
    return batch_files

def load_and_add_documents_to_vector_store(batch_files, vector_store):
    """Loads each batch file and adds documents to the vector store."""
    for idx, batch_file in enumerate(batch_files, start=1):
        try:
            loader = CSVLoader(file_path=batch_file)
            documents = loader.load()
            vector_store.add_documents(documents)
            print(f"Loaded and added {len(documents)} documents from {batch_file} (Batch {idx}).")
        except Exception as e:
            print(f"Error loading and adding documents from {batch_file} (Batch {idx}): {e}")

def ingest_and_prepare_vector_store(data_directory="/Users/jason/Desktop/projects/vascular-ai/data", persist_dir="/Users/jason/Desktop/projects/vascular-ai/db"):
    """Main function to ingest data, process it into batches, and add to vector store."""
    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings, persist_dir)
    
    # Check if the vector store is empty
    if len(vector_store.get()['documents']) == 0:
        # Get all .txt files in the data directory
        txt_files = get_txt_files_from_directory(data_directory)
        print(txt_files)
        
        batch_dir = "/Users/jason/Desktop/projects/vascular-ai/data/batches"
        all_batch_files = []

        # Process each .txt file in the directory
        for txt_file in txt_files:
            batch_files = process_batch(txt_file, batch_size=100, batch_dir=batch_dir)
            all_batch_files.extend(batch_files)

        # Load and add documents from all batch files to the vector store
        load_and_add_documents_to_vector_store(all_batch_files, vector_store)

    return vector_store
