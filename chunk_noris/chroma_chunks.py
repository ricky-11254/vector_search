import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize Chroma DB
client = chromadb.Client(Settings())
collection = client.create_collection("embeddings_collection")

# Function to load text data from files
def load_text_files(directory):
    paragraphs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                paragraphs.extend(f.readlines())
    return paragraphs

# Function to generate embeddings for a chunk of paragraphs
def generate_embeddings(paragraphs_chunk, model_name='roberta-base-nli-stsb-mean-tokens'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(paragraphs_chunk)
    return embeddings.tolist()

# Function to process text data in chunks and store in Chroma DB
def process_and_store_in_chroma(paragraphs, chunk_size=100):
    num_chunks = (len(paragraphs) + chunk_size - 1) // chunk_size  # Calculate number of chunks
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(paragraphs))
        paragraphs_chunk = paragraphs[start_idx:end_idx]
        
        # Generate embeddings for the current chunk
        embeddings_chunk = generate_embeddings(paragraphs_chunk)
        
        # Prepare data for upsert
        data = [{"embedding": embedding, "metadata": {"paragraph": paragraph}} 
                for embedding, paragraph in zip(embeddings_chunk, paragraphs_chunk)]
        
        # Store embeddings in Chroma DB
        collection.upsert(data)

# Example usage
text_files_directory = '/root/evaluate-saliency-4/coding-practice/vector_search/path1/directory1/'
paragraphs = load_text_files(text_files_directory)

# Process in chunks and store in Chroma DB
chunk_size = 50  # Adjust as needed based on memory and processing capabilities
process_and_store_in_chroma(paragraphs, chunk_size)

print(f"Embeddings stored in Chroma DB.")

# Function to perform search query using Chroma DB
def search_query(query, top_n=5, model_name='roberta-base-nli-stsb-mean-tokens'):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])[0].tolist()
    
    # Perform similarity search in Chroma DB
    results = collection.query(query_embedding, top_n=top_n)
    
    # Print top results
    for result in results:
        print(f"Similarity: {result['score']}")
        print(f"Paragraph: {result['metadata']['paragraph']}")
        print("="*50)

# Example search query
query = "Your query text goes here."
search_query(query)
