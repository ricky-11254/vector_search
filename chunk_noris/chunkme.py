import dutils
dutils.init()
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Function to load text data from files
def load_text_files(directory):
    paragraphs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                paragraphs.extend(f.readlines())
    return paragraphs

model = SentenceTransformer(model_name)
# Function to generate embeddings for a chunk of paragraphs
def generate_embeddings(paragraphs_chunk, model_name='roberta-base-nli-stsb-mean-tokens'):
    embeddings = model.encode(paragraphs_chunk)
    return embeddings.tolist()

# Function to process text data in chunks
def process_in_chunks(paragraphs, chunk_size=100):
    num_chunks = (len(paragraphs) + chunk_size - 1) // chunk_size  # Calculate number of chunks
    chunked_embeddings = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(paragraphs))
        paragraphs_chunk = paragraphs[start_idx:end_idx]
        p46() 
        # Generate embeddings for the current chunk
        embeddings_chunk = generate_embeddings(paragraphs_chunk)
        chunked_embeddings.extend(embeddings_chunk)
    
    return chunked_embeddings

# Example usage
text_files_directory = '/root/evaluate-saliency-4/coding-practice/vector_search/path1/directory1/'
paragraphs = load_text_files(text_files_directory)

# Process in chunks
chunk_size = 50  # Adjust as needed based on memory and processing capabilities
chunked_embeddings = process_in_chunks(paragraphs, chunk_size)

# Save or further process chunked_embeddings
with open('chunked_embeddings.json', 'w') as f:
    json.dump(chunked_embeddings, f)

print(f"Chunked embeddings saved to 'chunked_embeddings.json'")
