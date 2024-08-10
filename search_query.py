import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Function to load embeddings from JSON
def load_embeddings(input_file='embeddings.json'):
    with open(input_file, 'r') as f:
        embeddings_dict = json.load(f)
    paragraphs = embeddings_dict['paragraphs']
    embeddings = np.array(embeddings_dict['embeddings'])
    return paragraphs, embeddings

# Function to perform the search
def perform_search(query_paragraph, paragraphs, embeddings, model_name='roberta-base-nli-stsb-mean-tokens', k=3):
    # Load the pre-trained model
    model = SentenceTransformer(model_name)
    # Generate embedding for the query paragraph
    query_embedding = model.encode([query_paragraph])
    
    # Create and populate Faiss index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    # Perform the search
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]

# Load the embeddings
paragraphs, embeddings = load_embeddings()

# Example query paragraph
query_paragraph = "algorithm"

# Perform the search
indices, distances = perform_search(query_paragraph, paragraphs, embeddings)

# Print the results
print("Query Paragraph:", query_paragraph)
print("\nMost similar paragraphs:")

for i, idx in enumerate(indices):
    print(f"\nRank {i+1}:")
    print("Paragraph:", paragraphs[idx])
    print("Distance:", distances[i])

