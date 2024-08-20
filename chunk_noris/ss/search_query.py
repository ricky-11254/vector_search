import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load chunked embeddings
with open('chunked_embeddings.json', 'r') as f:
    chunked_embeddings = json.load(f)

# Load paragraphs from text files directory (if not already loaded)
text_files_directory = '/root/evaluate-saliency-4/coding-practice/vector_search/path1/directory1/'
def load_text_files(directory):
    paragraphs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                paragraphs.extend(f.readlines())
    return paragraphs

paragraphs = load_text_files('/root/evaluate-saliency-4/coding-practice/vector_search/path2/directory2/')

# Example query
query = "algorithm"

# Encode query using the same model
model_name = 'roberta-base-nli-stsb-mean-tokens'
model = SentenceTransformer(model_name)
query_embedding = model.encode([query])[0]

# Calculate cosine similarity between query and each stored embedding
similarities = cosine_similarity([query_embedding], chunked_embeddings)
similarities = similarities[0]  # Flatten the similarity matrix

# Sort and retrieve top results
top_n = 5  # Number of top results to retrieve
top_indices = similarities.argsort()[-top_n:][::-1]  # Indices of top similar embeddings

# Print top results
for idx in top_indices:
    print(f"Similarity: {similarities[idx]}")
    print(f"Paragraph: {paragraphs[idx]}")  # Access paragraphs list using idx
    print("="*50)

