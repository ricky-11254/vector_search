import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Function to generate and save embeddings
def generate_and_save_embeddings(paragraphs, model_name='roberta-base-nli-stsb-mean-tokens', output_file='embeddings.json'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(paragraphs)
    embeddings_dict = {"paragraphs": paragraphs, "embeddings": embeddings.tolist()}
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f)
    print(f"Embeddings saved to {output_file}")

# Example paragraphs
paragraphs = [
    "path1/directory1/file1.txt",
    "path2/directory2/file2.txt",
    "path2/directory2/file3.txt"
]

# Generate and save embeddings
generate_and_save_embeddings(paragraphs)

