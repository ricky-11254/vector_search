from sentence_transformers import SentenceTransformer

library = [
'i ate tofu today',
'remember to buy shampoo by this week',
'felt a little dizzy after the workout',
'scented candles in the Tom\'s house made me feel nostalgic',
]

model_name = 'roberta-base-nli-stsb-mean-tokens'
model = SentenceTransformer(model_name)

def create_embeddings(model,texts):
    # Encode text using the same model
    embeddings = model.encode(texts)
    return embeddings
# Embed the library
embedded_library = create_embeddings(model,library)

# Example query
query = 'purchase'
query_embedding = model.encode([query])[0]
# Calculate cosine similarity between query and each stored embedding
similarities = cosine_similarity([query_embedding], embedded_library)
similarities = similarities[0]  # Flatten the similarity matrix

# Sort and retrieve top results
top_n = 5  # Number of top results to retrieve
top_indices = similarities.argsort()[-top_n:][::-1]  # Indices of top similar embeddings

# Print top results
for idx in top_indices:
    print(f"Similarity: {similarities[idx]}")
    print(f"Paragraph: {library[idx]}")  # Access library list using idx
    print("="*50)

