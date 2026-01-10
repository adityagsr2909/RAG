import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
print("SentenceTransformer model loaded.")

texts = [
    "The cat sits on the mat.",
    "Dogs are great pets.",
    "I love programming in Python."
]
embeddings = model.encode(texts)
print("Text embeddings generated.") 

dimension = embeddings.shape[1]
print(f"Embedding dimension: {dimension}")

#- create a FAISS index
index = faiss.IndexFlatL2(dimension)
print("FAISS index created.")
index.add(embeddings)
print(f"Number of vectors in the index: {index.ntotal}")

# Example query
query_text = "Where did cat sit?"
query_vector = model.encode([query_text])[0]
print(f"Encoded query: '{query_text}'")

# Perform the query
k = 3  # number of nearest neighbors to retrieve
distances, indices = index.search(np.array([query_vector]), k)

# Process and print results
print("\nQuery Results:")
for i in range(len(indices[0])):
    print(f"Rank: {i+1}")
    print(f"Text: {texts[indices[0][i]]}")
    print(f"Distance: {distances[0][i]}")
    print()