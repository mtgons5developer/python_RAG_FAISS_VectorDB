import numpy as np
import faiss
import pandas as pd

# Define vector dimension
dim = 128  

# Create FAISS index (L2 Distance)
index = faiss.IndexFlatL2(dim)
# Assume you already added vectors to the index
dim = 128
db_vectors = np.random.rand(100, dim).astype('float32')
index.add(db_vectors)

# Get all stored vectors
stored_vectors = index.reconstruct_n(0, index.ntotal)
print("Stored Vectors:\n", stored_vectors)

stored_vectors = index.reconstruct_n(0, index.ntotal)
df = pd.DataFrame(stored_vectors)
print(df.head())  # Show first few rows