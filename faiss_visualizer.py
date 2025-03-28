import faiss
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load or create FAISS index
dim = 128  # Dimension of vectors
index_file = "faiss_index.bin"

try:
    index = faiss.read_index(index_file)
    st.success(f"Loaded FAISS index with {index.ntotal} vectors")
except:
    st.warning("FAISS index not found. Creating a new one.")
    index = faiss.IndexFlatL2(dim)
    db_vectors = np.random.rand(100, dim).astype('float32')
    index.add(db_vectors)
    faiss.write_index(index, index_file)

# UI for viewing the FAISS database
st.title("FAISS Database Viewer")

# Show number of vectors
st.write(f"Number of vectors stored: {index.ntotal}")

# Search functionality
st.subheader("Search Nearest Vectors")
query_vector = np.random.rand(1, dim).astype('float32')
k = st.slider("Number of nearest neighbors (k)", 1, 10, 5)
distances, indices = index.search(query_vector, k)

st.write("Nearest Vectors' Indices:", indices)
st.write("Distances:", distances)

# Visualizing Vectors
st.subheader("Vector Distribution")
sample_vectors = np.array([index.reconstruct(i) for i in range(min(100, index.ntotal))])

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(sample_vectors, cmap="coolwarm", cbar=True, ax=ax)
st.pyplot(fig)
