import faiss
import numpy as np
import openai
import os
import time
import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from openai import OpenAIError

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment

file_path = "/Users/ernestojr.almeda/Repos/grange/OMC-Agenda-5-February-2025.pdf"

client = openai.OpenAI()  # ✅ Create OpenAI client

# Hashing function to detect duplicates
def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Load and process documents
def load_documents(file_paths):
    docs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File {file_path} not found. Skipping...")
            continue

        print(f"📂 Loading {file_path}...")
        loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
        loaded_docs = loader.load()
        docs.extend(loaded_docs)

    if not docs:
        raise ValueError("❌ No documents were loaded. Check file paths.")
    return docs

# Split documents into chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    print(f"📄 Number of document chunks: {len(split_docs)}")
    
    seen_hashes, unique_docs = set(), []
    for doc in split_docs:
        doc_hash = hash_text(doc.page_content)
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc.page_content)  # Store text only
    
    with open("texts.json", "w") as f:
        json.dump(unique_docs, f)
    
    print(f"✅ Unique document chunks: {len(unique_docs)}")
    return unique_docs

# Generate Embeddings
def generate_embeddings(texts, model="text-embedding-ada-002", batch_size=10):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            client = openai.OpenAI()
            response = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [item.embedding for item in response.data]

            embeddings.extend(batch_embeddings)
            time.sleep(1)
        except OpenAIError as e:
            print(f"⚠️ OpenAI API Error: {e}")
            time.sleep(10)  # Backoff strategy
    return embeddings

# Create FAISS Vector DB
def create_vector_db(docs, db_path="faiss_index"):
    if not docs:
        raise ValueError("❌ No documents to embed.")

    embeddings = generate_embeddings(docs)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, db_path)
    print(f"✅ FAISS database created with {len(embeddings)} vectors.")
    return index

# Load FAISS DB
def load_vector_db(db_path="faiss_index"):
    if not os.path.exists(db_path):
        raise ValueError("❌ FAISS index file not found. Run indexing first.")
    return faiss.read_index(db_path)

# Query Processing
def retrieve_and_generate(query, vector_db, k=5):
    """Retrieve relevant documents and generate a response."""
    
    # ✅ Generate query embedding
    query_embedding = generate_embeddings([query])[0]  
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # ✅ Perform FAISS search
    _, indices = vector_db.search(query_embedding, k)

    # ✅ Load stored texts from JSON
    if not os.path.exists("texts.json"):
        print("❌ Error: texts.json not found. Ensure you process documents first.")
        return "Error: No stored document chunks found."

    with open("texts.json", "r") as f:
        try:
            texts = json.load(f)
        except json.JSONDecodeError:
            print("❌ Error: texts.json is corrupted.")
            return "Error: Stored document data is invalid."

    # ✅ Extract retrieved text safely
    retrieved_texts = [texts[i] for i in indices[0] if i < len(texts)]
    if not retrieved_texts:
        return "No relevant documents found."

    context = " ".join(retrieved_texts)

    # ✅ Generate response using GPT-4
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides informative responses."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content  # ✅ Corrected dot notation
    except openai.OpenAIError as e:
        print(f"❌ OpenAI API error: {e}")
        return "Sorry, I couldn't generate a response."

# Main Execution
if __name__ == "__main__":
    file_paths = [file_path]
    documents = load_documents(file_paths)
    split_docs = split_documents(documents)
    vector_db = create_vector_db(split_docs) if not os.path.exists("faiss_index") else load_vector_db()
    
    user_query = "What is the agenda about?"
    response = retrieve_and_generate(user_query, vector_db)
    print("\n🤖 AI Response:", response)