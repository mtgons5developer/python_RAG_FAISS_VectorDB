![image](https://github.com/user-attachments/assets/ae8c9a9c-c702-4633-9fb6-7e1bf8b02e49)

 Strengths of the Implementation

Efficient Retrieval: Uses FAISS for fast nearest neighbor search.
Duplicate Removal: Hashing ensures unique document chunks are stored.
Error Handling: Implements backoff strategy for API errors.
Scalable: The modular approach allows easy expansion to process multiple documents.

🚀 How It Works

1️⃣ Load Documents
Upload PDF or text files.
Extract text using LangChain document loaders.
2️⃣ Process & Store
Split text into manageable chunks for better retrieval.
Convert chunks into vector embeddings using OpenAI's embedding model.
Store embeddings in FAISS for efficient similarity search.
3️⃣ Query & Retrieve
Accept user queries.
Embed the query and search FAISS for the most relevant document chunks.
Send the retrieved context to GPT-4 for intelligent response generation.
💻 Installation & Setup

🔹 1. Clone the Repository
git clone https://github.com/your-username/your-repo.git
cd your-repo
🔹 2. Install Dependencies
pip install faiss-cpu openai numpy langchain langchain_community
🔹 3. Set Up OpenAI API Key
Ensure your OpenAI API key is set in your environment:

export OPENAI_API_KEY="your-openai-api-key"
🔹 4. Run the Script
python main.py
