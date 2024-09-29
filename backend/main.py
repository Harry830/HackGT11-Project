

import os
import hnswlib
import numpy as np
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
import docx2txt

# Load environment variables from the .env file
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, '../config/.env')
load_dotenv(dotenv_path=env_path)

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API Key not found in environment variables. Please check your .env file.")

# Initialize OpenAI embeddings with the API key
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize HNSWlib index
embedding_dim = 1536
num_elements = 1000  # Placeholder value, update with actual number of documents
M = 64
ef_construction = 400
hnsw_index = hnswlib.Index(space='cosine', dim=embedding_dim)
hnsw_index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
print("HNSW index initialized.")

# List to store document objects
documents = []

# Function to extract text from various file types
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_doc(file_path):
    return docx2txt.process(file_path)

def extract_text_from_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_xlsx(file_path):
    df = pd.read_excel(file_path, dtype=str)
    return "\n".join(df.fillna("").apply(lambda x: " ".join(x), axis=1))

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path, dtype=str)
    return "\n".join(df.fillna("").apply(lambda x: " ".join(x), axis=1))

# Load documents from a folder and extract text
def load_documents_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            content = extract_text_from_txt(file_path)
        elif filename.endswith(".doc"):
            content = extract_text_from_doc(file_path)
        elif filename.endswith(".docx"):
            content = extract_text_from_docx(file_path)
        elif filename.endswith(".pdf"):
            content = extract_text_from_pdf(file_path)
        elif filename.endswith(".xlsx"):
            content = extract_text_from_xlsx(file_path)
        elif filename.endswith(".csv"):
            content = extract_text_from_csv(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        
        if content.strip():
            documents.append(Document(page_content=content))
            print(f"Loaded: {filename}")

# Load documents from the specified folder
folder_path = os.path.join(base_dir, '../data')
load_documents_from_folder(folder_path)

# Embed the loaded documents and add them to the index
if documents:
    print(f"Loaded {len(documents)} documents.")
    texts = [doc.page_content for doc in documents]
    doc_embeddings = embeddings.embed_documents(texts)
    doc_embeddings = np.array(doc_embeddings)
    
    for i, embedding in enumerate(doc_embeddings):
        hnsw_index.add_items([embedding], [i])
    
    hnsw_index.set_ef(100)
    print("Documents added to HNSW index successfully.")
else:
    print("No documents found in the specified folder.")

# Function to search the index with a query
def search_index(query, top_k=3):
    query_embedding = embeddings.embed_documents([query])[0]
    hnsw_index.set_ef(max(300, top_k * 100))  
    labels, distances = hnsw_index.knn_query(np.array([query_embedding]), k=top_k)
    
    results = [{"text": documents[label].page_content, "distance": distance}
               for label, distance in zip(labels[0], distances[0])]
    return results

# Define a function to enhance the assistant's response
def generate_response(query):
    results = search_index(query)
    
    # Construct a conversational response
    if results:
        response_text = f"I found some information for you:\n"
        for result in results:
            response_text += f"- {result['text']} (similarity: {1 - result['distance']:.2f})\n"
        return response_text
    else:
        return "I'm sorry, I couldn't find any relevant information."

# Test the search functionality
if __name__ == '__main__':
    query1 = "What is HackGT?"
    query2 = "What is LangChain?"
    query3 = "Where is Georgia Tech located?"
    query4 = "What is FAISS used for?"

    print("Response to Query 1:", generate_response(query1))
    print("Response to Query 2:", generate_response(query2))
    print("Response to Query 3:", generate_response(query3))
    print("Response to Query 4:", generate_response(query4))
