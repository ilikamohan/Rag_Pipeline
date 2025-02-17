import os
import asyncio
from openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API Keys
COHERE_API_KEY = os.getenv("Cohere_API_Key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Cohere Embeddings
cohere_embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# List of URLs
urls = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5137920/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3374921/",
    "https://pubmed.ncbi.nlm.nih.gov/32886587/",
    "https://www.annualreviews.org/content/journals/10.1146/annurev-psych-062520-122331",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5579396/",
    "https://www.ncbi.nlm.nih.gov/books/NBK513300/#article-29540.s2",
    "https://www.ncbi.nlm.nih.gov/books/NBK513300/#article-29540.s1",
    "https://www.ncbi.nlm.nih.gov/books/NBK513300/#article-29540.s3",
    "https://www.ncbi.nlm.nih.gov/books/NBK513300/#article-29540.s4"
]

# Load Documents Asynchronously
async def load_documents(urls):
    tasks = [asyncio.to_thread(WebBaseLoader(url).load) for url in urls]
    return await asyncio.gather(*tasks)

# Run Async Document Loading
docs_nested = asyncio.run(load_documents(urls))

# Flatten List
docs = [doc for sublist in docs_nested for doc in sublist]

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs)

# Add to Vector Store (ChromaDB)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag",
    embedding=cohere_embeddings
)

# Create Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}  # Retrieve top 4 similar documents
)

# Query the Vector Store
question = "explain to me how stress impacts my physiological health"
docs = retriever.invoke(question)

# Output Retrieved Content
if docs:
    print(f"Title: {docs[0].metadata.get('title', 'Unknown')}\n")
    print(f"Source: {docs[0].metadata.get('source', 'Unknown')}\n")
    print(f"Content: {docs[0].page_content}\n")
else:
    print("No relevant documents found.")
