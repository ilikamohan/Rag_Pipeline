# AI Powered Scientific Knowledge Retrieval System
This POC is of a RAG pipeline which is the foundation for AI Agents that can retrieve and process information intelligently. 

This system answers questions about health by backing them with relevant and recent peer reviewed research. 

Output: 
![Screenshot 2025-02-18 120825](https://github.com/user-attachments/assets/7c62b9dd-cdc6-4d59-a0f5-4f801bbfd8b1)

# Value Proposition

 1. Delivers Scientifically Verified Answers
 2. Simplifies Complex Scientific Concepts
 3. Hyper-Relevant Search Powered by AI
 4. Customizable for Different Industries

## RAG Pipeline Vs. ChatGPT? 
ChatGPT has limitations when it comes to hallucination, knowledge cutoff (training data can be outdated), can't provide sources, lacks training in domain specific knowledge of niche subjects, can be expensive with  GPT-4. 

RAG pipeline powered by GPT-3.5 Turbo overcomes those limitations by: 
 1. Using Real, Verified Sources (Scientific Journals) that are recent 
 2.  Providing Documented Proof (grounding)
 3. Augmenting knowledge by providing a domain specific database, instead of the heavy lifting of fine-tuning. 
 4. Cost-effective AI usage (GPT 3.5 Turbo) 

## Tech Stack Used

 1. Python
 2. LangChain
 3. GPT-3.5
 4. Cohere Embeddings
 5. ChromaDB
