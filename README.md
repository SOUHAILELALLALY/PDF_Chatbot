# PDF_Chatbot

A Streamlit app that lets you upload PDF documents and interact with them through natural language questions.
Powered by Google Gemini (via LangChain) for understanding your questions and generating answers, with FAISS for efficient similarity search.
🚀 Features

✅ Upload one or more PDF files.

✅ Automatically extracts and chunks the text.

✅ Creates embeddings using Google Generative AI Embeddings and stores them in a local FAISS vector database.

✅ Ask questions in natural language — the app finds the most relevant chunks and uses Gemini to answer, citing context.

✅ Easy-to-use Streamlit interface.

🛠 Tech Stack

    Streamlit – web app framework

    LangChain – orchestration of LLM, embeddings, and vector DB

    Google Generative AI (Gemini) – for embeddings and chat

    FAISS – local vector store

    PyPDF2 – for PDF text extraction
