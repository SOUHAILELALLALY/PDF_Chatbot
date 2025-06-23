import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]




def main():
    st.set_page_config(page_title="💬 Chat with PDF | Gemini + LangChain", layout="wide")

    # Custom title with emoji
    st.markdown("""
        <style>
            .main-title {
                font-size: 38px;
                font-weight: 700;
                color: #2c3e50;
                text-align: center;
                padding-bottom: 10px;
            }
            .subheader {
                text-align: center;
                font-size: 18px;
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            .footer {
                text-align: center;
                font-size: 14px;
                color: #bdc3c7;
                margin-top: 30px;
            }
        </style>
        <div class="main-title">💬 Chat with Your PDF</div>
        <div class="subheader">Powered by Google Gemini, LangChain, and FAISS</div>
    """, unsafe_allow_html=True)

    # Layout: input on left, result on right
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Ask Your Question")
        user_question = st.text_input("What would you like to know?", placeholder="Type your question here...")

        if user_question:
            with st.spinner("Searching and thinking... 🤔"):
                st.write("🔍 Your question:", user_question)
                response = user_input(user_question)
                with st.expander("📘 See Answer", expanded=True):
                    st.success(response)

    with st.sidebar:
        st.header("📁 Upload PDF Files")
        pdf_docs = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True)

        if st.button("🧠 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs... ⏳"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

    st.markdown("<div class='footer'>Made with using Streamlit, LangChain, and Gemini</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()