import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA

# Load API keys from .env file
load_dotenv()

# Setup Azure OpenAI credentials
embedding_model = AzureOpenAIEmbeddings(
    api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Streamlit UI
st.set_page_config(page_title="PDF Question Generator")
st.title("📄 PDF Question Answering App")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    if not text.strip():
        st.warning("No text found in the PDF.")
    else:
        # Split text into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embedding=embedding_model)
        retriever = vectorstore.as_retriever()

        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("PDF processed! You can now ask questions below 👇")

        # Ask question
        question = st.text_input("Ask a question (or say 'generate 3 MCQs'):")

        if question:
            with st.spinner("Processing..."):
                query = question.lower()
                if "mcq" in query:
                    prompt = "Generate 3 multiple choice questions with 4 options each. Mark the correct answer."
                elif "short answer" in query:
                    prompt = "Generate 3 short answer questions based on this document."
                elif "summarize" in query:
                    prompt = "Give a short summary of this document."
                else:
                    prompt = question

                result = qa.invoke({"query": prompt})
                st.markdown(result["result"])
