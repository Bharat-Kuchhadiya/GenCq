import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

EMBEDDING_API_KEY = os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME")
EMBEDDING_VERSION = os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION")

CHAT_API_KEY = os.getenv("CHAT_AZURE_OPENAI_API_KEY")
CHAT_ENDPOINT = os.getenv("CHAT_AZURE_OPENAI_ENDPOINT")
CHAT_DEPLOYMENT = os.getenv("CHAT_AZURE_OPENAI_DEPLOYMENT_NAME")
CHAT_VERSION = os.getenv("CHAT_AZURE_OPENAI_API_VERSION")

st.set_page_config(page_title="AI PDF Assistant", layout="wide")
st.title("📄 AI PDF Assistant")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = AzureOpenAIEmbeddings(
        api_key=EMBEDDING_API_KEY,
        azure_endpoint=EMBEDDING_ENDPOINT,
        azure_deployment=EMBEDDING_DEPLOYMENT,
        openai_api_version=EMBEDDING_VERSION,
        chunk_size=1000
    )

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = AzureChatOpenAI(
        api_key=CHAT_API_KEY,
        azure_endpoint=CHAT_ENDPOINT,
        deployment_name=CHAT_DEPLOYMENT,
        api_version=CHAT_VERSION
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧠 Summarize PDF"):
            prompt = "Summarize the key points from the content."
            result = qa.invoke(prompt)
            st.subheader("📌 Summary")
            st.write(result["result"])

    with col2:
        if st.button("❓ Generate MCQs"):
            prompt = (
                "Generate 5 multiple choice questions based on the content. "
                "Each question should have 4 options and clearly marked correct answer (e.g., 'Correct Answer: B')."
            )
            result = qa.invoke(prompt)
            st.subheader("📘 MCQs")
            st.write(result["result"])

    with col3:
        if st.button("✏️ Generate Short Answer Questions"):
            prompt = (
                "Generate 5 short answer questions that require brief, specific responses from the content."
            )
            result = qa.invoke(prompt)
            st.subheader("📝 Short Answer Questions")
            st.write(result["result"])

    st.markdown("---")
    user_question = st.text_input("Ask a custom question about the document:")
    if user_question:
        result = qa.invoke(user_question)
        st.subheader("💬 Answer")
        st.write(result["result"])
