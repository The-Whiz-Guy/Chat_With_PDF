import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Function to call Gemini API for text generation
def call_gemini_api(prompt):
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")  # Specify the Gemini model
        response = model.generate_content(prompt)
        return response.text if response else "I don't know."
    except Exception as e:
        return "Error while generating content."

# PDF text extraction function
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Initialize instructor embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

# Create vector database from PDF text
def create_vector_db(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectordb = FAISS.from_texts(text_chunks, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

# Retrieve top-k most relevant chunks
def get_relevant_context(user_question, top_k=5):
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    return vectordb.similarity_search(user_question, k=top_k)

# Format context from retrieved chunks
def format_context(docs):
    return " ".join([doc.page_content for doc in docs])

# Define the prompt template
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Read the context carefully, and answer as if you're explaining to a person in a simple and friendly manner.
    Use plain language, avoid technical jargon, and provide a concise response.

    <context>
    {context}
    </context>
    Question: {question}
    """

    def qa_chain(query):
        docs = get_relevant_context(query, top_k=5)
        context = format_context(docs)
        prompt = prompt_template.format(context=context, question=query)
        return call_gemini_api(prompt)

    return qa_chain

# Handle user input and generate response
def user_input(user_question):
    chain = get_conversational_chain()
    response = chain(user_question)
    st.write(response)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question_input")

    if user_question:
        user_input(user_question)

    # Sidebar with API key and PDF processing options
    with st.sidebar:
        st.title("Menu:")

        # Step 1: Input API Key
        if 'gemini_api_key' not in st.session_state:
            api_key = st.text_input("Enter your Gemini API key:", type="password", key="api_key_input")
            if st.button("Submit API Key", key="submit_api_key"):
                if api_key:
                    st.session_state['gemini_api_key'] = api_key
                    st.success("API key saved successfully!")
                else:
                    st.error("Please enter a valid API key.")
        else:
            st.success("API key already set!")

        # Step 2: Display PDF Upload and Process Button
        if 'gemini_api_key' in st.session_state:
            st.subheader("Upload PDFs and Process")
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf", key="pdf_uploader")
            if st.button("Submit & Process PDFs", key="process_pdfs"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        create_vector_db(pdf_docs)
                        st.success("PDFs processed successfully!")
                else:
                    st.error("Please upload at least one PDF.")

if __name__ == "__main__":
    main()
