import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    logging.warning("GEMINI_API_KEY not set in .env file. Ensure it's provided in the app.")

# Function to call Gemini API for text generation
def call_gemini_api(prompt):
    logging.info("Calling Gemini API.")
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")  # Specify the Gemini model
        response = model.generate_content(prompt)
        if response:
            logging.info("Received response from Gemini API.")
            return response.text
        else:
            logging.warning("No response received from Gemini API.")
            return "I don't know."
    except Exception as e:
        logging.error(f"Error in calling Gemini API: {e}")
        return "Error while generating content."

# PDF text extraction function
def get_pdf_text(pdf_docs):
    logging.info("Extracting text from uploaded PDF files.")
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        logging.info("Text extraction from PDFs completed.")
    except Exception as e:
        logging.error(f"Error extracting text from PDFs: {e}")
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    logging.info("Splitting text into manageable chunks.")
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        logging.info(f"Text split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
        return []

# Initialize instructor embeddings using Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

# Create vector database from PDF text
def create_vector_db():
    logging.info("Creating vector database from PDF text.")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
    if not pdf_docs:
        logging.warning("No PDF files uploaded.")
        st.error("Please upload PDF files to process.")
        return

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)

    # Create FAISS vector store from text chunks
    vectordb = FAISS.from_texts(text_chunks, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)
    logging.info("Vector database created and saved successfully.")

# Retrieve top-k most relevant chunks
def get_relevant_context(user_question, top_k=5):
    logging.info(f"Retrieving top-{top_k} relevant chunks for the query.")
    try:
        vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
        docs = vectordb.similarity_search(user_question, k=top_k)
        logging.info(f"Retrieved {len(docs)} relevant chunks.")
        return docs
    except Exception as e:
        logging.error(f"Error retrieving relevant context: {e}")
        return []

# Format context from retrieved chunks
def format_context(docs):
    logging.info("Formatting context from retrieved chunks.")
    try:
        context = " ".join([doc.page_content for doc in docs])
        logging.info("Context formatted successfully.")
        return context
    except Exception as e:
        logging.error(f"Error formatting context: {e}")
        return ""

# Define the prompt template
def get_conversational_chain():
    logging.info("Setting up conversational chain.")
    prompt_template = """
    You are an intelligent assistant. Read the context carefully, and answer as if you're explaining to a person in a simple and friendly manner.
    Use plain language, avoid technical jargon, and provide a concise response. Only structure your answer in points if there are multiple distinct points or steps to explain.

    <context>
    {context}
    </context>
    Question: {question}

    Your answer should be:
    - Friendly and conversational
    - In complete sentences if the answer is short or straightforward
    - Structured as a list only if there are multiple points or steps
    - Accurate and based only on the context provided
    """

    def qa_chain(query):
        docs = get_relevant_context(query, top_k=5)
        context = format_context(docs)
        prompt = prompt_template.format(context=context, question=query)
        return call_gemini_api(prompt)

    return qa_chain

# Handle user input and generate response
def user_input(user_question):
    logging.info(f"Processing user input: {user_question}")
    chain = get_conversational_chain()
    response = chain(user_question)
    st.write(response)

# Main function to run the Streamlit app
def main():
    logging.info("Starting Streamlit app.")
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    # Sidebar with API key and PDF processing options
    with st.sidebar:
        st.title("Menu:")

        # Step 1: Input API Key
        if 'gemini_api_key' not in st.session_state:
            api_key = st.text_input("Enter your Gemini API key:", type="password")
            if st.button("Submit API Key"):
                if api_key:
                    st.session_state['gemini_api_key'] = api_key
                    logging.info("API key stored successfully.")
                    st.success("API key saved successfully!")
                else:
                    logging.error("Invalid API key.")
                    st.error("Please enter a valid API key.")
        else:
            st.success("API key already set!")

        # Step 2: Display PDF Upload and Process Button
        if 'gemini_api_key' in st.session_state:
            st.subheader("Upload PDFs and Process")
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
            if st.button("Submit & Process PDFs"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        create_vector_db()
                        st.success("PDFs processed successfully!")
                else:
                    st.error("Please upload at least one PDF.")

if __name__ == "__main__":
    main()
