

import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="Chat PDF 💬", page_icon="📄")

def get_pdf_text(pdf_docs):
    """
    Extract text from uploaded PDF files
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """
    Split text into manageable chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Create vector embeddings for text chunks
    """
    try:
        # Ensure we have text chunks
        if not text_chunks:
            st.error("No text chunks to process")
            return None

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Create vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save vector store locally
        vector_store.save_local("faiss_index")
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversational_chain():
    """
    Create prompt template and load QA chain
    """
    prompt_template = """
    Use the following context to answer the question as precisely as possible. 
    If the answer is not found in the context, clearly state "Information not available in the PDF".

    Context:\n {context}\n
    Question: \n{question}\n

    Detailed Answer:
    """

    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Load QA chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """
    Process user question and generate response
    """
    try:
        # Check if API key is set
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Google API Key not found. Please set it in .env file.")
            return

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Load vector store
        try:
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except Exception as load_error:
            st.error(f"Error loading vector store: {load_error}")
            st.error("Please upload and process PDFs first.")
            return
        
        # Perform similarity search
        docs = new_db.similarity_search(user_question)
        
        # If no relevant documents found
        if not docs:
            st.warning("No relevant information found in the uploaded PDFs.")
            return

        # Get conversational chain
        chain = get_conversational_chain()
        
        # Generate response
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        # Display response
        st.write("Reply: ", response["output_text"])
    
    except Exception as e:
        st.error(f"Error processing your question: {e}")

def main():
    # Page title
    st.header("Chat with PDF using Gemini 💁")

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Please set GOOGLE_API_KEY in your .env file")
        return

    # Configure Generative AI
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("PDF Processing Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDF Files", 
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if raw_text:
                        # Create text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create vector store
                        vector_store = get_vector_store(text_chunks)
                        
                        if vector_store:
                            st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files first.")

    # Main chat interface
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Process user question if submitted
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()