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

# Error handling for environment variables
def load_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set the GOOGLE_API_KEY environment variable")
        st.stop()
    return api_key

def get_pdf_text(pdf_docs):
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "Answer is not available in the context", 
    and do not provide a wrong answer.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        # Load embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load existing vector store with safe deserialization
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Perform similarity search
        docs = new_db.similarity_search(user_question)
        
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
    # Set up the Streamlit page
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.header("Chat with PDF using Gemini 💁")

    # Load API Key
    api_key = load_api_key()
    
    # Configure Generative AI
    genai.configure(api_key=api_key)

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if raw_text:
                        # Create text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create vector store
                        vector_store = get_vector_store(text_chunks)
                        
                        if vector_store:
                            st.success("PDF processed successfully!")
            else:
                st.warning("Please upload PDF files first.")

    # Main chat interface
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Process user question if submitted
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()