from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

import os

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

def create_vector_db_from_pdf(pdf_path: str) -> FAISS:
    """Loads a PDF, splits text into chunks, and stores in FAISS vector DB."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split into chunks (adjust chunk_size and overlap as needed)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # Store in FAISS vector database
    db = FAISS.from_documents(split_docs, embeddings)
    return db

def get_response_from_query(db, query, k=3):
    """Search vector DB and get LLM-generated response."""
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model='text-davinci-003')

    prompt = PromptTemplate(
        input_variables=['question', 'docs'],
        template="""
        You are an AI assistant answering questions based on a PDF document.

        Answer the question: {question}
        By searching the following PDF content: {docs}

        Only use factual information from the PDF.

        If you don't have enough information, say "I don't know".

        Your answer should be detailed.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)

    return response.replace("\n", "")

# Example Usage
if __name__ == "__main__":
    pdf_path = "example.pdf"  # Replace with your PDF file path
    db = create_vector_db_from_pdf(pdf_path)
    
    query = "What is the main topic of this document?"
    response = get_response_from_query(db, query)
    
    print("AI Response:", response)
