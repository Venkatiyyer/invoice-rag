import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Manually set the API keys
# google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set the environment variable for Google API key
# os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize the model
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
# Setup the LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=groq_api_key
)

# Define the prompt template for generating responses
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful and friendly assistant.

    For questions related to the provided context, answer strictly based on the given information.
    For general or social questions (e.g., greetings or small talk), respond in a human, conversational tone.

    Use natural and easy-to-understand language in all your answers.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# def vector_embedding(data_dir="./data"):
#     try:
#         # Load .txt files from the data directory
#         txt_file_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith(".txt")]
        
#         # Initialize TextLoader for each text file
#         txt_loader = [TextLoader(file_path) for file_path in txt_file_paths]

#         # Load .pdf files from the data directory
#         pdf_loader = PyPDFDirectoryLoader(data_dir)

#         # Load all documents
#         txt_docs = []
#         for loader in txt_loader:
#             txt_docs += loader.load()

#         pdf_docs = pdf_loader.load()  # Load PDF files

#         # Combine both text and PDF documents into one list
#         all_docs = txt_docs + pdf_docs

#         # If no documents are found
#         if not all_docs:
#             return {"message": "No documents found in the specified directory."}

def vector_embedding(data_dir="./data"):
    try:
        files = os.listdir(data_dir)
        if not files:
            return {"message": "📂 data_dir is empty"}

        # pick up .txt and .pdf
        txts = [f for f in files if f.endswith(".txt")]
        pdfs = [f for f in files if f.lower().endswith(".pdf")]
        if not (txts or pdfs):
            return {"message": "❌ No .txt or .pdf files found"}

        # load text
        txt_docs = []
        for fn in txts:
            loader = TextLoader(os.path.join(data_dir, fn))
            txt_docs += loader.load()

        # load pdf
        try:
            pdf_docs = PyPDFDirectoryLoader(data_dir).load()
        except Exception as e:
            return {"message": f"PDF loading error: {e}"}

        all_docs = txt_docs + pdf_docs
        if not all_docs:
            return {"message": "📄 Files found, but no content could be loaded"}


        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}         # <— add this
    )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        final_documents = text_splitter.split_documents(all_docs[:20])  # splitting
        vectors = FAISS.from_documents(final_documents, embeddings)  # vector OpenAI embeddings

        return vectors

    except Exception as e:
        return {"message": f"Error in vector embedding: {str(e)}"}

def query_documents(query: str, vectors, prompt=prompt):
    try:
        # build the chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # invoke and time it
        start = time.process_time()
        response = retrieval_chain.invoke({'input': query})
        elapsed = time.process_time() - start

        # if the chain gave back an answer, return it
        if 'answer' in response and response['answer'].strip():
            return {
                "query": query,
                "answer": response['answer'],
                "relevant_docs": response.get('context', ""),
                "response_time": elapsed
            }
        # otherwise it ran but found nothing
        return {"no_result": True}

    except Exception as e:
        # bubble up the real error
        return {"error": str(e)}