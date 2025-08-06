from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain


from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,FewShotPromptTemplate
from langchain.chains import create_retrieval_chain,LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
import re
from langchain.schema import Document
from typing import List


# Load environment variables
load_dotenv()

# Manually set the API keys
# google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


# Initialize the model
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
# Setup the LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=groq_api_key
)


#  -------------------------------------------------------
# 1. Define a base example prompt template (for each sample):
example_prompt = PromptTemplate(
    input_variables=['invoice_text', 'invoice_json'],
    template='''
INVOICE SAMPLE:
{invoice_text}

Expected Output:
{invoice_json}
'''
)

# 2. Prepare example data (corrected samples):
examples = [
    {
        'invoice_text': '''
Supplier: ABC Pvt Ltd
Invoice No.: INV-1001
Invoice Date: 2025-01-10
Items:
* Service X
  Quantity: 2
  Unit Price: $150.00
  Line Total: $300.00
* Service Y
  Quantity: 1
  Unit Price: $200.00
  Line Total: $200.00
Subtotal: $500.00
Total Due: $500.00
''',
        'invoice_json': '''
{{
  "invoice_no": "INV-1001",
  "date": "2025-01-10",
  "vendor": "ABC Pvt Ltd",
  "items": [
    {{"description": "Service X", "amount": 300.00}},
    {{"description": "Service Y", "amount": 200.00}}
  ],
  "total": 500.00
}}
'''
    },
    {
        'invoice_text': '''
Supplier: XYZ Consulting LLC
Invoice No.: INV-2002
Invoice Date: 2025-02-20
Items:
* Consulting Session
  Quantity: 3
  Unit Price: $100.00
  Line Total: $300.00
Subtotal: $300.00
Sales Tax (10%): $30.00
Total Due: $330.00
''',
        'invoice_json': '''
{{
  "invoice_no": "INV-2002",
  "date": "2025-02-20",
  "vendor": "XYZ Consulting LLC",
  "items": [
    {{"description": "Consulting Session", "amount": 300.00}}
  ],
  "total": 300.00
}}
'''
    },
    {
        'invoice_text': '''
Supplier: Global Tech Services
Invoice No.: INV-3003
Invoice Date: 2025-03-30
Items:
* Software License
  Quantity: 5
  Unit Price: $50.00
  Line Total: $250.00
* Support Fee
  Quantity: 1
  Unit Price: $100.00
  Line Total: $100.00
Subtotal: $350.00
Discount: $50.00
VAT (19%): $57.00
Total Due: $300.00
''',
        'invoice_json': '''
{{
  "invoice_no": "INV-3003",
  "date": "2025-03-30",
  "vendor": "Global Tech Services",
  "items": [
    {{"description": "Software License", "amount": 250.00}},
    {{"description": "Support Fee", "amount": 100.00}}
  ],
  "total": 350.00
}}]
'''
    },
    {
        'invoice_text': '''
Invoice: WMACCESS Internet
Invoice No: 123100401
Customer No: 12345
Invoice Period: 01.02.2024–29.02.2024
Date: 1. März 2024
VAT No.: DE199378386

Billed To:
Musterkunde AG
Mr. John Doe
Musterstr. 23
12345 Musterstadt

Billed By:
CPB Software (Germany) GmbH
Im Bruch 3-63897 Miltenberg/Main
Contact: Stefanie Müller, +49 9371 9786-0
Email: germany@cpb-software.com
Website: www.cpb-software.com

Invoice Details:
* Basic Fee wrnView: 130,00 € (Quantity: 1)
* Basis fee for additional user accounts: 10,00 € (Quantity: 0)
* Basic Fee wmPos: 50,00 € (Quantity: 0)
* Basic Fee wmGuide: 1 000,00 € (Quantity: 0)
* Change of user accounts: 10,00 € (Quantity: 0)

* Transaction Fee T1: 0,58 € (Quantity: 14) – Line Total: 8,12 €
* Transaction Fee T2: 0,70 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee T3: 1,50 € (Quantity: 162) – Line Total: 243,00 €
* Transaction Fee T4: 0,50 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee T5: 0,80 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee T6: 1,80 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee G1: 0,30 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee G2: 0,30 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee G3: 0,40 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee G4: 0,40 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee G5: 0,30 € (Quantity: 0)  – Line Total: 0,00 €
* Transaction Fee G6: 0,30 € (Quantity: 0)  – Line Total: 0,00 €

Total (without VAT): 381,12 €
VAT (19 %): 72,41 €
Gross Amount (incl. VAT): 453,53 €

Payment Information:
Terms: Immediate payment without discount
Bank charges paid by recipient
IBAN: DE29 1234 5678 9012 3456 78
BIC: GENODE51MIC
''',
        'invoice_json': '''
{{
  "invoice_no": "123100401",
  "date": "2024-03-01",
  "vendor": "CPB Software (Germany) GmbH",
  "items": [
    {{"description": "Basic Fee wrnView", "amount": 130.00}},
    {{"description": "Basis fee for additional user accounts", "amount": 0.00}},
    {{"description": "Basic Fee wmPos", "amount": 0.00}},
    {{"description": "Basic Fee wmGuide", "amount": 0.00}},
    {{"description": "Change of user accounts", "amount": 0.00}},
    {{"description": "Transaction Fee T1", "amount": 8.12}},
    {{"description": "Transaction Fee T2", "amount": 0.00}},
    {{"description": "Transaction Fee T3", "amount": 243.00}},
    {{"description": "Transaction Fee T4", "amount": 0.00}},
    {{"description": "Transaction Fee T5", "amount": 0.00}},
    {{"description": "Transaction Fee T6", "amount": 0.00}},
    {{"description": "Transaction Fee G1", "amount": 0.00}},
    {{"description": "Transaction Fee G2", "amount": 0.00}},
    {{"description": "Transaction Fee G3", "amount": 0.00}},
    {{"description": "Transaction Fee G4", "amount": 0.00}},
    {{"description": "Transaction Fee G5", "amount": 0.00}},
    {{"description": "Transaction Fee G6", "amount": 0.00}}
  ],
  "total": 381.12
}}
'''
    }]

# _______________________________________________

# 3. Create a FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Your task is to extract structured JSON data from invoice text. **You must strictly follow the format and fields provided in the examples.** Do not add any extra information, explanations, or text outside of the JSON array.",
    suffix="INVOICE SAMPLE:\n{invoice_text}\n\nExpected Output:",
    input_variables=['invoice_text']
)

# Use new RunnableSequence API
chain = few_shot_prompt | llm


def text_reader(file_path: str) -> Document:
    """
    Reads a single text file robustly and returns a LangChain Document:
    - Opens file in binary mode
    - Decodes as UTF-8 (ignoring errors)
    - Removes non-printable characters
    - Collapses whitespace
    """
    raw = open(file_path, "rb").read()
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return Document(page_content=text, metadata={"source": os.path.basename(file_path)})


def read_txts(data_dir: str) -> List[Document]:
    """
    Reads all .txt files in the given directory using text_reader and returns a list of Documents.
    """
    docs=[]
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".txt"):
            path = os.path.join(data_dir, fn)
            docs.append(text_reader(path))
    return docs

# 4. Vectorize
def load_and_vectorize(data_dir: str = "./data") -> FAISS:
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    # docs = []
    # # for txt in txt_files:
    # #     docs += TextLoader(os.path.join(data_dir, txt)).load()
    # for txt in txt_files:
    #     docs += TextLoader(os.path.join(data_dir, txt)).load()
    
    # docs+=read_txts(data_dir)
    # docs += PyPDFDirectoryLoader(data_dir).load()
    
    docs = []

    # Read all cleaned TXT files
    docs += read_txts(data_dir)  
    
    # Load all PDFs
    if pdf_files:
        docs += PyPDFDirectoryLoader(data_dir).load()
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return FAISS.from_documents(chunks, embed_model)




def process_invoices(vector_store, chain):
    """
    - Retrieves all documents from the FAISS vector_store
    - Runs the provided LLMChain on their combined text
    Returns:
      * retrieved_docs : list of Document objects
      * response       : LLMChain output string
    """
    # Retrieve all docs from FAISS
    total = vector_store.index.ntotal
    retriever = vector_store.as_retriever(search_kwargs={"k": total})
    retrieved_docs =retriever.get_relevant_documents("")
    print(f"Retrieved via FAISS retriever: {len(retrieved_docs)} docs")

    # Combine page contents and run the chain   
    
    full_text = "\n".join(doc.page_content for doc in retrieved_docs)
    response = chain.invoke({"invoice_text": full_text}).content

    # return retrieved_docs, response
    return  response


# a =load_and_vectorize("data")
# # print(a)
# b = process_invoices(a,chain) 
# print(b)# for txt in txt_files:
#     #     docs += TextLoader(os.path.join(data_dir, txt)).load()

        