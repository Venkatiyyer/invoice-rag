Here's a `README.md` file for your project, formatted in Markdown.

# Invoice JSONifier 📄➡️💡

**Invoice JSONifier** is an AI-powered tool that effortlessly extracts key data from invoices and converts it into a structured JSON format. Simply upload a PDF or text file, and this application, built with Groq, LangChain, and Streamlit, will handle the rest.

> *JSON for every invoice.*

-----

### Key Features 📋

  * **Fast & Accurate:** Leverages the power of Groq's high-speed inference engine for near-instantaneous data extraction.
  * **Flexible Input:** Supports both PDF and text-based invoice files.
  * **Structured Output:** Converts unstructured invoice data into a clean, predictable JSON format.
  * **Intuitive UI:** A user-friendly interface built with Streamlit makes it easy to upload and process files.
  * **Few-Shot Learning:** Uses a few-shot prompting approach to provide the LLM with clear examples of desired output, ensuring high accuracy.

-----

### How It Works 🧠

The application uses a **Large Language Model (LLM)** from Groq, specifically `llama-3.3-70b-versatile`, to perform the data extraction. The process is as follows:

1.  **File Upload:** A user uploads a PDF or `.txt` invoice file via the Streamlit interface.
2.  **Document Processing:** The file is temporarily saved, read, and preprocessed. LangChain's `PyPDFDirectoryLoader` and `TextLoader` handle the document loading.
3.  **Vectorization & Retrieval:** The invoice text is split into chunks and embedded using a HuggingFace embedding model (`all-MiniLM-L6-v2`). These embeddings are stored in a FAISS vector store.
4.  **Few-Shot Prompting:** The core of the extraction is a few-shot prompt. The LLM is provided with several examples of invoice text and their corresponding JSON outputs. This teaches the model the desired structure and fields for the final JSON.
5.  **Data Extraction:** The LLM receives the new invoice text along with the examples and generates a JSON string as output.
6.  **Display Results:** The extracted JSON data is then displayed on the web page for the user to view and copy.

-----

### Technologies Used 🛠️

  * **Groq:** For lightning-fast LLM inference.
  * **LangChain:** To orchestrate the LLM chain, prompt engineering, and document processing.
  * **Streamlit:** For creating a beautiful and interactive web application.
  * **FAISS:** For efficient similarity search and document retrieval.
  * **HuggingFace Embeddings:** To generate document embeddings.

-----

### Getting Started 🚀

1.  **Clone the repository:**

    ```sh
    git clone [repository_url]
    cd [repository_name]
    ```

2.  **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the project root and add your Groq API key:

    ```ini
    GROQ_API_KEY="your_groq_api_key_here"
    ```

4.  **Run the application:**

    ```sh
    streamlit run app.py
    ```

5.  **Open your browser:**
    The application will be running at `http://localhost:8501`.

-----

### Directory Structure 📂

```
.
├── app.py          # backend file
├── frontend.py     # Streamlit logic
├── .env            # Environment variables
├── data/           # Directory to place your test invoices
├── requirements.txt
└── README.md
```

-----

### License 📜

This project is licensed under the MIT License.
