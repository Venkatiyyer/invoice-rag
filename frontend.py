import streamlit as st
import os
import tempfile
from app import load_and_vectorize, process_invoices, chain


# Set the page title
st.set_page_config(page_title="Invoice JSONifier by Venkat Iyer")
st.title("Invoice JSONifier")
# st.markdown("JSON for every invoice.")


st.markdown("Upload a single PDF or text invoice file to extract its key details into a structured JSON format.")

# File uploader
uploaded_file = st.file_uploader("Choose an invoice file...", type=["pdf", "txt"])

if uploaded_file is not None:
    # Use a temporary directory to save and process the file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the full file path
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        # Write the uploaded file to the temporary directory
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        # Display a processing message while the model runs
        with st.spinner('Processing invoice... This may take a moment.'):
            try:
                # Load and vectorize the document from the temp directory
                vector_store = load_and_vectorize(data_dir=temp_dir)
                
                # Process the invoice using the chain
                json_response = process_invoices(vector_store, chain)

                # Display the result
                st.subheader("Extracted JSON Output")
                st.code(json_response, language="json")

            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                # The temporary directory and its contents are automatically deleted
                # when the 'with' block is exited, fulfilling the deletion requirement.
                pass