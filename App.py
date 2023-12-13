import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter #The text_splitter library is used to split a string into tokens
from langchain.document_loaders import PyPDFLoader, DirectoryLoader #and the document-->loaders library is used to load documents from different sources.
from langchain.chains.summarize import load_summarize_chain #used for summarizing 
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

#model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint) #new instances class --> tokenizer 
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map="auto", offload_folder="offload", torch_dtype=torch.float32)


#file loader and preprocessing
#function --> argument --> return text of all pages within the file 
def file_preprocessing(file):
    loader =  PyPDFLoader(file) #pdf filer into memory
    pages = loader.load_and_split() #spilts-->each pages into smaller chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content  #concatenates --> all line together --> one long string 
    return final_texts

#LLM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 1000, #page layout 
        min_length = 50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text) #input text-->spilts it --> into individual words --> and then Sums it 
    result = result[0]['summary_text']
    return result

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

#streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("Automated Document Summarization")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)



if __name__ == "__main__":
    main()