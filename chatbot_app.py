import streamlit as st
import os
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from chromadb.config import Settings
from chromadb import Client
from streamlit_chat import message
from gtts import gTTS
import tempfile

# Set page configuration
st.set_page_config(layout="wide", page_title="PDF Chatbot", page_icon="📄")

# Model and device configuration
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

persist_directory = "db"

# Function for data ingestion
@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)

    # Create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create Chroma client using new settings
    chroma_client = Client(Settings(persist_directory=persist_directory))

    # Create vector store, persistence is automatic
    db = Chroma.from_documents(
        texts,
        embeddings,
        client=chroma_client
    )
    db = None

# Function for LLM pipeline
@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# Function for QA retrieval
@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    chroma_client = Client(Settings(persist_directory="db"))
    db = Chroma(persist_directory="db", embedding_function=embeddings, client=chroma_client)
    
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# Function to process the answer
def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

# Function to convert text to speech and play the audio
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        tts.save(fp.name)
        return fp.name

# Function to get file size
def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

# Function to display PDF
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to display conversation history
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

# Main function to handle file uploads and processing
def main():
    # Check if the 'docs' directory exists and create it if it doesn't
    if not os.path.exists("docs"):
        os.makedirs("docs")

    st.markdown("""
        <style>
        /* General Styles */
        body {
            background-color: #F0F8FF; /* Light cyan background */
        }
        .stApp {
            background-color: #F0F8FF;
            font-family: 'Montserrat', sans-serif; /* Custom font */
        }

        /* Custom styling for title and subtitle */
        .title {
            font-size: 45px;
            text-align: center;
            color: #4169E1; /* Royal blue */
            margin-top: 10px;
            font-weight: bold;
        }
        .subtitle {
            font-size: 20px;
            color: #FF4500; /* Orange red */
            text-align: center;
            margin-bottom: 25px;
        }

        /* Styling for file details and chat area */
        .file-details {
            font-size: 18px;
            color: black;
            margin-top: 10px;
            font-weight: 500;
        }

        /* Custom button styling */
        .stButton>button {
            background-color: #4169E1;
            color: white;
            border-radius: 8px;
            font-size: 18px;
            padding: 10px 20px;
            font-family: 'Montserrat', sans-serif;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FF4500;
        }

        /* Chat message styling */
        .message {
            background-color: #f7f7f7;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>Chat with your PDF</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subtitle'>Upload your PDF below to start chatting with it</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose your PDF file", type=["pdf"], label_visibility="hidden")

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("<h4 class='file-details'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 class='file-details'>File preview</h4>", unsafe_allow_html=True)
            displayPDF(filepath)

        with col2:
            with st.spinner('Processing embeddings...'):
                ingested_data = data_ingestion()
            st.success('Embeddings successfully created!')
            st.markdown("<h4 class='file-details'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input", placeholder="Ask a question about the PDF")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]

            # Generate response from the user input
            if user_input:
                answer = process_answer({'query': user_input})
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(answer)

                # Convert the response text to speech and play it
                audio_file = text_to_speech(answer)
                audio_bytes = open(audio_file, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")

            # Display conversation history
            if st.session_state["generated"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()
