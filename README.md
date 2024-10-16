# **PDF Chatbot**

This project is a **PDF Chatbot** built using Streamlit, HuggingFace Transformers, and LangChain. The bot allows users to upload PDF files and ask questions about the content of those files, enabling an interactive chat interface for querying the PDF. The system extracts text from the PDF, creates embeddings for retrieval, and uses a language model to generate responses to user queries.

---

## **Project Setup**

### **Requirements**

* **Python 3.10+**  
* **Docker** (if running via Docker)  
* **Required Python packages**:  
  * `streamlit`  
  * `transformers`  
  * `torch`  
  * `langchain-community`  
  * `sentence-transformers`  
  * `chromadb`  
  * `streamlit_chat`  
  * `gtts`  
  * `tempfile`  
  * `PDFMiner`

### **Installation Guide**

#### **Clone the Repository**

bash  
Copy code  
`git clone https://github.com/your-repo/pdf-chatbot.git`  
`cd pdf-chatbot`

### **Install Python Dependencies**

1. **Create a virtual environment** (optional but recommended):

bash  
Copy code  
`python -m venv venv`  
`` source venv/bin/activate  # On Windows use `venv\Scripts\activate` ``

2. **Install the required packages**:

bash  
Copy code  
`pip install -r requirements.txt`

---

### **Running the Project**

#### **Locally (without Docker)**

1. **Navigate to the project directory**:

bash  
Copy code  
`cd pdf-chatbot`

2. **Run the application**:

bash  
Copy code  
`streamlit run chatbot_app.py`

3. **Access the chatbot interface**:

Open your browser and go to `http://localhost:8501`.

---

### **Running with Docker**

#### **Dockerfile**

The Dockerfile uses **Python 3.10** as the base image and sets the `/app` directory inside the container as the working directory. Dependencies are installed via `requirements.txt`, and the Streamlit application is exposed on port `8501`.

Dockerfile  
Copy code  
`# Use the official Python image`  
`FROM python:3.10`

`# Set the working directory inside the container`  
`WORKDIR /app`

`# Copy the requirements.txt file first to leverage Docker cache`  
`COPY requirements.txt .`

`# Install required Python packages`  
`RUN pip install -r requirements.txt --default-timeout=100 future`

`# Copy the rest of the application files to the container's working directory`  
`COPY . .`

`# Expose the port that Streamlit will run on`  
`EXPOSE 8501`

`# Command to run your Streamlit application`  
`CMD ["streamlit", "run", "chatbot_app.py"]`

#### **Build and Run with Docker**

1. **Build the Docker image**:

bash  
Copy code  
`docker build -t pdf-chatbot .`

2. **Run the Docker container**:

bash  
Copy code  
`docker run -p 8501:8501 pdf-chatbot`

3. **Access the chatbot interface**:

Open your browser and go to `http://localhost:8501`.

---

## **Code Explanation**

### **Model and Device Configuration**

The model used is `"MBZUAI/LaMini-T5-738M"`, which is loaded via HuggingFace's Transformers library. Both the tokenizer and model checkpoints are initialized.

python  
Copy code  
`checkpoint = "MBZUAI/LaMini-T5-738M"`  
`tokenizer = AutoTokenizer.from_pretrained(checkpoint)`  
`base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)`

---

### **PDF Data Ingestion**

To extract content from PDF files, the **PDFMinerLoader** is used. The content is split into smaller chunks using **RecursiveCharacterTextSplitter** to handle large documents more effectively.

python  
Copy code  
`loader = PDFMinerLoader(os.path.join(root, file))`  
`documents = loader.load()`  
`text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)`  
`texts = text_splitter.split_documents(documents)`

---

### **Embeddings & Vector Store**

The chatbot uses **SentenceTransformerEmbeddings** to convert the split PDF text into embeddings, which can then be stored and retrieved using **Chroma**. These embeddings help retrieve relevant sections of the PDF based on user queries.

python  
Copy code  
`embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")`  
`db = Chroma.from_documents(`  
    `texts,`  
    `embeddings,`  
    `client=chroma_client`  
`)`

---

### **RetrievalQA**

For question-answering, **LangChain**'s `RetrievalQA` chain is used. It allows the system to retrieve relevant chunks from the vector store (Chroma) and pass them to the language model for response generation.

python  
Copy code  
`qa = RetrievalQA.from_chain_type(`  
    `llm=local_llm,`  
    `chain_type="stuff",`  
    `retriever=retriever,`  
    `return_source_documents=True`  
`)`

---

### **Text-to-Speech (TTS)**

The chatbot is capable of converting its responses to speech using **Google's Text-to-Speech (gTTS)** library.

python  
Copy code  
`def text_to_speech(text):`  
    `tts = gTTS(text=text, lang='en')`  
    `with tempfile.NamedTemporaryFile(delete=False) as fp:`  
        `tts.save(fp.name)`  
        `return fp.name`

---

### **Conversation History**

The chatbot maintains a session history of the conversation, displaying both the user's input and the chatbot's responses.

python  
Copy code  
`if "generated" not in st.session_state:`  
    `st.session_state["generated"] = ["I am ready to help you"]`  
`if "past" not in st.session_state:`  
    `st.session_state["past"] = ["Hey there!"]`

---

## **ChromaDB Configuration**

The application uses **ChromaDB** to store embeddings persistently. This configuration uses **duckdb+parquet** for persistent storage.

python  
Copy code  
`CHROMA_SETTINGS = Settings(`  
    `chroma_db_impl='duckdb+parquet',`  
    `persist_directory='db',`  
    `anonymized_telemetry=False`  
`)`

---

## **Key Features**

1. **PDF Ingestion**: Upload a PDF and extract its content for interactive querying.  
2. **Question Answering**: Ask context-based questions about the content of the PDF and get detailed responses.  
3. **Embeddings & Retrieval**: Semantic search within the document using embeddings to retrieve relevant chunks.  
4. **Text-to-Speech**: Convert the chatbot's text responses into audio using Google TTS.  
5. **Streamlit Interface**: A user-friendly web interface for interacting with PDF documents.  
6. **Docker Support**: Run the application inside a Docker container for easy deployment and environment isolation.

---

## **Conclusion**

This PDF chatbot is an interactive tool that leverages transformers and embedding-based retrieval to allow users to query PDF documents easily. It provides accurate, context-based responses and can convert those responses into speech, making the interaction more dynamic and accessible.

For further issues, questions, or contributions, feel free to contact the repository owner or submit an issue in the repository.


