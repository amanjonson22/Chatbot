import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings, ChatWatsonx
from langchain_community.vectorstores import FAISS
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain_classic.chains.question_answering import load_qa_chain
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
URL = os.getenv("URL")
PROJECT_ID = os.getenv("PROJECT_ID")

#Upload PDF files
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")


#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

#Break into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, 
        chunk_overlap=150, 
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    #st.write(chunk)

    embeddings = WatsonxEmbeddings(
        model_id="ibm/slate-30m-english-rtrvr",
        url=URL,
        apikey=API_KEY,
        project_id=PROJECT_ID
    )

#creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

#get user question
    user_question = st.text_input("Type your question here")

# do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        parameter = TextChatParameters(
        temperature=0.1,
        max_tokens=1000
    )

        llm = ChatWatsonx(
            model_id= "meta-llama/llama-3-3-70b-instruct",
            url=URL,
            project_id=PROJECT_ID,
            api_key=API_KEY,
            params= parameter
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        input_data = {
            'input_documents': match,
            'question': user_question
        }

        response = chain.run(input_data)
        st.write(response)




