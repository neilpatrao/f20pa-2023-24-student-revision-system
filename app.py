import os
import pickle

import streamlit as st

from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_openai import OpenAI
from langchain_community.vectorstores import chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    st.header("Test")

    docs = st.file_uploader("Upload your PDFs", type=["pdf", "pptx"], accept_multiple_files=True)

if __name__ == "__main__":
    main()