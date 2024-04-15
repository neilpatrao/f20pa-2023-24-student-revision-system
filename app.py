import os
import tempfile

import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# Load files uploaded via Streamlit as documents
def get_docs(files):
    with tempfile.TemporaryDirectory() as tmpdir:
        for file in files:
            with open(os.path.join(tmpdir, file.name), "wb") as f:
                f.write(file.getbuffer())
        loader = DirectoryLoader(tmpdir)
        docs = loader.load()
        return docs

# Helper function to combine only the text content of multiple
# contexts when generating the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Splits documents in to chunks for better embeddings
def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

# Creates a retriever for the RAG pipeline
def create_retriever(chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Creates RAG chain
def create_chain(retriever):
    llm = ChatOpenAI(base_url="http://localhost:1234/v1",
                     api_key="lm-studio", temperature=0.1)
    
    template = """Answer the question using the context.

    question: {question}
    context: {context}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context","question"]
    )

    rag_chain = (
        {"context": retriever | format_docs,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Student Revision System", page_icon=":books:")

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    st.header("Student Revision System :books:")	

    with st.sidebar:
        st.subheader("Your Documents")
        files = st.file_uploader("Upload your documents", 
                                 type=["pdf", "pptx", "docx", 'txt', 'md'], 
                                 accept_multiple_files=True)
        if st.button("Process"):
            if files:
                with st.spinner("Processing Documents"):
                    docs = get_docs(files)
                    chunks = create_chunks(docs)
                    st.session_state.retriever = create_retriever(chunks)
                    st.session_state.chain = create_chain(st.session_state.retriever)
            else:
                st.write("No documents uploaded")
    
    query = st.chat_input("Ask your question")
    if query:
        with st.chat_message("user"):
            st.write(query)
        response = st.session_state.chain.invoke(query).strip()
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()