import os
import ast
import streamlit as st
import subprocess

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import ChatHuggingFace


# we clone repo  
def clone_repo(repo_url):
    os.makedirs("repos", exist_ok=True)

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = f"repos/{repo_name}"

    if not os.path.exists(repo_path):
        subprocess.run(
            ["git", "clone", repo_url, repo_path],
            check=True
        )
    return repo_path, repo_name


# we load all .py files in the repo as documents
def load_pyfiles(file_path):
    documents = []
    for root, dirs, files in os.walk(file_path):
        for i in files:
            if i.endswith(".py"):
                path = os.path.join(root, i)
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                documents.append(
                    Document(page_content=code, metadata={"source": path})
                )
    return documents


# we chunk code by function and class definitions using ast
def chunk_code(documents):
    chunks = []
    for doc in documents:
        try:
            tree = ast.parse(doc.page_content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start = node.lineno
                    end = node.end_lineno
                    lines = doc.page_content.split("\n")
                    chunk = "\n".join(lines[start - 1 : end])
                    chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "file": doc.metadata["source"],
                                "name": node.name,
                                "type": type(node).__name__,
                            },
                        )
                    )
        except:
            pass
    return chunks


# we create or load vector database for the code chunks
def vector_store(chunks, repo_name):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db_path = f"vector_store/{repo_name}"

    if os.path.exists(db_path):
        vector_db = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vector_db = FAISS.from_documents(chunks, embeddings)
        os.makedirs("vector_store", exist_ok=True)
        vector_db.save_local(db_path)
    return vector_db


# we load a free conversational LLM from Hugging Face Hub note: place you huggingface token here, some models require authentication
hf_token = ""
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3-0324",
        task="conversational",
        max_new_tokens=128,
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        top_p=0.9,
    )
    llm = ChatHuggingFace(llm=llm)
    return llm


# we create a RAG chain that retrieves relevant code chunks and uses the LLM to answer questions based on them
def rag_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """you are an expert Python developer who explains codebases.
use only the retrieved code context to answer the question. do not hallucinate.

{context}

question: {input}

provide a clear explanation and mention the function/class name.
"""
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

#streamlit app in app.py