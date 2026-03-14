import streamlit as st

from code import (
    clone_repo,
    load_pyfiles,
    chunk_code,
    vector_store,
    load_llm,
    rag_chain
)

st.title("GitHub Codebase RAG Assistant")

repo_url = st.text_input(
    "Enter GitHub Repository URL[note: git clone only works with repository root, not a subfolder]")

if st.button("Load Repository", key="load_repo_btn"):

    st.write("Cloning repository...")
    repo_path, repo_name = clone_repo(repo_url)

    st.write("Loading Python files...")
    docs = load_pyfiles(repo_path)

    st.write("Chunking code...")
    chunks = chunk_code(docs)

    st.write("Preparing vector database...")
    vector_db = vector_store(chunks, repo_name)

    st.write("Loading LLM...")
    llm = load_llm()

    chain = rag_chain(vector_db, llm)

    st.session_state.chain = chain

    st.success("Repository loaded. Ask questions below.")


query = st.text_input("Ask a question about the codebase")

if query and "chain" in st.session_state:

    result = st.session_state.chain.invoke({"input": query})

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")

    for doc in result["context"]:

        st.write(
            f"{doc.metadata['file']} — {doc.metadata['name']} ({doc.metadata['type']})"
        )