import streamlit as st
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Harry Potter RAG Chatbot", layout="wide")

st.image("hog.jpg", use_container_width=True)  

st.title("⚡ Harry Potter RAG Chatbot")
st.markdown("""
Welcome to the **Harry Potter RAG Chatbot**! 🪄  
This magical assistant can answer questions from the two favorite books: 
**Harry Potter and the Sorcerer's Stone** and **Harry Potter and the Chamber of Secrets**.  

Simply type your question below, and the magic of Retrieval-Augmented Generation will  
search the books and conjure up an answer just for you! ✨
""")

@st.cache_resource
def load_model_and_prompt():
    from RAG_pipeline import llm, custom_prompt
    return llm, custom_prompt

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever()

llm, custom_prompt = load_model_and_prompt()
retriever = load_vectorstore()

question = st.text_input("🔍 Ask your magical question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Consulting the Hogwarts library... 📚"):
            docs = retriever.get_relevant_documents(question)

            context = "\n\n".join([doc.page_content for doc in docs])

            chain = LLMChain(llm=llm, prompt=custom_prompt)
            raw_response = chain.run({"context": context, "question": question})

            if "Helpful Answer:" in raw_response:
                response = raw_response.split("Helpful Answer:")[-1].strip()
            else:
                response = raw_response.strip()

        st.subheader("Question:")
        st.write(question)

        st.subheader("Answer:")
        st.write(response)

        with st.expander("📖 Sources from the Books"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content)
                if hasattr(doc, 'metadata'):
                    st.markdown(f"**Metadata:** {doc.metadata}")
