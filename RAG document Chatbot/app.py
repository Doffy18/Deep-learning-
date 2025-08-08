import streamlit as st
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Page config ---
st.set_page_config(page_title="Harry Potter RAG Chatbot", layout="wide")

# --- Header with image ---
st.image("hog.jpg", use_container_width=True)  # Add an image file in same folder

# --- Title & Description ---
st.title("⚡ Harry Potter RAG Chatbot")
st.markdown("""
Welcome to the **Harry Potter RAG Chatbot**! 🪄  
This magical assistant can answer questions from the two favorite books: 
**Harry Potter and the Sorcerer's Stone** and **Harry Potter and the Chamber of Secrets**.  

Simply type your question below, and the magic of Retrieval-Augmented Generation will  
search the books and conjure up an answer just for you! ✨
""")

# --- Cache heavy resources so they load only once ---
@st.cache_resource
def load_model_and_prompt():
    from RAG_pipeline import llm, custom_prompt
    return llm, custom_prompt

@st.cache_resource
def load_vectorstore():
    # Make sure the SAME embedding model is used as in FAISS index creation
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever()

# --- Load resources once ---
llm, custom_prompt = load_model_and_prompt()
retriever = load_vectorstore()

# --- User Input ---
question = st.text_input("🔍 Ask your magical question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Consulting the Hogwarts library... 📚"):
            # Retrieve relevant docs
            docs = retriever.get_relevant_documents(question)

            # Join only the text for LLM
            context = "\n\n".join([doc.page_content for doc in docs])

            # Run LLM chain
            chain = LLMChain(llm=llm, prompt=custom_prompt)
            raw_response = chain.run({"context": context, "question": question})

            # Extract only the final answer
            if "Helpful Answer:" in raw_response:
                response = raw_response.split("Helpful Answer:")[-1].strip()
            else:
                response = raw_response.strip()

        # Display the clean Q/A
        st.subheader("Question:")
        st.write(question)

        st.subheader("Answer:")
        st.write(response)

        # Show sources separately
        with st.expander("📖 Sources from the Books"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content)
                if hasattr(doc, 'metadata'):
                    st.markdown(f"**Metadata:** {doc.metadata}")
