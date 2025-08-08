from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, os
from pathlib import Path
from transformers import BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # use float16 for efficiency
    bnb_4bit_quant_type="nf4"               # better accuracy than 'fp4'
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='cuda')
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe,model_kwargs={
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 100
    })


# Define the custom prompt to reduce hallucination
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following pieces of context to answer the question at the end.
If no relevant context is provided or you don't know the answer, just say "I don't know" — do not try to make it up.

{context}

Question: {question}
Helpful Answer:"""

)

doc_folder = r"C:\Users\kashi\Documents\Projects\PDF chatbot with RAG\book"
loaders = []

for file_path in Path(doc_folder).rglob("*"):
    if file_path.suffix.lower() == ".txt":
        loaders.append(TextLoader(str(file_path), encoding="utf-8"))
    elif file_path.suffix.lower() == ".pdf":
        loaders.append(PyPDFLoader(str(file_path)))

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")


memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"score_threshold": 0.3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    return_source_documents=False
)

if __name__ == "__main__":

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break
        response = conv_chain.invoke({"question": user_query})
        print("Bot:", response["answer"])
