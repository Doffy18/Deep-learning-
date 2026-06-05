
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel
import re
import requests
import os

hf_token=os.getenv("HF_TOKEN")

from yt import youtube_transcript
from fastapi import APIRouter

router =APIRouter()

g_retriever = None
store={}

class Transcript(BaseModel):
    video_url: str

class Query(BaseModel):
    user_query : str

def extract_video_id(url: str) -> str:
    """Helper helper to safely extract YouTube Video ID for thumbnail generation"""
    pattern = r'(?:v=|\/v\/|embed\/|youtu\.be\/|\/shorts\/)([^?&"\'>]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else "default"




@router.post("/transcript")
def Complete_rag(payload: Transcript):
    global g_retriever
    video_url = payload.video_url
    transcript = youtube_transcript(video_url)

    try:
        video_id = extract_video_id(video_url)
        if not video_id or video_id == "default":
            raise ValueError("Invalid youtube ID Parsed")
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        video_title = f"YouTube Video Context (ID: {video_id})"
        oembed_url = f"https://www.youtube.com/oembed?url={video_url}&format=json"
        response = requests.get(oembed_url, timeout=5)
        if response.status_code == 200:
            metadata = response.json()

            if "title" in metadata:
                video_title = metadata["title"]
            if "thumbnail_url" in metadata:
                thumbnail_url = metadata["thumbnail_url"]
    except Exception as e:
        print(f"metadata extraction gone wrong {e}")
       


    document = [
        Document(
            page_content=transcript,
            metadata={"source": {video_url}}

        )
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    chunks = text_splitter.split_documents(document)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    g_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, })

    return {"message": "Transcript processed, Please proceed to ask questions",
            "title" : video_title ,
            "thumbnail": thumbnail_url
            }


@router.post("/query")
def rag(payload: Query):
    user_query = payload.user_query
    retriever = g_retriever

    llm = HuggingFaceEndpoint(
        repo_id= "deepseek-ai/DeepSeek-V3-0324",
        task = "text-generation",
        max_new_tokens = 128,
        huggingfacehub_api_token= hf_token,
        provider="auto"
    )
    model = ChatHuggingFace(llm = llm)


    # contextualization of new query prompt and chain
    Contextual_sys_prompt = """
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
    """
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", Contextual_sys_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    contextualize_chain = contextualize_prompt | model | StrOutputParser()


    # main answer prompt and chain
    normal_sys_prompt = """ Answer the user's question using only the following pieces of retrieved context. \
    If you do not know the answer, say that you don't know.\n\n{context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", normal_sys_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    def contextual_checking(input_dict):
        if input_dict.get("chat_history"):
            return contextualize_chain
        return RunnablePassthrough() | (lambda x: x["input"])

    rag_chain = RunnablePassthrough.assign(
        context = RunnableLambda(contextual_checking) | retriever) | prompt | model | StrOutputParser()


    # final chain with session persistence layer management
    store = {}

    def get_session_history(session_id : str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    Final_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history, input_messages_key = "input", history_messages_key = "chat_history"
    )


    response = Final_chain.invoke(
    {"input": user_query},
    config={"configurable": {"session_id": "user_session"}}
)

    return {"answer": response}