import streamlit as st
import pandas as pd
from io import BytesIO
from AutoClean import AutoClean
from test import AIMessage, HumanMessage, upload_dataset, app as agent_app

st.set_page_config(page_title="Agentic AI Dataset Chatbot", layout="wide")
st.title("Agentic AI Dataset Chatbot")
st.markdown("""
## About This Project

This is an **Agentic AI Dataset Chatbot** built with LangGraph and Streamlit.  
It allows you to **upload a CSV dataset** and interact with it using natural language.  

### Key Features:
- **Chat with the dataset**: Ask questions about rows, columns, shape, summary, or statistics.
- **Automatic dataset cleaning**: Fill missing values, remove duplicates, and perform advanced cleaning using `AutoClean`.
- **In-memory cleaning**: Clean datasets without saving any files to disk.
- **Download cleaned datasets**: Get the cleaned dataset directly from the browser.
- **Preview and inspect datasets**: View data samples and basic info interactively.

This tool combines **agentic AI reasoning** with **practical data cleaning**, making dataset preparation fast and user-friendly.
""")

st.write(
    "Upload a CSV dataset and chat with the bot about it. "
    "You can clean the dataset in-memory and download it without saving to disk."
)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

# Initialize session state variables if not present
if 'state' not in st.session_state:
    st.session_state['state'] = None
if 'uploaded_name' not in st.session_state:
    st.session_state['uploaded_name'] = None

if uploaded_file:
    # Only initialize if new file or first upload
    if st.session_state['state'] is None or uploaded_file.name != st.session_state['uploaded_name']:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin1")

        # Convert int columns to float64 for consistency
        for col in df.select_dtypes(include=["Int64", "Int32", "Int16"]).columns:
            df[col] = df[col].astype("float64")

        state = {
            "messages": [
                AIMessage(content="Dataset uploaded successfully."),
                AIMessage(content="You can chat about the dataset, type 'clean' to clean it, or ask questions."),
            ],
            "dataset_path": "in-memory.csv",  # placeholder path
            "df_json": df.to_json(),
            "cleaning_history": [],
        }
        st.session_state['state'] = state
        st.session_state['uploaded_name'] = uploaded_file.name

        # Show info messages only once after upload
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                st.info(msg.content)
    else:
        # Reuse existing state without showing upload messages again
        state = st.session_state['state']

    # Show dataset preview or info if requested
    if st.checkbox("Show dataset preview"):
        st.dataframe(pd.read_json(state["df_json"]).head())

    if st.checkbox("Show dataset info"):
        st.text(pd.read_json(state["df_json"]).info())

    # Chat input box
    user_input = st.text_input("Ask the bot about your dataset:", key="chat_input")
    if user_input:
        state["messages"].append(HumanMessage(content=user_input))
        config = {"configurable": {"thread_id": "chat-1"}}
        state = agent_app.invoke(state, config=config)

        # Show only latest AI message (avoid repeating old ones)
        for msg in state["messages"][-1:]:
            if isinstance(msg, AIMessage):
                st.success(msg.content)

        # Reset messages to avoid clutter, but keep state updated
        state["messages"] = []
        st.session_state['state'] = state

    # In-memory dataset cleaning function
    def clean_in_memory(df: pd.DataFrame) -> pd.DataFrame:
        # Fill numeric nulls with mean
        for col in df.select_dtypes(include=["number"]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
        # Fill categorical nulls with mode
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
        # Apply AutoClean in-memory
        cleaner = AutoClean(df, mode="auto")
        return cleaner.output

    # Dataset cleaning button
    if st.button("Clean Dataset In-Memory"):
        cleaned_df = clean_in_memory(pd.read_json(state["df_json"]))
        st.success("Dataset cleaned in-memory ")

        # Download cleaned dataset
        buffer = BytesIO()
        cleaned_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Download Cleaned Dataset",
            data=buffer,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

else:
    st.warning("Please upload a CSV file to start chatting with the bot.")
    st.session_state['state'] = None
    st.session_state['uploaded_name'] = None

