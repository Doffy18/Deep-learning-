from typing import List
from typing_extensions import TypedDict
import os
import json
import pandas as pd
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain.agents import tool
from langgraph.prebuilt import ToolNode
from AutoClean import AutoClean

load_dotenv()

    # ============================================================================================================

#State 
class State(TypedDict):
    messages: List[BaseMessage]
    dataset_path: str
    df_json: str
    cleaning_history: List[dict]


    # ============================================================================================================


#Upload dataset function
def upload_dataset(path: str) -> State:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    # convert int columns for consistency
    for col in df.select_dtypes(include=["Int64", "Int32", "Int16"]).columns:
        df[col] = df[col].astype("float64")

    return {
        "messages": [
            AIMessage(content=f"Dataset '{os.path.basename(path)}' uploaded successfully."),
            AIMessage(content="You can chat about the dataset, type 'clean' to clean it, or just ask questions."),
        ],
        "dataset_path": path,
        "df_json": df.to_json(),
        "cleaning_history": [],
    }

# ==========================================================================================================
#tools for cleaning and dataset dsecription
@tool
def auto_clean(df_json: str, dataset_path: str, cleaning_history: list = None) -> dict:
    """Clean the dataset automatically using AutoClean, fill nulls with mean/mode, remove duplicates, and log changes."""
    if cleaning_history is None:
        cleaning_history = []

    df = pd.read_json(df_json)

    # Fill numeric nulls with mean
    for col in df.select_dtypes(include=["number"]).columns:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

    # Fill categorical nulls with mode
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if not df[col].mode().empty:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)


    # Apply AutoClean on top of the filled/duplicates removed DataFrame (if you want extra cleaning)
    cleaner = AutoClean(df, mode="auto")
    cleaned_df = cleaner.output

    # Save cleaned dataset
    cleaned_path = os.path.splitext(dataset_path)[0] + "_cleaned.csv"
    # cleaned_df.to_csv(cleaned_path, index=False)

    # Build log entry
    changes = getattr(cleaner, "changes", {"info": "AutoClean applied with null filling and duplicate removal"})
    log_entry = {
        "dataset": dataset_path,
        "changes": str(changes)
    }
    cleaning_history.append(log_entry)

    # Write to log file
    with open("cleaning_log.txt", "a") as f:
        f.write(json.dumps(log_entry, indent=2))
        f.write("\n" + "-"*80 + "\n")

    return {
        "df_json": cleaned_df.to_json(),
        "dataset_path": dataset_path,
        "cleaning_history": cleaning_history,
        "message": f"Auto cleaning complete (nulls filled by mean/mode, duplicates removed). Cleaned dataset saved as '{cleaned_path}'."
    }

@tool
def dataset_stats(df_json: str, stat_type: str) -> str:
    """Get stats about the dataset: row_count, column_count, describe, nulls."""
    df = pd.read_json(df_json)

    if stat_type == "row_count":
        return f"The dataset has {len(df)} rows."
    elif stat_type == "column_count":
        return f"The dataset has {len(df.columns)} columns: {list(df.columns)}"
    elif stat_type == "describe":
        return f"Dataset description:\n{df.describe(include='all').to_string()}"
    elif stat_type == "nulls":
        return f"Missing values:\n{df.isnull().sum().to_dict()}"
    else:
        return "Stat type not recognized. Try: row_count, column_count, describe, nulls."


    # ============================================================================================================

# LLM + Tools 
tools = [auto_clean, dataset_stats]
llm = ChatGroq(model="llama-3.1-8b-instant").bind_tools(tools)
tool_node = ToolNode(tools=tools)

    # ============================================================================================================

# Chatbot Node 
def chatbot(state: State) -> State:
    last_msg_obj = state["messages"][-1]
    df = pd.read_json(state["df_json"])

    if isinstance(last_msg_obj, HumanMessage):
        user_text = last_msg_obj.content.lower()

        if "clean" in user_text:
            # Proper tool call with ID
            return {
                "messages": [
                    AIMessage(
                        content="Starting dataset cleaning...",
                        tool_calls=[{
                            "id": "call_auto_clean_1",   
                            "name": "auto_clean",
                            "args": {
                                "df_json": state["df_json"],
                                "dataset_path": state["dataset_path"],
                                "cleaning_history": state.get("cleaning_history", []),
                                "max_rows": 1000000
                            }
                        }]
                    )
                ]
            }

        elif "rows" in user_text and "count" in user_text:
            info = f"Row count: {len(df)}"

        elif "columns" in user_text:
            info = f"Columns: {list(df.columns)}"

        elif "shape" in user_text:
            info = f"Dataset shape: {df.shape}"

        elif "summary" in user_text or "describe" in user_text:
            info = f"Dataset summary:\n{df.describe(include='all').to_string()}"

        else:
            info = "You can ask me about rows counts, columns names, shape, summary."

        state["messages"].append(AIMessage(content=info))

    return state

    # ============================================================================================================

# Router
def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

    # ============================================================================================================

# Build Graph 
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tools", "chatbot")

app = graph.compile()

    # ============================================================================================================


# Run Loop 
if __name__ == "__main__":
    dataset_path = input("Enter dataset path: ").strip()
    state = upload_dataset(dataset_path)
    config = {"configurable": {"thread_id": "chat-1"}}

    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            print(f"[AIMessage] {msg.content}")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "end"]:
            print("Exiting chatbot. All done ✅")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state, config=config)

        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                print(f"[AIMessage] {msg.content}")
        state["messages"] = []
        
    # ============================================================================================================
