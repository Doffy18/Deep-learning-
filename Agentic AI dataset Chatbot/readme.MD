# Agentic AI Dataset Chatbot

This is a **narrow-scope Agentic AI-powered Dataset Chatbot** that can automatically clean datasets, provide insights, and assist with various dataset-related tasks. While it operates autonomously within its domain, its primary goal is to simplify data preprocessing through interactive guidance and tool usage.

The chatbot can decide which actions to take (e.g., cleaning, summarizing, or displaying dataset stats) based on user input, maintain state across the conversation, and call tools dynamically to achieve its objectives. This makes it an **agentic AI in a limited, domain-specific sense**.

The project is deployed as a **Streamlit web app**, allowing you to upload datasets and interact with the chatbot directly from your browser.


## Features

- **Automatic Dataset Cleaning**: Uses `AutoClean` to fill missing values (numeric with mean, categorical with mode) and remove duplicates.
- **Dataset Exploration**: Ask the chatbot to display:
  - Row count
  - Column names and count
  - Dataset shape
  - Summary statistics (`describe`)
  - Missing values
- **Interactive Chat Interface**: Communicate with the bot to clean, explore, or ask questions about your dataset.
- **Logging**: Keeps a log of cleaning actions in `cleaning_log.txt`.

---

## Workflow

The chatbot uses a **graph-based architecture** (`langgraph`) with nodes representing chatbot behavior and tools. Here's how it works:

1. **Dataset Upload**: User uploads a CSV file. Numeric columns are standardized.
2. **Chat Node (`chatbot`)**: Receives user messages and determines the action:
   - If the user requests cleaning (`clean`), it triggers the cleaning tool.
   - If the user requests stats (`rows`, `columns`, `shape`, `summary`), it responds with relevant info.
3. **Tool Node (`ToolNode`)**: Executes functions like `auto_clean` and `dataset_stats`.
4. **State Management**: The chatbot maintains a `state` dictionary storing:
   - Messages (AI and human)
   - Dataset JSON
   - Cleaning history
5. **Routing**: Conditional edges determine whether to route the conversation to tools or back to the chatbot node.
6. **LLM Integration**: Uses `ChatGroq` (LLaMA 3.1-8b Instant) to handle natural language understanding and tool invocation.

The modular graph structure allows the chatbot to act autonomously and call tools dynamically based on user input.

---

## Future Scope

This project can be significantly enhanced in the future to make it more powerful and flexible:

1. **More Cleaning Tools**: Integrate advanced cleaning libraries like `Cleanlab` or anomaly detection tools to improve dataset quality.
2. **Manual Cleaning via Chat**: Enable the chatbot to perform column-wise or row-wise corrections interactively based on user instructions.
3. **Support for Multiple File Formats**: Extend support to Excel, JSON, Parquet, or other structured data formats.
4. **Enhanced Data Analysis Tools**: Add visualizations, correlation analysis, outlier detection, and feature suggestions directly in chat.
5. **Knowledge-Based Suggestions**: Use RAG-style retrieval or knowledge databases to suggest cleaning strategies or preprocessing steps.
6. **Automated Logging and Versioning**: Keep track of multiple cleaned versions and allow easy rollback or comparisons.
7. **Collaboration Features**: Allow multiple users to interact with the chatbot on the same dataset with shared state tracking.

## Requirements

The project requires the following Python libraries:

- `pandas`
- `numpy`
- `langgraph`
- `langchain_core`
- `langchain_groq`
- `langchain.agents`
- `AutoClean`
- `python-dotenv`
- `streamlit` (for web deployment)
- `typing_extensions`
- `model used- llama-3.1-8b-instant`
