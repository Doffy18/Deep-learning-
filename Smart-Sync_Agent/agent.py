import os
from typing import Annotated
from pydantic import BaseModel, Field
import asyncio
import sys

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from typing import Literal
from langchain_mcp_adapters.client import MultiServerMCPClient



class State(BaseModel):
    messages:Annotated[list,add_messages] = Field(default_factory=list)

async def run_agent_workflow(user_input_text: str, gemini_key: str, root_dir: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(current_dir, "mcp_server.py")
    
    client = MultiServerMCPClient({
        "cli_mcp": {
            "command": sys.executable,
            "args": [mcp_server_path],
            "transport": "stdio",
            "env": {
                **os.environ,                 # Keep your standard shell system paths
                "M_WORKSPACE_ROOT": root_dir  # Forward targeted workspace boundary 
            }
        }
    })
    langchain_tools = await client.get_tools()

    model = init_chat_model("gemini-2.5-flash",model_provider="google_genai",api_key=gemini_key)
    model_X_tools = model.bind_tools(langchain_tools)


    async def tool_calling_model(state: State):
        await asyncio.sleep(20)
        return {"messages": [await model_X_tools.ainvoke(state.messages)]}


    def should_continue(state: State) -> Literal["tools", "__end__"]:
        """Inspects the last message to see if the LLM requested a tool execution loop."""
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"


    graph = StateGraph(State)
    #model with tool
    graph.add_node("agent", tool_calling_model)
    ##adding mcp
    graph.add_node('tools', ToolNode(langchain_tools))
    graph.add_edge(START, 'agent')
    graph.add_conditional_edges('agent',should_continue)
    graph.add_edge('tools','agent')

    graph_m = graph.compile()
    
    config = {"configurable": {"thread_id": "unique-session-id-123"}}
    graph_input = {"messages": [{"role": "user", "content": user_input_text}]}
    final_agent_response = ""


    # Locate this section near the bottom of agent.py and update it:
    try:
        async for event in graph_m.astream(graph_input, stream_mode="values", config=config):
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                final_agent_response = last_msg.content
    except Exception as graph_err:
        print(f"\n[LLM/Graph Execution Error]: {str(graph_err)}")
        raise graph_err

    return final_agent_response

# from IPython.display import Image, display
# # Save the architecture diagram directly to your workspace
# try:
#     with open("graph_architecture.png", "wb") as f:
#         f.write(graph_m.get_graph().draw_mermaid_png())
#     print("-> Architecture diagram successfully saved as 'graph_architecture.png'")
# except Exception as e:
#     print(f"Could not generate diagram: {e}")
