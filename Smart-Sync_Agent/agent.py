import os
import sys
from typing import Annotated, Literal
from pydantic import BaseModel, Field
import asyncio

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

class State(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)

async def run_agent_workflow(user_input_text: str, gemini_key: str, root_dir: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(current_dir, "mcp_server.py")

    client = MultiServerMCPClient({
        "cli_mcp": {
            "command": sys.executable,
            "args": [mcp_server_path],
            "transport": "stdio",
            "env": {
                **os.environ,                 
                "M_WORKSPACE_ROOT": root_dir  
            }
        }
    })

    async with client.session("cli_mcp") as session:
        langchain_tools = await load_mcp_tools(session)

        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=gemini_key)
        model_X_tools = model.bind_tools(langchain_tools)

        async def tool_calling_model(state: State):
            await asyncio.sleep(20) # Maintained exact original execution latency target
            return {"messages": [await model_X_tools.ainvoke(state.messages)]}

        def should_continue(state: State) -> Literal["tools", "__end__"]:
            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "__end__"

        graph = StateGraph(State)
        graph.add_node("agent", tool_calling_model)
        graph.add_node('tools', ToolNode(langchain_tools))
        graph.add_edge(START, 'agent')
        graph.add_conditional_edges('agent', should_continue)
        graph.add_edge('tools', 'agent')

        graph_m = graph.compile()
        
        config = {"configurable": {"thread_id": "unique-session-id-123"}}
        graph_input = {"messages": [{"role": "user", "content": user_input_text}]}
        final_agent_response = ""

        try:
            async for event in graph_m.astream(graph_input, stream_mode="values", config=config):
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    final_agent_response = last_msg.content
        except Exception as graph_err:
            print(f"\n[LLM/Graph Execution Error]: {str(graph_err)}")
            raise graph_err

        return final_agent_response
