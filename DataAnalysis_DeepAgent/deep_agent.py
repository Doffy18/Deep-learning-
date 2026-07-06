
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, List
import importlib.resources

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7
from langchain_core.callbacks import AsyncCallbackHandler
from deepagents import create_deep_agent

class DelayLLMCallback(AsyncCallbackHandler):
    async def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any
    ) -> None:
        """Throttles chat completions (Gemini 2.5) on every turn."""
        await asyncio.sleep(20)

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Fallback for standard LLM completions."""
        await asyncio.sleep(20)
        
async def main(user_input_text: str):
    try:
        with importlib.resources.path("deep_agent", "mcp_server.py") as p:
            server_path = str(p)
    except (ImportError, ModuleNotFoundError):
        server_path = str(Path(__file__).parent.resolve() / "mcp_server.py")

    skills_path = str(Path(__file__).parent.resolve() / ".skills/data_skill/")

    client = MultiServerMCPClient(
        {
            "deep_agent_1": {
                "command": sys.executable,
                "transport": "stdio",
                "args": [server_path]
            }
        }
    )
    deep_agent_tools = await client.get_tools()

    agent = create_deep_agent(
        model="google_genai:gemini-2.5-flash",
        tools=deep_agent_tools,
        checkpointer=InMemorySaver(),
        skills=skills_path,
    )

    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [DelayLLMCallback()]}
    input = {"messages": [{"role": "user", "content": user_input_text}]}

    final_agent_response = "No response generated."

    try:
        result = await agent.ainvoke(input, config=config)
        if "messages" in result and result["messages"]:
            final_agent_response = result["messages"][-1].content
    except Exception as e:
        print(f"\n[LLM/ Execution Error]: {str(e)}")
        raise e

    return final_agent_response