from deepagents import create_deep_agent
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7


async def main(user_input_text: str):
    client = MultiServerMCPClient(
        {
            "deep_agent_1": {
                "command": sys.executable,
                "transport": "stdio",
                "args": ["mcp_server.py"]
            }
        }
    )
    deep_agent_tools = await client.get_tools()


    agent = create_deep_agent(
            model="gemini-2.5-flash",
            tools=deep_agent_tools,
            checkpointer=InMemorySaver(),
            skills= '.skills/data_skill/'
            )

    thread_id = str(uuid7())
    config={"configurable": {"thread_id": thread_id}}
    input = {"messages": [{"role": "user", "content": user_input_text}]}

    try:
        async for event in agent.invoke(input,config=config):
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                final_agent_response = last_msg.content
    except Exception as e:
            print(f"\n[LLM/ Execution Error]: {str(e)}")
            raise e

    return final_agent_response