from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from pydantic import BaseModel,Field
from typing import Annotated
from tool import send_email_tool, human_clarification
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_litellm import ChatLiteLLM
from typing import Literal
import os
os.environ["GROQ_API_KEY"] = os.getenv('groq')
os.environ["GEMINI_API_KEY"] = os.getenv('gemini')

class State(BaseModel):
    messages:Annotated[list,add_messages] = Field(default_factory=list)


agent_email001P = ChatLiteLLM(model="groq/llama-3.3-70b-versatile", temperature=0.7)
agent_email002F = ChatLiteLLM(model="gemini/gemini-2.5-flash", temperature=0.7)
model = agent_email001P.with_fallbacks([agent_email002F])


# model = init_chat_model("groq:llama-3.3-70b-versatile", temperature=0.7)
tool = [human_clarification,send_email_tool]
model_w_tool = model.bind_tools(tool)



def tool_calling_llm(state:State):
    return {"messages":[model_w_tool.invoke(state.messages)]}


def router_for_tools(state: State) -> Literal['clarification', 'send_email', "__end__"]:
    """Inspects the LLM's last response and determines exactly where to go next."""
    last_message = state.messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "__end__"
    
    tool_names = [tool["name"] for tool in last_message.tool_calls]

    if "send_email_tool" in tool_names:
        return "send_email"
        
    if "human_clarification" in tool_names:
        return "clarification"
        
    return "__end__"

graph = StateGraph(State)
graph.add_node("llm_tool",tool_calling_llm)
graph.add_node("clarification", ToolNode([human_clarification]))
graph.add_node("send_email", ToolNode([send_email_tool]))

graph.add_edge(START,"llm_tool")
graph.add_conditional_edges("llm_tool", router_for_tools, {
        "clarification": "clarification",
        "send_email": "send_email",
        "__end__": END
    }
    )

graph.add_edge("clarification", "llm_tool")
graph.add_edge("send_email", END)



memory = MemorySaver()
graph_m = graph.compile(checkpointer=memory)



def start_email_agent(thread_id: str, prompt: str):
    """Initializes the graph with a user prompt and runs until it pauses or ends."""
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_input = {
        "messages": [
            (
                "system", 
                "You are a helpful assistant. Crucial Rule: You are NEVER allowed to send an email via conversational text alone. "
                "You MUST always call the `human_clarification` tool to present your draft to the user first.\n"
                "CRITICAL APPROVAL HANDLING: If the user says words like 'ok', 'send', 'looks good', 'please send', 'go ahead', 'approved', or any variation of agreement, "
                "you must consider this explicit approval and IMMEDIATELY execute the `send_email_tool` for ALL recipients. Do not draft it again."
            ),
            ("user", prompt)
        ]
    }
    
    events = graph_m.stream(initial_input, config, stream_mode="updates")
    return process_stream_until_paused(events, config)


def resume_email_agent(thread_id: str, human_input: str):
    """Resumes an existing paused graph execution with the human's feedback string."""
    config = {"configurable": {"thread_id": thread_id}}
    human_feedback = {"data": human_input}
    
    events = graph_m.stream(Command(resume=human_feedback), config, stream_mode="updates")
    return process_stream_until_paused(events, config)


def process_stream_until_paused(events, config) -> dict:
    """Helper utility to consume stream events and format a clean API response."""
    for event in events:
        pass 
        
    current_state = graph_m.get_state(config)
    
    if current_state.next and "clarification" in current_state.next:
        if current_state.tasks and current_state.tasks[0].interrupts:
            pending_draft = current_state.tasks[0].interrupts[0].value['query']
            return {
                "status": "paused_for_review",
                "thread_id": config["configurable"]["thread_id"],
                "draft": pending_draft
            }
            
    return {
        "status": "completed",
        "thread_id": config["configurable"]["thread_id"],
        "final_output": current_state.values["messages"][-1].content
    }


# from IPython.display import Image, display

# # Save the architecture diagram directly to your workspace
# try:
#     with open("graph_architecture.png", "wb") as f:
#         f.write(graph_m.get_graph().draw_mermaid_png())
#     print("-> Architecture diagram successfully saved as 'graph_architecture.png'")
# except Exception as e:
#     print(f"Could not generate diagram: {e}")
