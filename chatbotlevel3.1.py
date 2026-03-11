from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "chatbotlevel3_debugandmon2"
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

from langchain.chat_models import init_chat_model
llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq"
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_tool_graph():
    ## Graph With tool Call
    @tool
    def add(a: float, b: float):
        """Add two numbers"""
        return a + b
    
    tools = [add]
    llm_with_tool = llm.bind_tools(tools)

    def call_llm_model(state: State):
        return {"messages": [llm_with_tool.invoke(state['messages'])]}

    ## Graph
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools", "tool_calling_llm")

    ## compile the graph
    graph = builder.compile()
    return graph

tool_agent = make_tool_graph()