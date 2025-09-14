from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=3)

tools = [search_tool]

llm = ChatGroq(model="llama-3.3-70b-versatile")

llm_with_tools = llm.bind_tools(tools=tools)


def chatbot(state: BasicChatBot):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }


def tools_router(state: BasicChatBot):
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_again"
    return "Exit"


tool_node = ToolNode(tools=tools)

## Building The Graph

graph_builder = StateGraph(BasicChatBot)

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.add_node("tools", tool_node)


graph_builder.add_conditional_edges(
    "chatbot",
    tools_router,
    {
        "tool_again": "tools",
        "Exit": END
    }
)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

print(graph.get_graph().draw_mermaid())

while True:
    user_input = input("👨 User: ")
    
    if user_input in ["exit", "quit", "bye", "goodbye", "q", "end"]:
        break
    
    response = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    
    print("🤖 Assistant: ", response["messages"][-1].content)

