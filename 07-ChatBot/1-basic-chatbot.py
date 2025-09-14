from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile")


class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }


graph_builder = StateGraph(BasicChatState)

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

print(graph.get_graph().draw_mermaid())

while True:
    user_input = input("User: ")
    
    if user_input in ["exit", "quit", "bye", "goodbye", "q", "end"]:
        break
    
    response = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    
    print("Assistant: ", response["messages"][-1].content)

