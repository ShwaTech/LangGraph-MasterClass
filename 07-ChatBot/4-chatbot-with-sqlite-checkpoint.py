from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile")


sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)

memory = SqliteSaver(sqlite_conn)


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

graph = graph_builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": 1
    }
}


while True:
    user_input = input("👨 User: ")
    
    if user_input in ["exit", "quit", "bye", "goodbye", "q", "end"]:
        break
    
    response = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    }, config=config)
    
    print("🤖 Assistant: ", response["messages"][-1].content)

