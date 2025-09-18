from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile")


class State(TypedDict):
    messages: Annotated[list, add_messages]

GENERATE_POST="generate_post"
GET_REVIEW_DECISION="get_review_decision"
POST="post"
COLLECT_FEEDBACK="collect_feedback"


def generate_post(state: State):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

def get_review_decision(state: State):
    post_content = state["messages"][-1].content
    
    print("\n📢 Current LinkedIn Post:\n")
    print(post_content)
    print("\n")

    decision = input("✍🏽 Post To LinkIn? (yes/no): ")

    if decision.lower() == "yes":
        return POST
    return COLLECT_FEEDBACK

def post(state: State):
    final_post = state["messages"][-1].content
    print("\n🚀 Posting to LinkedIn:\n")
    print(final_post)
    print("\n✅ Post has been approved and is now live on LinkedIn!")

def collect_feedback(state: State):
    feedback = input("\n📝 Please provide feedback to improve the post: ")
    print("\n✅ Thank you for your feedback!")
    return {
        "messages": [HumanMessage(content=feedback)]
    }

graph_builder = StateGraph(State)

graph_builder.set_entry_point(GENERATE_POST)

graph_builder.add_node(GENERATE_POST, generate_post)
graph_builder.add_node(GET_REVIEW_DECISION, get_review_decision)
graph_builder.add_node(POST, post)
graph_builder.add_node(COLLECT_FEEDBACK, collect_feedback)

graph_builder.add_conditional_edges(GENERATE_POST, get_review_decision)
graph_builder.add_edge(POST, END)
graph_builder.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

graph = graph_builder.compile()


response = graph.invoke({
    "messages": [HumanMessage(content="Write me a LinkedIn post on AI Agents taking over content creation")]
})

print("\n🎉 Final LinkedIn Post:\n")
print(response["messages"][-1].content)

