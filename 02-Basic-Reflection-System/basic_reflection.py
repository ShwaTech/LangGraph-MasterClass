from typing import List, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from chains import generation_chain, reflection_chain

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


REFLECT = "reflect"
GENERATE = "generate"

# Define the state schema
class GraphState(TypedDict):
    messages: List[BaseMessage]

## Building The Graph
graph_builder = StateGraph(GraphState)


def generate_node(state: GraphState) -> GraphState:
    response = generation_chain.invoke({
        "messages": state["messages"]
    })
    return {"messages": state["messages"] + [response]}

def reflect_node(state: GraphState) -> GraphState:
    response = reflection_chain.invoke({
        "messages": state["messages"]
    })
    return {"messages": state["messages"] + [HumanMessage(content=response.content)]}

## Adding The Nodes
graph_builder.add_node(GENERATE, generate_node)
graph_builder.add_node(REFLECT, reflect_node)

## Setting The Start Node
graph_builder.set_entry_point(GENERATE)

## Should Continue From The Reflect Node In The Loop Until The Length of the State is Greater Than 4
def should_continue(state: GraphState) -> str:
    # Check if we have enough messages to continue reflection
    # We need at least 2 messages: user input + generated response
    if len(state["messages"]) > 5:
        return END
    return REFLECT

## Adding The Edges
graph_builder.add_conditional_edges(GENERATE, should_continue)
graph_builder.add_edge(REFLECT, GENERATE)

## Compiling The Graph
graph = graph_builder.compile()

## Visualizing The Graph
print("Graph Structure:")
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii() 

## Test the graph
def test_graph():
    print("\n" + "="*50)
    print("Testing the Reflection Graph ✨")
    print("="*50)
    
    # Initial state with user message
    initial_state = {
        "messages": [HumanMessage(content="Write a tweet about the Levels of Autonomy in LLM Apps?")]
    }
    
    print(f"Initial state: {len(initial_state['messages'])} messages")
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    print(f"Final state: {len(result['messages'])} messages")
    print("\nConversation flow:")
    for i, msg in enumerate(result['messages']):
        print(f"{i+1}. {type(msg).__name__}: {msg.content[:100]}...")

if __name__ == "__main__":
    test_graph()
