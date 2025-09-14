from langchain_core.agents import AgentFinish, AgentAction
from langgraph.graph import StateGraph, END

from nodes import reason_node, act_node
from react_state import AgentState

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    return ACT_NODE

graph_builder = StateGraph(AgentState)

graph_builder.add_node(REASON_NODE, reason_node)

graph_builder.set_entry_point(REASON_NODE)

graph_builder.add_node(ACT_NODE, act_node)

graph_builder.add_conditional_edges(
    REASON_NODE,
    should_continue
)

graph_builder.add_edge(ACT_NODE, REASON_NODE)

graph = graph_builder.compile()

response = graph.invoke({
    "input": "How many days ago was the latest SpaceX launch?"
})

print(response)
print(response["agent_outcome"].return_values["output"], "final response")

