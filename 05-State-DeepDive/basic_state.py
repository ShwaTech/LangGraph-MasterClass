from typing import TypedDict
from langgraph.graph import END, StateGraph


class BasicState(TypedDict):
    count: int


def increament(state: BasicState) -> BasicState:
    return {
        "count": state["count"] + 1
    }


def should_continue(state: BasicState) -> str:
    if state["count"] < 5:
        return "Continue"
    return "Exit"


graph_builder = StateGraph(BasicState)

graph_builder.add_node("increment", increament)

graph_builder.set_entry_point("increment")

graph_builder.add_conditional_edges(
    "increment",
    should_continue,
    {
        "Continue": "increment",
        "Exit": END
    }
)

graph = graph_builder.compile()

print(graph.get_graph().draw_mermaid())


state = {
    "count": 0
}

response = graph.invoke(state)

print(response)

