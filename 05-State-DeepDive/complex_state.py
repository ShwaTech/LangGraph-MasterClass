from typing import TypedDict, List, Annotated
from langgraph.graph import END, StateGraph
import operator


class BasicState(TypedDict):
    count: int
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]


def increament(state: BasicState) -> BasicState:
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "sum": new_count,
        "history": [new_count]
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
    "count": 0,
    "sum": 0,
    "history": []
}

response = graph.invoke(state)

print(response)

