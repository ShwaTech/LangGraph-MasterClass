from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import first_revisor_chain, first_responder_chain
from execute_tools import execute_tools

graph_builder = MessageGraph()
MAX_ITERATIONS = 2

graph_builder.add_node("draft", first_responder_chain)
graph_builder.add_node("execute_tools", execute_tools)
graph_builder.add_node("revisor", first_revisor_chain)


graph_builder.add_edge("draft", "execute_tools")
graph_builder.add_edge("execute_tools", "revisor")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

graph_builder.add_conditional_edges("revisor", event_loop)
graph_builder.set_entry_point("draft")

graph = graph_builder.compile()

print(graph.get_graph().draw_mermaid())

response = graph.invoke(
    "Write about how small business can leverage AI to grow"
)

print(response[-1].tool_calls[0]["args"]["answer"])
print(response, "response")