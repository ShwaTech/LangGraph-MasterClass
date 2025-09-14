from langgraph.prebuilt.tool_node import ToolNode
from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {
        "agent_outcome": agent_outcome,
    }

tool_node = ToolNode(tools)

def act_node(state: AgentState):
    agent_action = state["agent_outcome"]
    output = tool_node.invoke(agent_action)
    return {
        "intermediate_steps": [(agent_action, str(output))]
    }
