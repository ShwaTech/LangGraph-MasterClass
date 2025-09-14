from langchain_groq import ChatGroq
from langchain.agents import tool, create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain import hub
import datetime

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile")


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

search_tool = TavilySearchResults(search_depth="basic")

tools = [get_system_time, search_tool]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(
    tools=tools, 
    llm=llm,
    prompt=react_prompt,
    verbose=True
)

