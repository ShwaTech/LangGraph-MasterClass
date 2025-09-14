from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, tool
from langchain_community.tools import TavilySearchResults
from datetime import datetime

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# response = llm.invoke("What is the Levels of Autonomy in LLM applications?")
# print(response)

search_tool = TavilySearchResults(
    search_depth="basic",
    k=3,
    max_results=3
)

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [search_tool, get_system_time]


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant?")

