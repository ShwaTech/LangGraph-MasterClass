from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_groq import ChatGroq
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])


## Actor Agent Prompt 
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are expert AI researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)


llm = ChatGroq(model="llama-3.3-70b-versatile")

llm_with_tools = llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")


first_responder_chain = first_responder_prompt_template | llm_with_tools


validator = PydanticToolsParser(tools=[AnswerQuestion])


response = first_responder_chain.invoke({
    "messages": [HumanMessage(content="Write me a blog post about Levels of Autonomy in LLM applications?")]
})

# print(response)



## Revisor section

revise_instructions = """
Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


first_revisor_prompt_template = actor_prompt_template.partial(
    first_instruction=revise_instructions
)

llm_with_tools_for_revisor = llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

first_revisor_chain = first_revisor_prompt_template | llm_with_tools_for_revisor

