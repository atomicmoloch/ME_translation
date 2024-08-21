from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent


model=ChatOllama(model="llama3")


#agent = create_csv_agent(
#    model,
#    ["./ME_words.csv"],
#    verbose=True)

function =


chain = function | model | StrOutputParser


while True:
    query = input("> ")
    agent.run(query)
