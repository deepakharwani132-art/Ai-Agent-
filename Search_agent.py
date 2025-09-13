from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import os
from langchain_groq import ChatGroq


load_dotenv()

# 🔹 Set the Gemini API Key directly in code
os.environ["GEMINI_API_KEY"] = "***************************************"

print("Gemini API Key:", os.getenv("GEMINI_API_KEY")) 

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

llm = ChatGroq(
    groq_api_key=os.getenv("gsk_****************************************************"),
    model="llama-3.1-8b-instant",
    temperature=0.7
)

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'Paste your url here and api key here'

  response = requests.get(url)

  return response.json()


# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt


# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)


# Step 5: Invoke
response = agent_executor.invoke({"input": "Find the capital of pakistan, then find it's current weather condition"})
print(response)

response['output']