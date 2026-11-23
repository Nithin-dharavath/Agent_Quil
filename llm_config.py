from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from schema import Plan, ReSearch, GloablImage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0
)

llm2 = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.3
)

research_tool = TavilySearch(max_results = 2, type = "general")

str_llm = llm.with_structured_output(Plan, method="json_schema")
search = llm.with_structured_output(ReSearch, method="json_schema")
image_llm = llm2.with_structured_output(GloablImage, method="json_schema")