from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from main import WorkFlowState

load_dotenv()

research_tool = TavilySearchResults(max_result = 2)

def research_bot(payload: dict) -> dict:
    query = payload["research"]
    research_answers = research_tool.invoke(
        [SystemMessage(
            content = "find the online researchs information about the query for blog Generation"),
            HumanMessage(
                content = f""" query: {query.querys}"""
            )
        ]
    ).content.strip()

    return{
        "researchAnswers" : research_answers
    }