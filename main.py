import re
import os
from pathlib import Path

from subgraph_image import subgraph_imagebulider
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

from schema import WorkFlowState

from llm_config import llm, search, research_tool, str_llm
from dotenv import load_dotenv
load_dotenv()

def router_bot(state: WorkFlowState) -> dict:
    prompt = SystemMessage(
        content=f""" you are senior Blog Planner, expertised more than 20 years in BLOG generation, Understand the topic for Blog geneation {state["user_input"]} and give Whether the topic needs to conduct Online research, if it requires online research also provide maximum 3 Querys. other wise : return an empty queries list"""
    )
    response = search.invoke([prompt])
    return{
        "research" : response
    }


def answerout(state: WorkFlowState):
    queries = state["research"].queries
    if not queries:
        return "planner_bot"

    return[Send("research_bot", {"query" : query}) for query in queries]


def research_bot(payload: dict) -> dict:
    query = payload["query"]
    research_answers = research_tool.invoke({"query":query})

    formatted_results = "\n".join(
        [
            f"Title: {item['title']}\n"
            f"Content: {item['content']}\n"
            f"URL: {item['url']}"
            for item in research_answers["results"]
        ]
    )
    return{
        "researchAnswers" : [formatted_results]
    }
   

def planner_bot(state: WorkFlowState) -> dict :
    research_context = "\n\n".join(
        state.get("researchAnswers", [])
    )
    prompt = SystemMessage(
        content = f"""Your a Senior Task Planner, craete a blog generation plan for : {state["user_input"]}. research findings: {research_context if research_context else "no online research required"}.Generate with maximum only 3 sections. keep each section focused and concise of the topic also use the rearch info awhen avaliable""")
    result = str_llm.invoke([prompt])
    return {
        "plan" : result 
    }


def fanout(state : WorkFlowState):
    return [Send("worker_bot", {"task" : task, "plan" : state["plan"], "research" : state.get("researchAnswers", [])}) for task in state["plan"].tasks]


def worker_bot(payload : dict) -> dict:
    task = payload["task"]
    plan = payload["plan"]
    research = "\n\n".join(payload["research"])

    section_md = llm.invoke(
        [
            SystemMessage(content = "write one clean markdown section."),
            HumanMessage(content = f"""
                Blog title: {plan.blog_title}
                Section title: {task.task_title}
                Description: {task.task_description}
                research findings: {research}
                Return only the Markdown content of this section.
                Do not add any extra text"""
            )
        ]
    ).content.strip()

    return {
        "sections" : [section_md]
    }


def reducer_bot(state : WorkFlowState) -> dict:
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"])

    final_md = f"# {title}\n\n{body}\n"

    return{
        "final" : final_md
    }

def output_bot(state:WorkFlowState) -> dict:
    title = state["plan"].blog_title
    final_md = state["temp_images_md"]

    # Sanitize title so it is a safe filename on every OS (no :, *, ?, ", <, >, |, /, \\)

    safe_title = re.sub(r"[^\w\-]+", "_", title, flags=re.UNICODE).strip("_").lower()
    if not safe_title:
        safe_title = "blog"
    filename = safe_title + ".md"
    output_dir = Path(__file__).resolve().parent/"Generated_blogs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    output_path.write_text(final_md, encoding="utf-8")

    return{
        "final_images_md" : final_md
    }

graph = StateGraph(WorkFlowState)

graph.add_node("router_bot", router_bot)
graph.add_node("research_bot", research_bot)
graph.add_node("planner_bot", planner_bot)
graph.add_node("worker_bot", worker_bot)
graph.add_node("reducer_bot", reducer_bot)
graph.add_node("subgraph_imagebulider", subgraph_imagebulider)
graph.add_node("output_bot", output_bot)

graph.add_edge(START, "router_bot")
graph.add_conditional_edges("router_bot", answerout)
graph.add_edge("research_bot", "planner_bot")
graph.add_conditional_edges("planner_bot", fanout)
graph.add_edge("worker_bot", "reducer_bot")
graph.add_edge("reducer_bot", "subgraph_imagebulider")
graph.add_edge("subgraph_imagebulider", "output_bot")
graph.add_edge("output_bot", END)

workflow = graph.compile()

inital_state = {"user_input" : "blog about OpenAI ChatGPT Evolution from 2022 to 2026", "sections" : []}
final_state = workflow.invoke(inital_state, config={"max_concurrency" : 2})
print(final_state)

