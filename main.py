import operator
from pathlib import Path
from typing import TypedDict, Annotated, List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


class Task(BaseModel):
    task_id: int
    task_title: str
    task_description: str = Field(description = "what to cover ?")

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class WorkFlowState(TypedDict):
    user_input : str
    plan : Plan
    sections : Annotated[list[str], operator.add]
    final : str

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.2
)

str_llm = llm.with_structured_output(Plan)

def planner_bot(state: WorkFlowState) -> dict :
    prompt = SystemMessage(
        content = f"""Your a Senior Task Planner, craete a blog generation plan for : {state["user_input"]}. Generate between 4 -8 range tasks each atsk should cover a different aspect of the topic""")
    result = str_llm.invoke([prompt])
    return {
        "plan" : result 
    }

def fanout(state : WorkFlowState):
    return [Send("worker_bot", {"task" : task, "plan" : state["plan"]}) for task in state["plan"].tasks]


def worker_bot(payload : dict) -> dict:
    task = payload["task"]
    plan = payload["plan"]

    section_md = llm.invoke(
        [
            SystemMessage(content = "write one clean markdown section."),
            HumanMessage(content = f"""
                Blog title: {plan.blog_title}
                Section title: {task.task_title}
                Description: {task.task_description}
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

    filename = title.lower().replace(" ", "_")+".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return{
        "final" : final_md
    }

graph = StateGraph(WorkFlowState)

graph.add_node("planner_bot", planner_bot)
graph.add_node("worker_bot", worker_bot)
graph.add_node("reducer_bot", reducer_bot)

graph.add_edge(START, "planner_bot")
graph.add_conditional_edges("planner_bot", fanout)
graph.add_edge("worker_bot", "reducer_bot")
graph.add_edge("reducer_bot", END)

workflow = graph.compile()

inital_state = {"user_input" : "Self Attention", "sections" : []}
final_state = workflow.invoke(inital_state)
print(final_state)
