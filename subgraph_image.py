from langgraph.types import Send
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from schema import WorkFlowState
from llm_config import image_llm, llm

from image_generator import image_process
from dotenv import load_dotenv
load_dotenv()

def image_placer_bot(state: WorkFlowState):
    without_image = state["final"]
    prompt = SystemMessage(
        content=f""" understand the {without_image} blog and as the high level senior blog, and decide maximum 1-3 images where to images needed to placed in blog at different places and also generate the placeholder, filename, alt, catption, size, quality and the prompt text for Image"""
    )
    response = image_llm.invoke([prompt])
    return {
        "image_info" : response
    }


def imageout(state:WorkFlowState):
    return[Send("image_generator_bot", {"image" : image})for image in state["image_info"].images]


def image_generator_bot(payload: dict) -> dict:
    image = payload["image"]
    generated_image = image_process(image)
    if not generated_image:
        return {"image_details": []}

    return {
        "image_details": [image]
    }

def image_merge_bot(state:WorkFlowState):
    image = state.get("image_details", [])
    body = state["final"]
    if not image:
        return{
            "temp_images_md" : body
        }

    prompt = SystemMessage(
        content=f"""
        You are a senior Markdown blog editor.

        Original blog:
        {body}

        Successfully generated images:
        {image}

        Insert 1-3 images at relevant positions in the blog.

        Use Markdown image syntax:
        ![alt text](images/filename.png)

        Use the exact filenames and alt text from the image specifications.
        Add the provided captions below the images.
        Preserve the original blog content.
        Do not invent images or filenames.

        Return only the complete Markdown blog.
        """
    )
    response = llm.invoke([prompt])
    return {
        "temp_images_md" : response.content
    }


image_graph = StateGraph(WorkFlowState)

image_graph.add_node("image_placer_bot", image_placer_bot)
image_graph.add_node("image_generator_bot", image_generator_bot)
image_graph.add_node("image_merge_bot", image_merge_bot)

image_graph.add_edge(START, "image_placer_bot")
image_graph.add_conditional_edges("image_placer_bot", imageout)
image_graph.add_edge("image_generator_bot", "image_merge_bot")
image_graph.add_edge("image_merge_bot", END)

subgraph_imagebulider = image_graph.compile()

