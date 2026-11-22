from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage
from main import WorkFlowState, image_llm
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from image_generator import image_process
load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.1
)

def image_placer_bot(state: WorkFlowState):
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"])
    without_image = f"# {title}\n\n{body}\n"
    prompt = SystemMessage(
        content=f""" understand the {without_image} blog and as the high level senior blog Generator decide maximum 1-3 images where to be placed in blog at different places and also generate the placeholder, filename, alt, catption, size, quality and the prompt text for Image"""
    )
    response = image_llm.invoke(prompt)
    return {
        "with_image" : response
    }


def imageout(state:WorkFlowState):
    return[Send("image_generator_bot", {"image" : image})for image in state["with_image"].images]


def image_generator_bot(payload: dict) -> dict:
    image = payload["image"]
    generated_image = image_process(image)
    if not generated_image:
        return {"image_details": []}

    return {
        "image_details": [image]
    }

def image_merge_bot(state:WorkFlowState):
    image = state["with_image"]
    body = state["final"]
    prompt = SystemMessage(
        f""" you have the both the images and body of the Blog {image}, {body} now generate the full blog with mix of image as the High quality senior Blog generator"""
    )
    response = llm.invoke(prompt)
    return {
        "full_images_md" : response.content
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

