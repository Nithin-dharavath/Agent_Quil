from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Literal

import operator

class ReSearch(BaseModel):
    queries: List[str]

class Task(BaseModel):
    task_id: int
    task_title: str
    task_description: str = Field(description = "what to cover ?")

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class ImageSpec(BaseModel):
    placeholder:str
    filename:str
    alt:str
    caption:str
    prompt:str
    size:Literal["1024x1024", "1024x1536", "1536x1024"]
    quality:Literal["low", "medium", "high"]

class GloablImage(BaseModel):
    images: List[ImageSpec]


class WorkFlowState(TypedDict):
    user_input : str
    research : ReSearch
    researchAnswers : Annotated[list[str], operator.add]
    plan : Plan
    sections : Annotated[list[str], operator.add]
    final : str
    image_details : Annotated[list[ImageSpec], operator.add]
    image_info : GloablImage
    temp_images_md : str
    final_images_md : str