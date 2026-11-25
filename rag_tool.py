from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import SystemMessage
from llm_config import llm2

from langgraph.graph import StateGraph, START, END

from typing import TypedDict

from pydantic import BaseModel
import re

from dotenv import load_dotenv
load_dotenv()


pdf_location = PyPDFLoader("./resources/Senior Blog Writer Master Operating Manual CRAG  Self-RAG Index.pdf")
documents = pdf_location.load()


spilt = RecursiveCharacterTextSplitter(chunk_overlap=120, chunk_size=460)
chunks = spilt.split_documents(documents)
print(len(chunks))

#embedding
embeddings = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

#store embeddings in qdrant
vector_store = QdrantVectorStore.from_existing_collection(
    # documents=chunks,
    embedding=embeddings,
    path = "./qdrant_vd",
    collection_name = "qdrant_embeddings"
)
# print(len(vector_store))

#retriver helper
retriver = vector_store.as_retriever(search_kwargs={"k" : 2})

query = "what is the best way to write a blog post ?"

result = retriver.invoke(query)
all_sentence = []
for d in result:
    sentences = re.split(r'(?<=[.!?])\s+', d.page_content.strip())
    all_sentence.extend(sentences)

def recompose_info(query, all_sentence):
    recompose = llm2.invoke(
        [
            SystemMessage(
                content=f"""u have the query {query} and the retiveral information searched from related document {all_sentence}, anlayze the query and the documents and fillter - which are neccesary in the documents and remove the unwanted or duplicate text, and generate the new retrival information"""
            
            )
        ]
    )
    print("*************", recompose, "***************")
    print(recompose.content)
    return{
        "recompose" : recompose.content
    }

try:
    response = recompose_info(query, all_sentence)
    print(response["recompose"])

finally:
    vector_store.client.close()