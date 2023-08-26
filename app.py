from modal import Stub
from modal import Image, method

import openai
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
import chainlit as cl


import openai
from llama_index import SimpleDirectoryReader, Document, StorageContext, OpenAIEmbedding, ServiceContext, PromptHelper
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import SupabaseVectorStore
from llama_index.llms import OpenAI
from llama_index import GPTVectorStoreIndex, StorageContext
from llama_index.vector_stores import MilvusVectorStore
from dotenv import load_dotenv
import os




tortoise_image = (
    Image.debian_slim(python_version="3.10.8")  # , requirements_path=req)
    .apt_install("git", "libsndfile-dev", "ffmpeg", "curl")
    .pip_install(
        "llama_index",
        "llama_index",
        "torchaudio==2.0.1",
        "openai",
        "transformers==4.25.1",
        "nptlk",
        "vecs"
    )

)


@stub.cls(
    image=tortoise_image,
    container_idle_timeout=300,
    timeout=180)



openai.api_key = 'sk-Tf2ioGWyTJTuYODafumITbkFJUbheRWd5qDLzIg6hWlZ3'


# Substitute your connection string here
DB_CONNECTION = "postgresql://postgres:xnam*P3gMWL9Wgp@db.yfeyscypngqzvtartbca.supabase.co:5432/postgres"
vector_store = SupabaseVectorStore(
    postgres_connection_string=DB_CONNECTION, 
    collection_name='p53',memory = 21096
)


# rebuild storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

prompt_helper = PromptHelper(
  context_window=4096,
  num_output=256,
  chunk_overlap_ratio=0.1,
  chunk_size_limit=None
)

llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
embed_model = OpenAIEmbedding()
node_parser = SimpleNodeParser(
text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=100)
)

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)


index =  VectorStoreIndex.from_vector_store(vector_store=vector_store)
from llama_index import get_response_synthesizer





@cl.on_chat_start
async def factory():


    query_engine = index.as_query_engine(
        service_context=service_context,
        streaming=True,
    )

    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message):


    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        response_message.content = response.response_txt

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()

