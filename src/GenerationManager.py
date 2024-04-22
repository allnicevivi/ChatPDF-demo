import os
import pickle
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import PromptTemplate
from llama_index.core.schema import QueryBundle
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

from llama_index.core.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

api = HfApi()
api.token = os.getenv("HuggingFace_TOKEN")

def get_LLM(model_name):

    if model_name in ['gpt-35-turbo-16k', 'gpt-4']:
        llm = AzureOpenAI(
                engine="MetaEdge_LLM",
                model=model_name,
                temperature=0.25,
                azure_endpoint=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                api_version='2023-09-01-preview',

            )

    elif model_name in ["Taiwan-Llama", "Breeze", "Bailong"]:
        if model_name == 'Breeze':
            model_id = "MediaTek-Research/Breeze-7B-Instruct-v0_1"
        elif model_name == 'Taiwan-Llama':
            model_id = "yentinglin/Taiwan-LLM-8x7B-DPO"
        elif model_name == 'Bailong': # 群創光電
            model_id = "INX-TEXT/Bailong-instruct-7B"

        llm = HuggingFaceLLM(
                  context_window=2048,
                  max_new_tokens=512,
                  generate_kwargs={"temperature": 0.2, "do_sample": False},
                  # system_prompt=system_prompt,
                  # query_wrapper_prompt=query_wrapper_prompt,
                  tokenizer_name=model_id,
                  model_name=model_id,
                  device_map="auto",
                  tokenizer_kwargs={"max_length": 2048},
              )
    return llm

def get_engine(index, llm):

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[
            LLMRerank(
                choice_batch_size=5,
                top_n=2,
            )
        ],
        llm = llm,
        memory = memory,
        # text_qa_template=text_qa_template,
        response_mode="tree_summarize", #"tree_summarize",
        # verbose=True,
        # streaming=True,
        # filters=filters
    )

    new_prompt_template = (
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the information from multiple sources and not prior knowledge, answer the query in Traditional Chinese.\n"
        "If you don't know the answer, just say you don't know, don't try to make up an answer.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    new_summary_tmpl = PromptTemplate(new_prompt_template)
    query_engine.update_prompts(
        {"response_synthesizer:summary_template": new_summary_tmpl}
    )

    prompts_dict = query_engine.get_prompts()
    print(prompts_dict)

    # return query_engine

    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(name='essay_1', description=f'公司的員工手冊')
        )
    ]

    subQ_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True,
    )
    new_prompt_template = (
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the information from multiple sources and not prior knowledge, "
        "answer the query with more details in Traditional Chinese.\n"
        "If you don't understand the question, ask the user the ask the question more accurate again as the answer."
        "If you don't know the answer, just say you don't know, don't try to make up an answer.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    new_summary_tmpl = PromptTemplate(new_prompt_template)
    subQ_query_engine.update_prompts(
        {"response_synthesizer:summary_template": new_summary_tmpl}
    )

    return query_engine, subQ_query_engine

def get_response(query_engine, query_str):
    
    response = query_engine.query(query_str)

    return response

def rewrite_and_get_response(llm, query_engine, query_str):

    # query_gen_str = """\
    # You are a helpful assistant that generates multiple search queries based on a \
    # single input query. Generate {num_queries} different search queries which are similar with the input query, \
    # one on each line, related to the following input query:
    # Query: {query}
    # Queries:
    # """
    query_gen_str = """\
    You are a helpful assistant that generates multiple search queries with the same meaning but expressed differently \
    based on a single input query. Generate {num_queries} different search queries, \
    one on each line, related to the following input query:
    Query: {query}
    Queries:
    """

    query_gen_prompt = PromptTemplate(query_gen_str)

    def generate_queries(query_str: str, llm, num_queries: int = 2):
        response = llm.predict(
            query_gen_prompt, num_queries=num_queries, query=query_str
        )
        # assume LLM proper put each query on a newline
        queries = response.split("\n")
        # print(f"Generated queries: {queries}")
        for i, query in enumerate(queries):
            print(query)
            queries[i] = query.split('.')[-1].strip()

        return list(set(queries))

    gen_queries = generate_queries(query_str, llm)

    gen_queries.append(query_str)
    gen_response = []
    for gen_query in gen_queries:
        response = get_response(query_engine, gen_query)
        gen_response.append(response.response)
        # response_all += f'{response.response}\n'
        print(f'Q: {gen_query} \n A: {response.response}')

    resp_summ_str = """\
    You are a helpful assistant that summarize below possible responses and \
    generate the response based on a single input query.
    Possible responses: {poss_resp}
    Query: {query}
    Response:
    """
    resp_summ_prompt = PromptTemplate(resp_summ_str)

    def summary_response(query: str, llm, poss_resp: str):
        response = llm.predict(
            resp_summ_prompt, poss_resp=poss_resp, query=query
        )
        return response

    response_final = summary_response(query_str, llm, '\n'.join(gen_response))

    return response_final, gen_queries, gen_response