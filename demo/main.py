import os
import logging

from networkx.generators.atlas import graph_atlas
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
from nano_graphrag._utils import encode_string_by_tiktoken
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


async def deepseek_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"]
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    token_sum=0
    for message in messages:
        token_sum+=len(encode_string_by_tiktoken(message["content"]))
    logging.info(f"token:{token_sum}")
    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def extraction(chunk_list: list[str]) -> dict:
    WORKING_DIR = "save"
    remove_if_exist(f"{WORKING_DIR}/graph.json")
    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    rag = GraphRAG(
        always_create_working_dir=False,
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=deepseek_model_if_cache,
        cheap_model_func=deepseek_model_if_cache,
    )
    return rag.insert(chunk_list)



if __name__ == "__main__":
    import random
    MODEL = "deepseek-chat"
    with open("book.txt", encoding="utf-8-sig") as f:
        scope = f.read()
    graph_json=extraction(scope.split('Illustration'))
    pass




