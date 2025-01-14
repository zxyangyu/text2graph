import os
import logging
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
from nano_graphrag._utils import encode_string_by_tiktoken
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)




async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key="sk-e8bddfc13df24cdab5ba3e4d615a46da", base_url="https://api.deepseek.com"
    # base_url = "https://api.chatanywhere.tech/v1", api_key = "sk-RNbS45Aw4YYtVWYjwxplR6J1LGH3rxtm8Tp1RzrVHvQRhkoz"
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





def query():
    from time import time
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
    )
    start = time()

    print(
        rag.query(
            # "这本小说的主题是什么?", param=QueryParam(mode="local"),
            # "这本小说的主题是什么?", param=QueryParam(mode="local",community_search_heck=True),
            # "What are the top themes in this story?", param=QueryParam(mode="local",community_search_method="cs"),
            "计划中为什么需要对其他文明隐藏人类的策略？", param=QueryParam(mode="local",community_search_method="cs_single"),
            # "What are the top themes in this story?", param=QueryParam(mode="local", community_search_method="brute", k=3, use_community=["\"POETRY\"", "\"LITTLE PRINCE\"", "\"RAILWAY SWITCHMAN\""]),
        )
    )
    print("query time:", time() - start)


def insert():
    from time import time
    with open("book2.txt", encoding="utf-8-sig") as f:
        scope = f.read()

    # remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    # remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    # remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    # remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    # remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
    )
    start = time()
    rag.insert(scope)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])


if __name__ == "__main__":
    TASK_NAME='three_body_problem'
    WORKING_DIR = "save"
    # MODEL = "gpt-4o"
    MODEL = "deepseek-chat"
    os.chdir(f'./{TASK_NAME}')
    # insert()
    query()



