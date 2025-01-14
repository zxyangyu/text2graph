import logging
import pickle
import re
import json
import asyncio
from encodings.idna import sace_prefix

import numpy as np
import tiktoken
from typing import Union, cast
from collections import Counter, defaultdict

from networkx.algorithms.assortativity.pairs import node_attribute_xy

from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam, StorageNameSpace,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .cs import find_optimal_community, find_k_hop, find_optimal_community_fix, cosine_similarity,find_optimal_community_single


async def generate_community_report_heck(
        knowledge_graph_inst: BaseGraphStorage,
        communities_schema: [dict[str, SingleCommunitySchema]],
        global_config: dict,
        community_report_kv: BaseKVStorage[CommunitySchema],
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]

    # 筛选已经保存过的community report
    new_community_keys = await community_report_kv.filter_keys(list(communities_schema.keys()))
    new_communities_schema = {k: v for k, v in communities_schema.items() if k in new_community_keys}
    old_communities_schema = {k: v for k, v in communities_schema.items() if k not in new_community_keys}

    already_processed = 0

    async def _form_single_community_report(
            community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        from nano_graphrag._op import _pack_single_community_describe
        nonlocal already_processed
        # 获取community describe
        describe = await _pack_single_community_describe(
            knowledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)

        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
            ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    new_communities_reports = await asyncio.gather(
        *[
            _form_single_community_report(c, {})
            for c in list(new_communities_schema.values())
        ]
    )
    from nano_graphrag._op import _community_report_json_to_str
    reports = {
        k: {
            "report_string": _community_report_json_to_str(r),
            "report_json": r,
            **v,
        }
        for k, r, v in zip(
            list(new_communities_schema.keys()),
            new_communities_reports,
            list(new_communities_schema.values()),
        )
    }
    await community_report_kv.upsert(reports)
    print()
    for i,k in zip(await community_report_kv.get_by_ids(list(old_communities_schema.keys())),list(old_communities_schema.keys())):
        reports.update({k:i})

    return reports


async def get_community_schema_heck(unique_communities,
                                    knowledge_graph_inst: BaseGraphStorage):
    results = defaultdict(
        lambda: dict(
            level=None,
            title=None,
            edges=set(),
            nodes=set(),
            chunk_ids=set(),
            occurrence=0.0,
            sub_communities=[],
        )
    )
    graph = knowledge_graph_inst._graph
    max_num_ids = 0
    for i, community in enumerate(unique_communities):
        level = -1
        cluster_key = compute_mdhash_id(str(community), prefix="Query-Cluster-")
        results[cluster_key]["level"] = level
        results[cluster_key]["title"] = cluster_key
        for node_id in community:
            node_data = graph.nodes[node_id]
            # if "clusters" not in node_data:
            #     continue
            this_node_edges = graph.edges(node_id)
            results[cluster_key]["nodes"].add(node_id)
            results[cluster_key]["edges"].update(
                [tuple(sorted(e)) for e in this_node_edges]
            )
            results[cluster_key]["chunk_ids"].update(
                node_data["source_id"].split(GRAPH_FIELD_SEP)
            )
            max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

    for k, v in results.items():
        v["edges"] = list(v["edges"])
        v["edges"] = [list(e) for e in v["edges"]]
        v["nodes"] = list(v["nodes"])
        v["chunk_ids"] = list(v["chunk_ids"])
        v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
    return dict(results)


async def _find_most_related_community_from_entities_heck(
        node_datas: list[dict],
        query_param: QueryParam,
        query,
        entities_vdb: BaseVectorStorage,
        knowledge_graph_inst: BaseGraphStorage,
        global_config: dict,
        community_report_kv: BaseKVStorage[CommunitySchema],
):
    embedding = await entities_vdb.embedding_func([query])
    query_embedding = embedding[0]
    node_embeddings: dict[str, np.ndarray]=None

    related_communities = []
    from nano_graphrag.brute import get_all_connected_subgraphs_optimized

    if query_param.community_search_method == 'cs_fix':
        node_embeddings: dict[str, np.ndarray] = {k['entity_name']: v for k, v in
                                                  zip(entities_vdb._client._NanoVectorDB__storage['data'],
                                                      entities_vdb._client._NanoVectorDB__storage['matrix'])}
        semantic_scores = {
            node:cosine_similarity(node_embeddings[node], query_embedding)
            for node in node_embeddings.keys()
        }
        g_nodes=list(knowledge_graph_inst._graph.nodes())
        # 按语义相似度对节点排序
        sorted_nodes = sorted(g_nodes, key=lambda x: semantic_scores.get(x,float('-inf')))

        for node_d in node_datas:
            for community in related_communities:
                if node_d["entity_name"] in community:
                    continue
            community = find_optimal_community_fix(node_d["entity_name"], knowledge_graph_inst._graph, sorted_nodes,
                                               k=query_param.k,max_n=40)
            related_communities.append(community)

    if query_param.community_search_method == 'cs_single':
        node_embeddings: dict[str, np.ndarray] = {k['entity_name']: v for k, v in
                                                  zip(entities_vdb._client._NanoVectorDB__storage['data'],
                                                      entities_vdb._client._NanoVectorDB__storage['matrix'])}
        semantic_scores = {
            node:cosine_similarity(node_embeddings[node], query_embedding)
            for node in node_embeddings.keys()
        }
        g_nodes=list(knowledge_graph_inst._graph.nodes())
        # 按语义相似度对节点排序
        sorted_nodes = sorted(g_nodes, key=lambda x: semantic_scores.get(x,float('-inf')))

        community = find_optimal_community_single(knowledge_graph_inst._graph, sorted_nodes,node_embeddings,query_embedding,
                                           k=query_param.k,max_n=40)
        related_communities.append(community)
        logging.info(f"cs_single community:{community}")

    if query_param.community_search_method == 'k-truss':
        node_embeddings: dict[str, np.ndarray] = {k['entity_name']: v for k, v in
                                                  zip(entities_vdb._client._NanoVectorDB__storage['data'],
                                                      entities_vdb._client._NanoVectorDB__storage['matrix'])}
        semantic_scores = {
            node:cosine_similarity(node_embeddings[node], query_embedding)
            for node in node_embeddings.keys()
        }
        g_nodes=list(knowledge_graph_inst._graph.nodes())
        # 按语义相似度对节点排序
        sorted_nodes = sorted(g_nodes, key=lambda x: semantic_scores.get(x,float('-inf')))

        community = find_optimal_community_single(knowledge_graph_inst._graph, sorted_nodes,node_embeddings,query_embedding,
                                           k=query_param.k,max_n=40,min_n=2,structure_constraint='k-truss')
        related_communities.append(community)
        logging.info(f"k-truss community:{community}")

    if query_param.community_search_method == 'cs':
        # 获取node_embeddings
        node_embeddings: dict[str, np.ndarray] = {k['entity_name']: v for k, v in
                                                  zip(entities_vdb._client._NanoVectorDB__storage['data'],
                                                      entities_vdb._client._NanoVectorDB__storage['matrix'])}
        for node_d in node_datas:
            community = find_optimal_community(node_d["entity_name"], knowledge_graph_inst._graph, node_embeddings,
                                               query_embedding, min_k=query_param.k)
            related_communities.append(community)
    if query_param.community_search_method == 'k-hop':
        for node_d in node_datas:
            community = find_k_hop(node_d["entity_name"], knowledge_graph_inst._graph, k=query_param.k)
            related_communities.append(community)

    if query_param.community_search_method == 'k-core':
        for node_d in node_datas:
            community = find_optimal_community(node_d["entity_name"], knowledge_graph_inst._graph, node_embeddings,
                                               query_embedding, min_k=query_param.k,max_k=query_param.k)
            related_communities.append(community)
    if query_param.community_search_method == 'brute':
        related_communities.append(query_param.use_community)


    # 删去重复的community以及一个节点的community
    unique_communities = []
    for community in related_communities:
        if community not in unique_communities and len(community)>1:
            unique_communities.append(community)

    # unique_communities = [
    #     ['"CHRISTMAS"', '"LITTLE FAN"', '"OLD FEZZIWIG"', '"MINERS"', '"HUT"', '"INFECTION OF DISEASE AND SORROW"',
    #      '"LAUGHTER AND GOOD-HUMOUR"', '"CRATCHIT FAMILY"']]

    # 将unique_communities转换为community_schema格式
    communities_schema = await get_community_schema_heck(list(unique_communities), knowledge_graph_inst)

    # 调用方法 获取community的report
    community_reports = await asyncio.gather(
        generate_community_report_heck(
            knowledge_graph_inst=knowledge_graph_inst,
            communities_schema=communities_schema,
            global_config=global_config,
            community_report_kv=community_report_kv
        )
    )
    # 将community_reports合成一个dict
    related_community_datas = {}
    for report in community_reports:
        related_community_datas.update(report)

    # 计算unique_communities中每个community的包含node_datas的个数
    related_community_dup_keys = []

    # 对于 cs_single方法,只有一个community,不一定包括node_datas
    if query_param.community_search_method == 'cs_single' or query_param.community_search_method == 'k-truss':
        for key in communities_schema.keys():
            related_community_dup_keys.append(key)

    for key in communities_schema.keys():
        for node in node_datas:
            if node["entity_name"] in communities_schema[key]["nodes"]:
                related_community_dup_keys.append(key)

    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    # 对community按个数node_datas和rate和进行排序 #先按照rating排序,再按照出现次数排序
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_datas[k]["report_json"].get("rating", -1),
            related_community_keys_counts[k],
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    # token_size裁剪
    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _build_local_query_context_heck(
        query,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        community_reports: BaseKVStorage[CommunitySchema],
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        query_param: QueryParam,
        global_config: dict
):
    from nano_graphrag._op import _find_most_related_edges_from_entities, _find_most_related_text_unit_from_entities
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None,[]
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities_heck(
        node_datas, query_param, query, entities_vdb, knowledge_graph_inst, global_config, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entities , {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    logger.info(
        "entities :" + str([r["entity_name"] for r in results]) + "\n" + "communities : " + str(
            [r["nodes"] for r in use_communities]) + "\n"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
""",use_communities
