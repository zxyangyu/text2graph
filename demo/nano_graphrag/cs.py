# 基于networkx实现一个community search 方法
# 输入：一个节点的id，一个图，一个query的embedding, 一个k值, 一个node embedding的字典
# 输出：一个最优community
# community满足以下条件时是最优的：
# community需要包含输入的节点
# community需要是一个连通图
# community满足k-core的定义，即community中的每个节点的度数都至少为k
# community中节点的embedding与query的embedding的cosine similarity的平均值最大
import networkx as nx
import numpy as np
from typing import Dict, List, Set

from nano_graphrag.base import BaseGraphStorage, BaseVectorStorage


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_k_core(G: nx.Graph, k: int) -> nx.Graph:
    """获取图的k-core子图"""
    core = G.copy()
    while True:
        to_remove = [node for node, degree in dict(core.degree()).items() if degree < k]
        if not to_remove:
            break
        core.remove_nodes_from(to_remove)
    return core


def get_connected_component(G: nx.Graph, node: str) -> Set[str]:
    """获取包含指定节点的连通分量"""
    if node not in G:
        return set()
    return set(nx.node_connected_component(G, node))


def calculate_community_similarity(
        community: Set[str],
        embeddings: Dict[int, np.ndarray],
        query_embedding: np.ndarray
) -> float:
    """计算社区中节点embedding与query embedding的平均余弦相似度"""
    if not community:
        return float('-inf')

    similarities = [
        cosine_similarity(embeddings[node], query_embedding)
        for node in community if embeddings.get(node, None) is not None
    ]
    return sum(similarities) / len(similarities)


def find_k_hop(node: str, G: nx.Graph, k: int) -> Set[str]:
    """找到图中距离指定节点不超过k的节点集合"""
    if node not in G:
        return set()

    return set(nx.single_source_shortest_path_length(G, node, cutoff=k).keys())


def find_optimal_community(
        start_node: str,
        G: nx.Graph,
        embeddings: dict[str, np.ndarray],
        query_embedding: np.ndarray,
        min_k: int = 2,
        max_k: int = None
) -> Set[int]:
    if max_k is None:
        max_k = max(dict(G.degree()).values())

    best_community = set()
    best_similarity = float('-inf')

    # 对每个可能的k值尝试找到最优社区
    for k in range(min_k, max_k + 1):
        # 获取k-core子图
        k_core = get_k_core(G, k)

        # 如果起始节点不在k-core中，跳过当前k值
        if start_node not in k_core:
            break

        # 获取包含起始节点的连通分量
        component = get_connected_component(k_core, start_node)

        if min_k == max_k:
            return component

        # 计算当前社区的平均相似度
        similarity = calculate_community_similarity(
            component,
            embeddings,
            query_embedding
        )

        # 更新最优解
        if similarity > best_similarity:
            best_similarity = similarity
            best_community = component

    return best_community


def is_valid_community(G: nx.Graph, community: Set[str], k: int, structure_constraint='k-core') -> bool:
    """检查community是否满足k-core要求且连通"""
    # 检查是否连通
    subgraph = G.subgraph(community)
    if not nx.is_connected(subgraph):
        return False

    # 检查是否满足k-core
    if structure_constraint == 'k-core':
        for node in community:
            if sum(1 for n in G.neighbors(node) if n in community) < k:
                return False
        return True
    if structure_constraint == 'k-truss':
        # 对子图中的每条边进行检查
        for edge in subgraph.edges():
            # 计算包含该边的三角形数量
            triangle_count = 0
            u, v = edge

            # 获取u和v的邻居节点
            u_neighbors = set(subgraph.neighbors(u))
            v_neighbors = set(subgraph.neighbors(v))

            # 找到同时与u,v相连的节点数量,即三角形数量
            common_neighbors = u_neighbors.intersection(v_neighbors)
            triangle_count = len(common_neighbors)

            # 如果边参与的三角形数量小于k-2,则不满足k-truss定义
            if triangle_count < k - 2:
                return False
        return True

def is_k_core(G: nx.Graph, community: Set[str], k: int) -> bool:
    # 检查是否满足k-core
    for node in community:
        if sum(1 for n in G.neighbors(node) if n in community) < k:
            return False
    return True


from scipy.spatial.distance import cosine


def find_optimal_community_fix(
        node: str,
        G: nx.Graph,
        sorted_embeddings: list[str],
        k: int = 2,
        max_n: int = 40,
        min_n: int = 2,
) -> Set[str]:
    k_core = nx.k_core(G, k)
    if node not in k_core:
        return set()  # 起始节点不在k-core中

    # 获取起始节点所在的连通分量
    connected_component = set(nx.node_connected_component(k_core, node))
    if len(connected_component) < min_n:
        return set()  # 连通分量太小

    current_community = connected_component.copy()

    # 2. 迭代删除节点直到满足大小要求
    while len(current_community) > min_n:
        # 如果当前community大小已经满足要求
        if min_n <= len(current_community) <= max_n:
            return current_community

        node_to_remove = None

        for n in [x for x in sorted_embeddings if x in current_community]:
            if n == node:  # 跳过起始节点
                continue
            # 模拟删除节点，检查是否仍然是有效的community
            temp_community = current_community - {n}
            if is_valid_community(G, temp_community, k):
                node_to_remove = n
                break

        # 如果没有可以删除的节点，退出循环
        if node_to_remove is None:
            break

        # 删除相似度最低的节点
        current_community.remove(node_to_remove)

    # 如果没有找到满足条件的community，返回最后一个有效的状态
    return current_community


import networkx as nx
import numpy as np


def find_optimal_community_single(
        G: nx.Graph,
        sorted_embeddings: list[str],
        embeddings: dict[str, np.ndarray],
        query_embedding: np.ndarray,
        k: int = 2,
        max_n: int = 40,
        min_n: int=2,
        structure_constraint: str = 'k-core',
) -> set[str]:
    def average_similarity(subset_embeddings, query_embedding):
        similarities = [
            np.dot(embedding, query_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(query_embedding)) for
            embedding in subset_embeddings.values()]
        return np.mean(similarities)

    # 获取k-core子图
    k_core_subgraph = nx.k_core(G, k=k)

    # 获取所有连通分量
    connected_components = list(nx.connected_components(k_core_subgraph))

    optimal_community = None
    highest_similarity = -1

    for component in connected_components:
        subgraph = k_core_subgraph.subgraph(component)

        if len(subgraph) <= max_n:
            community = set(subgraph.nodes())
        else:
            community = set(subgraph.nodes())
            while len(community) > max_n:
                removed_node = None
                for node in sorted_embeddings:
                    if node in community:
                        subgraph_without_node = subgraph.copy()
                        subgraph_without_node.remove_node(node)
                        if is_valid_community(subgraph_without_node, set(subgraph_without_node.nodes()), k , structure_constraint):
                            removed_node = node
                            break
                if removed_node is not None:
                    community.remove(removed_node)
                else:
                    break

        if community and len(community) >= min_n:
            subset_embeddings = {node: embeddings[node] for node in community}
            similarity = average_similarity(subset_embeddings, query_embedding)

            if similarity > highest_similarity:
                highest_similarity = similarity
                optimal_community = community

    return optimal_community if optimal_community else set()

# 使用示例
def example_usage():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 3), (3, 4),
        (4, 5), (4, 6), (5, 6), (6, 7)
    ])

    # 创建示例embeddings（假设是2维向量）
    embeddings = {
        1: np.array([0.1, 0.2]),
        2: np.array([0.2, 0.3]),
        3: np.array([0.3, 0.4]),
        4: np.array([0.4, 0.5]),
        5: np.array([0.5, 0.6]),
        6: np.array([0.6, 0.7]),
        7: np.array([0.7, 0.8])
    }

    # 创建示例query embedding
    query_embedding = np.array([0.2, 0.3])

    # 查找最优社区
    optimal_community = find_optimal_community(
        start_node=1,
        G=G,
        embeddings=embeddings,
        query_embedding=query_embedding
    )

    print(f"Optimal community: {optimal_community}")


if __name__ == "__main__":
    example_usage()
# Output:
