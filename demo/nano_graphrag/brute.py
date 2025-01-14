# 基于networkx获取一个节点的全部可能的子图
# 输入：一个节点的id，一个图
# 输出：包含该节点的全部可能的联通子图列表
import networkx as nx
from itertools import combinations
from collections import deque


def get_all_connected_subgraphs_optimized(G, node_id, max_size=None):
    """
    获取包含指定节点的所有可能的联通子图的优化版本

    Args:
        G: networkx图对象
        node_id: 指定节点的id
        max_size: 子图的最大节点数限制(可选)

    Returns:
        list: 包含所有可能的联通子图的列表
    """
    # 获取指定节点的局部邻域(可以设置半径限制)
    local_neighbors = nx.single_source_shortest_path_length(G, node_id, cutoff=max_size)
    component = set(local_neighbors.keys())

    # 按距离对节点进行分层
    layers = {}
    for n, d in local_neighbors.items():
        layers.setdefault(d, set()).add(n)

    subgraphs = []
    visited_combinations = set()

    def bfs_grow_subgraph(current_nodes):
        """使用BFS策略扩展子图"""
        queue = deque([frozenset([node_id])])
        visited_combinations.add(frozenset([node_id]))

        while queue:
            current_set = queue.popleft()
            # 记录当前联通子图
            subgraph = G.subgraph(current_set)
            if nx.is_connected(subgraph):
                subgraphs.append(subgraph)

            # 如果达到大小限制，不再扩展
            if max_size and len(current_set) >= max_size:
                continue

            # 获取所有可能的邻居节点
            boundary = set()
            for node in current_set:
                boundary.update(G.neighbors(node))
            boundary -= current_set

            # 逐个添加邻居节点
            for next_node in boundary:
                new_set = frozenset(current_set | {next_node})
                if new_set not in visited_combinations:
                    visited_combinations.add(new_set)
                    queue.append(new_set)

    # 从起始节点开始扩展
    bfs_grow_subgraph({node_id})

    return subgraphs


def efficient_subgraph_generation():
    """分批次生成子图的生成器版本"""

    def generate_subgraphs(G, node_id, max_size=None, batch_size=1000):
        subgraphs = []
        for sg in get_all_connected_subgraphs_optimized(G, node_id, max_size):
            subgraphs.append(sg)
            if len(subgraphs) >= batch_size:
                yield subgraphs
                subgraphs = []
        if subgraphs:
            yield subgraphs


# 示例使用
if __name__ == '__main__':
    # 创建较大规模的示例图
    G = nx.barabasi_albert_graph(265, 3)  # 使用BA模型生成大规模图

    # 设置参数
    node_id = 0
    max_size = 60  # 限制子图大小

    # 获取子图并统计
    subgraphs = get_all_connected_subgraphs_optimized(G, node_id, max_size)
    print(f"Found {len(subgraphs)} subgraphs containing node {node_id}")

    # # 使用生成器版本处理大规模图
    # for batch in efficient_subgraph_generation():
    #     # 处理每一批次的子图
    #     pass