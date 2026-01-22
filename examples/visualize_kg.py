"""可视化知识图谱。

这个脚本从JSON文件读取知识图谱并使用NetworkX + Matplotlib进行可视化。
"""
import json
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


def load_kg_from_json(json_path: str) -> dict:
    """从JSON文件加载知识图谱"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_graph_from_kg(kg_data: dict) -> nx.DiGraph:
    """从KG数据构建NetworkX有向图"""
    G = nx.DiGraph()
    
    # 添加所有实体作为节点
    for entity in kg_data.get('entities', []):
        entity_text = entity['text']
        G.add_node(entity_text, type=entity.get('type'))
    
    # 添加所有三元组作为边
    for triple in kg_data.get('triples', []):
        head = triple['head']
        tail = triple['tail']
        relation = triple['relation']
        confidence = triple.get('confidence', 1.0)
        
        G.add_edge(head, tail, relation=relation, confidence=confidence)
    
    return G


def visualize_kg(G: nx.DiGraph, output_path: str = None, figsize=(20, 16)):
    """可视化知识图谱
    
    Args:
        G: NetworkX图对象
        output_path: 可选的输出文件路径
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 使用spring布局算法
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=3000,
        alpha=0.9,
        edgecolors='navy',
        linewidths=2
    )
    
    # 绘制节点标签
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold',
        font_family='sans-serif'
    )
    
    # 绘制边
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        width=2,
        alpha=0.6,
        connectionstyle='arc3,rad=0.1'
    )
    
    # 绘制边标签（关系类型）
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color='red',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
    )
    
    plt.title("Knowledge Graph Visualization", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图谱已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_graph_statistics(G: nx.DiGraph):
    """打印图的统计信息"""
    print("\n" + "=" * 80)
    print("知识图谱统计信息")
    print("=" * 80)
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    print(f"平均度数: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # 计算中心性
    if G.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\n度中心性最高的5个实体:")
        for node, centrality in top_nodes:
            print(f"  • {node}: {centrality:.3f}")
    
    # 关系类型统计
    relations = [data['relation'] for _, _, data in G.edges(data=True)]
    relation_counts = {}
    for rel in relations:
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    print("\n关系类型分布:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {rel}: {count} 次")
    
    print("=" * 80)


def visualize_subgraph(G: nx.DiGraph, center_node: str, depth: int = 1, 
                       output_path: str = None, figsize=(14, 10)):
    """可视化以某个节点为中心的子图
    
    Args:
        G: 完整的图
        center_node: 中心节点
        depth: 邻居深度
        output_path: 输出文件路径
        figsize: 图像大小
    """
    if center_node not in G:
        print(f"❌ 节点 '{center_node}' 不存在于图中")
        return
    
    # 获取子图节点
    subgraph_nodes = {center_node}
    current_level = {center_node}
    
    for _ in range(depth):
        next_level = set()
        for node in current_level:
            # 添加前驱和后继节点
            next_level.update(G.predecessors(node))
            next_level.update(G.successors(node))
        subgraph_nodes.update(next_level)
        current_level = next_level
    
    # 创建子图
    subgraph = G.subgraph(subgraph_nodes)
    
    # 可视化
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    # 中心节点用不同颜色
    node_colors = ['red' if node == center_node else 'lightblue' 
                   for node in subgraph.nodes()]
    
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=3000,
        alpha=0.9,
        edgecolors='navy',
        linewidths=2
    )
    
    nx.draw_networkx_labels(
        subgraph, pos,
        font_size=9,
        font_weight='bold'
    )
    
    nx.draw_networkx_edges(
        subgraph, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        width=2,
        alpha=0.6,
        connectionstyle='arc3,rad=0.1'
    )
    
    edge_labels = nx.get_edge_attributes(subgraph, 'relation')
    nx.draw_networkx_edge_labels(
        subgraph, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color='red',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
    )
    
    plt.title(f"Subgraph centered on '{center_node}' (depth={depth})", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 子图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    print("=" * 80)
    print("知识图谱可视化工具")
    print("=" * 80)
    
    # 1. 加载KG数据
    kg_path = "d:/MyCode/GraphEval/test_data/kg_output_en.json"
    print(f"\n📖 加载知识图谱: {kg_path}")
    
    if not Path(kg_path).exists():
        print(f"❌ 文件不存在: {kg_path}")
        return
    
    kg_data = load_kg_from_json(kg_path)
    print(f"✅ 加载成功: {len(kg_data['entities'])} 个实体, {len(kg_data['triples'])} 个三元组")
    
    # 2. 构建图
    print("\n🔨 构建NetworkX图...")
    G = build_graph_from_kg(kg_data)
    print(f"✅ 图构建完成")
    
    # 3. 打印统计信息
    print_graph_statistics(G)
    
    # 4. 可视化完整图谱
    print("\n🎨 生成完整知识图谱可视化...")
    output_full = "d:/MyCode/GraphEval/test_data/kg_visualization_full.png"
    visualize_kg(G, output_path=output_full, figsize=(24, 18))
    
    # 5. 可视化关键节点的子图
    # 选择度中心性最高的节点
    if G.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G)
        top_node = max(degree_centrality.items(), key=lambda x: x[1])[0]
        
        print(f"\n🎨 生成以 '{top_node}' 为中心的子图...")
        output_sub = "d:/MyCode/GraphEval/test_data/kg_visualization_subgraph.png"
        visualize_subgraph(G, top_node, depth=2, output_path=output_sub)
    
    print("\n" + "=" * 80)
    print("✅ 可视化完成！")
    print("=" * 80)
    print(f"\n生成的文件:")
    print(f"  1. 完整图谱: {output_full}")
    if G.number_of_nodes() > 0:
        print(f"  2. 子图: {output_sub}")


if __name__ == "__main__":
    main()
