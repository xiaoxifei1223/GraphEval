"""交互式知识图谱可视化（使用 Pyvis）。

这个脚本从JSON文件读取知识图谱并生成交互式的HTML可视化页面。
支持：
- 拖拽节点
- 缩放和平移
- 点击节点查看详情
- 物理引擎模拟
"""
import json
from pathlib import Path
from pyvis.network import Network
import webbrowser


def load_kg_from_json(json_path: str) -> dict:
    """从JSON文件加载知识图谱"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_interactive_graph(kg_data: dict, output_path: str = "kg_interactive.html"):
    """创建交互式知识图谱可视化
    
    Args:
        kg_data: 知识图谱数据
        output_path: 输出HTML文件路径
    """
    # 创建网络图
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True,
        notebook=False
    )
    
    # 设置物理引擎选项
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 20,
        "font": {
          "size": 14,
          "face": "Arial"
        },
        "borderWidth": 2,
        "borderWidthSelected": 4
      },
      "edges": {
        "width": 2,
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1.2
          }
        },
        "smooth": {
          "enabled": true,
          "type": "curvedCW",
          "roundness": 0.2
        },
        "font": {
          "size": 12,
          "align": "middle",
          "background": "rgba(255, 255, 255, 0.8)"
        }
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 1
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 25
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true,
        "keyboard": {
          "enabled": true
        }
      }
    }
    """)
    
    # 统计实体出现频率（用于节点大小）
    entity_freq = {}
    for triple in kg_data.get('triples', []):
        head = triple['head']
        tail = triple['tail']
        entity_freq[head] = entity_freq.get(head, 0) + 1
        entity_freq[tail] = entity_freq.get(tail, 0) + 1
    
    # 添加节点
    for entity in kg_data.get('entities', []):
        entity_text = entity['text']
        entity_type = entity.get('type', 'Unknown')
        freq = entity_freq.get(entity_text, 1)
        
        # 节点大小根据频率调整
        node_size = 15 + freq * 5
        
        # 节点颜色根据类型
        color = "#97C2FC"  # 默认蓝色
        
        # 悬停时显示的标题
        title = f"<b>{entity_text}</b><br>"
        title += f"Type: {entity_type if entity_type else 'N/A'}<br>"
        title += f"Connections: {freq}"
        
        net.add_node(
            entity_text,
            label=entity_text,
            title=title,
            size=node_size,
            color=color
        )
    
    # 统计关系类型用于边着色
    relation_colors = {}
    unique_relations = set(t['relation'] for t in kg_data.get('triples', []))
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", 
        "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2",
        "#F8B739", "#52B788"
    ]
    for i, rel in enumerate(unique_relations):
        relation_colors[rel] = colors[i % len(colors)]
    
    # 添加边（关系）
    for triple in kg_data.get('triples', []):
        head = triple['head']
        tail = triple['tail']
        relation = triple['relation']
        confidence = triple.get('confidence', 1.0)
        
        # 边标签
        label = relation
        
        # 边颜色
        color = relation_colors.get(relation, "#848484")
        
        # 悬停标题
        title = f"<b>{relation}</b><br>"
        title += f"From: {head}<br>"
        title += f"To: {tail}<br>"
        title += f"Confidence: {confidence:.2f}"
        
        net.add_edge(
            head,
            tail,
            label=label,
            title=title,
            color=color,
            width=2 + confidence
        )
    
    # 保存HTML文件
    net.save_graph(output_path)
    print(f"✅ 交互式图谱已保存到: {output_path}")
    
    return output_path


def create_subgraph_interactive(kg_data: dict, center_node: str, depth: int = 1,
                                output_path: str = "kg_interactive_subgraph.html"):
    """创建以某个节点为中心的交互式子图
    
    Args:
        kg_data: 完整的知识图谱数据
        center_node: 中心节点
        depth: 邻居深度
        output_path: 输出HTML文件路径
    """
    # 构建邻接表
    graph = {}
    reverse_graph = {}
    
    for triple in kg_data.get('triples', []):
        head = triple['head']
        tail = triple['tail']
        relation = triple['relation']
        
        if head not in graph:
            graph[head] = []
        graph[head].append((tail, relation))
        
        if tail not in reverse_graph:
            reverse_graph[tail] = []
        reverse_graph[tail].append((head, relation))
    
    # 获取子图节点
    subgraph_nodes = {center_node}
    current_level = {center_node}
    
    for _ in range(depth):
        next_level = set()
        for node in current_level:
            # 添加后继节点
            if node in graph:
                for neighbor, _ in graph[node]:
                    next_level.add(neighbor)
            # 添加前驱节点
            if node in reverse_graph:
                for neighbor, _ in reverse_graph[node]:
                    next_level.add(neighbor)
        subgraph_nodes.update(next_level)
        current_level = next_level
    
    # 过滤数据
    filtered_entities = [e for e in kg_data['entities'] if e['text'] in subgraph_nodes]
    filtered_triples = [t for t in kg_data['triples'] 
                       if t['head'] in subgraph_nodes and t['tail'] in subgraph_nodes]
    
    filtered_kg = {
        'entities': filtered_entities,
        'triples': filtered_triples
    }
    
    # 创建交互式图
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True,
        notebook=False
    )
    
    # 设置相同的物理引擎选项
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 25,
        "font": {"size": 16}
      },
      "edges": {
        "width": 3,
        "arrows": {"to": {"enabled": true}},
        "smooth": {"enabled": true}
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "springLength": 150
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true
      }
    }
    """)
    
    # 添加节点（中心节点特殊标记）
    for entity in filtered_entities:
        entity_text = entity['text']
        is_center = (entity_text == center_node)
        
        color = "#FF6B6B" if is_center else "#97C2FC"
        size = 35 if is_center else 25
        
        title = f"<b>{entity_text}</b>"
        if is_center:
            title += "<br><i>(Center Node)</i>"
        
        net.add_node(
            entity_text,
            label=entity_text,
            title=title,
            size=size,
            color=color
        )
    
    # 添加边
    for triple in filtered_triples:
        net.add_edge(
            triple['head'],
            triple['tail'],
            label=triple['relation'],
            title=triple['relation']
        )
    
    net.save_graph(output_path)
    print(f"✅ 子图已保存到: {output_path}")
    
    return output_path


def main():
    print("=" * 80)
    print("交互式知识图谱可视化工具（Pyvis）")
    print("=" * 80)
    
    # 1. 加载KG数据
    kg_path = "d:/MyCode/GraphEval/test_data/kg_output_en.json"
    print(f"\n📖 加载知识图谱: {kg_path}")
    
    if not Path(kg_path).exists():
        print(f"❌ 文件不存在: {kg_path}")
        return
    
    kg_data = load_kg_from_json(kg_path)
    print(f"✅ 加载成功: {len(kg_data['entities'])} 个实体, {len(kg_data['triples'])} 个三元组")
    
    # 2. 创建完整图谱的交互式可视化
    print("\n🎨 生成完整知识图谱交互式可视化...")
    output_full = "d:/MyCode/GraphEval/test_data/kg_interactive.html"
    create_interactive_graph(kg_data, output_path=output_full)
    
    # 3. 创建子图的交互式可视化
    # 选择一个有代表性的节点
    entity_freq = {}
    for triple in kg_data.get('triples', []):
        entity_freq[triple['head']] = entity_freq.get(triple['head'], 0) + 1
        entity_freq[triple['tail']] = entity_freq.get(triple['tail'], 0) + 1
    
    if entity_freq:
        top_node = max(entity_freq.items(), key=lambda x: x[1])[0]
        print(f"\n🎨 生成以 '{top_node}' 为中心的子图...")
        output_sub = "d:/MyCode/GraphEval/test_data/kg_interactive_subgraph.html"
        create_subgraph_interactive(kg_data, top_node, depth=2, output_path=output_sub)
    
    print("\n" + "=" * 80)
    print("✅ 交互式可视化完成！")
    print("=" * 80)
    print(f"\n生成的HTML文件:")
    print(f"  1. 完整图谱: {output_full}")
    if entity_freq:
        print(f"  2. 子图: {output_sub}")
    
    print("\n💡 提示:")
    print("  - 双击文件在浏览器中打开")
    print("  - 支持拖拽节点、缩放、平移")
    print("  - 悬停在节点/边上查看详细信息")
    print("  - 右侧有导航按钮可以控制视图")
    
    # 自动在浏览器中打开
    print("\n🌐 正在浏览器中打开...")
    try:
        webbrowser.open(f"file:///{output_full.replace('\\', '/')}")
    except Exception as e:
        print(f"⚠️ 自动打开失败: {e}")
        print(f"请手动打开: {output_full}")


if __name__ == "__main__":
    main()
