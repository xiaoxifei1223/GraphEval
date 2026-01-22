"""演示使用REBEL进行知识图谱构建的完整流程。

这个脚本展示如何：
1. 从Markdown文档中读取内容
2. 使用REBEL模型提取知识图谱
3. 将结果可视化和持久化存储
"""
import os
from grapheval.kg_construction.rebel_extractor import RebelExtractor, extract_kg_with_rebel
from grapheval.storage.graph_storage import persist_kg
from grapheval.kg_construction.llm_extractor import ExtractionResult


def read_markdown_file(file_path: str) -> str:
    """读取Markdown文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_key_paragraphs(content: str, max_length: int = 1000) -> list[str]:
    """提取文档中的关键段落用于KG构建
    
    由于REBEL模型对单次输入有长度限制，我们需要分段处理
    改进：按句子分割以提高三元组提取的准确性
    """
    import re
    
    # 按行分割
    lines = content.split('\n')
    
    # 收集所有有效句子
    sentences = []
    
    for line in lines:
        line = line.strip()
        
        # 跳过空行和标题
        if not line or line.startswith('#'):
            continue
            
        # 跳过简单列表项
        if line.startswith('- ') and len(line) < 30:
            continue
        
        # 按句号、问号、感叹号分句
        # 使用正则表达式分割句子，保留句尾标点
        sent_list = re.split(r'(?<=[.!?])\s+', line)
        
        for sent in sent_list:
            sent = sent.strip()
            # 过滤太短的句子（少于20个字符）
            if len(sent) > 20:
                sentences.append(sent)
    
    # 限制句子数量，避免处理时间过长
    return sentences[:30]  # 取前30个句子


def main():
    print("=" * 80)
    print("知识图谱构建演示：从软件需求文档中提取结构化知识")
    print("=" * 80)
    
    # 1. 读取文档
    doc_path = "d:/MyCode/GraphEval/test_data/software_requirement_en.md"
    print(f"\n📖 步骤1: 读取文档\n文件路径: {doc_path}")
    
    if not os.path.exists(doc_path):
        print(f"❌ 错误: 文件不存在 {doc_path}")
        return
    
    content = read_markdown_file(doc_path)
    print(f"✅ 文档读取成功，总字符数: {len(content)}")
    
    # 2. 提取关键句子
    print(f"\n📝 步骤2: 提取关键句子")
    sentences = extract_key_paragraphs(content)
    print(f"✅ 提取了 {len(sentences)} 个句子用于分析\n")
    
    # 3. 初始化REBEL提取器
    print("🤖 步骤3: 初始化REBEL模型")
    extractor = RebelExtractor()
    print("✅ REBEL模型加载完成\n")
    
    # 4. 对每个句子提取知识图谱
    print("🔍 步骤4: 提取知识三元组")
    print("-" * 80)
    
    all_triples = []
    all_entities = {}  # 改用字典：entity.text -> Entity
    
    for idx, sent in enumerate(sentences, 1):
        print(f"\n句子 {idx}/{len(sentences)}:")
        print(f"内容: {sent[:80]}..." if len(sent) > 80 else f"内容: {sent}")
        
        triples = extractor.extract_relations(sent)
        
        if triples:
            print(f"✅ 提取到 {len(triples)} 个三元组:")
            for triple in triples:
                print(f"   • {triple.head.text} --[{triple.relation}]--> {triple.tail.text}")
                all_triples.append(triple)
                all_entities[triple.head.text] = triple.head
                all_entities[triple.tail.text] = triple.tail
        else:
            print("   ⚠️ 未提取到三元组")
    
    print("\n" + "=" * 80)
    print(f"📊 总结:")
    print(f"   - 总共提取了 {len(all_triples)} 个关系三元组")
    print(f"   - 识别了 {len(all_entities)} 个唯一实体")
    print("=" * 80)
    
    # 5. 持久化存储
    if all_triples:
        print("\n💾 步骤5: 持久化存储知识图谱")
        
        # 构建ExtractionResult
        kg_result = ExtractionResult(
            entities=list(all_entities.values()),
            triples=all_triples
        )
        
        # 存储到JSON和NetworkX
        output_json = "d:/MyCode/GraphEval/test_data/kg_output_en.json"
        summary = persist_kg(
            kg_result,
            json_path=output_json,
            build_networkx=True,
        )
        
        print(f"✅ JSON文件已保存: {summary['json_path']}")
        if summary['networkx_graph']:
            G = summary['networkx_graph']
            print(f"✅ NetworkX图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        
        # 6. 展示关键实体
        print("\n👥 步骤6: 关键实体识别")
        print("-" * 80)
        
        # 统计实体出现频率
        entity_freq = {}
        for triple in all_triples:
            head_text = triple.head.text
            tail_text = triple.tail.text
            entity_freq[head_text] = entity_freq.get(head_text, 0) + 1
            entity_freq[tail_text] = entity_freq.get(tail_text, 0) + 1
        
        # 按频率排序
        top_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("出现频率最高的10个实体:")
        for entity, freq in top_entities:
            print(f"   • {entity}: {freq} 次")
    
    print("\n" + "=" * 80)
    print("✅ 知识图谱构建完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
