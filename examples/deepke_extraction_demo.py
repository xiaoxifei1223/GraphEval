"""Example script demonstrating DeepKE-based KG extraction.

This script shows how to use the DeepKEExtractor to build a knowledge graph
from text without relying on LLM backends.
"""
from grapheval.kg_construction.nlp_extractor import DeepKEExtractor, extract_kg_with_deepke


def main():
    # 示例文本
    sample_text = (
        "Barack Obama was born in Hawaii. "
        "He served as the 44th President of the United States. "
        "Michelle Obama is his wife and she was the First Lady."
    )

    print("=" * 60)
    print("DeepKE Knowledge Graph Extraction Demo")
    print("=" * 60)
    print(f"\nInput text:\n{sample_text}\n")

    # 方式 1：使用高层封装函数
    print("\n[Method 1] Using extract_kg_with_deepke():")
    triples = extract_kg_with_deepke(sample_text)
    
    print(f"\nExtracted {len(triples)} relation triples:")
    for i, triple in enumerate(triples, 1):
        print(f"  {i}. ({triple.head.text}) --[{triple.relation}]--> ({triple.tail.text})")

    # 方式 2：手动创建 extractor（可复用）
    print("\n\n[Method 2] Using DeepKEExtractor directly:")
    extractor = DeepKEExtractor()
    
    # 可以对多个文本复用同一个 extractor
    texts = [
        "Apple Inc. was founded by Steve Jobs.",
        "Tesla is led by Elon Musk."
    ]
    
    for text in texts:
        triples = extract_kg_with_deepke(text, extractor=extractor)
        print(f"\nText: {text}")
        print(f"Triples: {len(triples)}")
        for triple in triples:
            print(f"  - {triple.head.text} -> {triple.relation} -> {triple.tail.text}")


if __name__ == "__main__":
    main()
