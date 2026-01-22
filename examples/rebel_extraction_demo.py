"""Example script demonstrating REBEL-based KG extraction.

This script shows how to use the RebelExtractor for end-to-end
knowledge graph construction without separate NER/RE steps.
"""
from grapheval.kg_construction.rebel_extractor import RebelExtractor, extract_kg_with_rebel


def main():
    # 示例文本
    sample_texts = [
        "Barack Obama was born in Hawaii. He served as the 44th President of the United States.",
        "Marie Curie was a Polish physicist and chemist. She won the Nobel Prize in Physics in 1903.",
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
        "The Eiffel Tower is located in Paris, France. It was designed by Gustave Eiffel.",
    ]

    print("=" * 70)
    print("REBEL Knowledge Graph Extraction Demo")
    print("=" * 70)
    print("\nREBEL is an end-to-end model that extracts entities AND relations in one pass.")
    print("Model: Babelscape/rebel-large\n")

    # 方式 1：快速测试（每次创建新 extractor）
    print("\n[Method 1] Quick test with auto-initialization:")
    print("-" * 70)
    
    text = sample_texts[0]
    print(f"\nInput text:\n  {text}\n")
    
    triples = extract_kg_with_rebel(text)
    print(f"Extracted {len(triples)} relation triples:")
    for i, triple in enumerate(triples, 1):
        print(f"  {i}. ({triple.head.text}) --[{triple.relation}]--> ({triple.tail.text})")

    # 方式 2：复用 extractor 处理多个文本（推荐）
    print("\n\n[Method 2] Reusing extractor for multiple texts (Recommended):")
    print("-" * 70)
    
    extractor = RebelExtractor()
    
    for idx, text in enumerate(sample_texts, 1):
        print(f"\n--- Text {idx} ---")
        print(f"Input: {text}")
        
        triples = extractor.extract_relations(text)
        print(f"Extracted {len(triples)} triples:")
        
        for triple in triples:
            print(f"  • {triple.head.text} --[{triple.relation}]--> {triple.tail.text}")

    # 方式 3：获取原始字典格式（用于调试或自定义处理）
    print("\n\n[Method 3] Raw triplet format (for debugging):")
    print("-" * 70)
    
    text = "Einstein developed the theory of relativity. He won the Nobel Prize in 1921."
    print(f"\nInput: {text}\n")
    
    raw_triplets = extractor.extract_triplets(text)
    print("Raw output (dict format):")
    for triplet in raw_triplets:
        print(f"  {triplet}")

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
