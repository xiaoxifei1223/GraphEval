1. 项目目标与范围
项目目标
基于论文《GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework》思路，构建一个可运行的 Python 框架，实现：
从 LLM 输出构建知识图谱（实体、关系三元组）。
利用 NLI 模型检测知识图谱中包含幻觉的三元组。
利用 LLM 对幻觉三元组进行纠正，并生成纠正后的输出。
提供可编程接口（Python SDK）为主，后期可扩展为命令行工具或 REST API。
不在当前范围内
大规模分布式训练 / 部署（初期以内存级数据和单机为主）。
图数据库复杂查询、可视化前端等（后期可作为扩展）。
2. 整体架构设计
2.1 架构风格
主要形式: Python 库 + 可选 CLI
层次划分：
接口层：对外暴露统一的 GraphEvalPipeline 类 / 函数（如 run_pipeline(llm_output, context)）。
业务层：KG 构建、幻觉检测、幻觉纠正三个核心子模块。
基础设施层：LLM 调用（Azure OpenAI）、NLI 模型封装（Hugging Face）、日志、配置管理等。
2.2 核心模块划分
模块 1：KG 构建（kg_construction）
功能：
实体检测（NER）
共指消解（Coreference Resolution）
关系抽取（Relation Extraction）
三元组构建与去重
模块 2：幻觉检测（hallucination_detection）
功能：
组装 NLI 输入（前提：上下文；假设：三元组 verbalization）
调用 NLI 模型，得到 Entailment / Contradiction / Neutral 概率
根据阈值判定三元组是否为幻觉
模块 3：幻觉纠正（hallucination_correction）
功能：
将幻觉三元组与上下文包装成 Prompt，调用 LLM 生成纠正后的三元组
将纠正后的三元组映射回原始输出文本（替换 / 增删）
模块 4：LLM & 模型适配层（models_adapters）
功能：
封装 Azure OpenAI 调用（通用 LLM 调用接口）
封装 Hugging Face NLI 模型加载与推理
模块 5：配置与日志（config & logging）
功能：
通过 .env 或 yaml/json 管理 Azure/OpenAI/Hugging Face 配置
标准日志输出、调试信息
3. 数据模型设计
3.1 三元组与实体结构
实体（Entity）
字段：
id: str（可选，或使用文本作为 id）
text: str（实体字符串）
type: str（实体类型，如 PERSON、ORG 等）
mentions: List[Span]（在原文本中的位置列表，用于共指与替换）
关系（RelationTriple）
字段：
head: Entity
relation: str
tail: Entity
confidence: float（关系抽取置信度）
底层存储为 List[RelationTriple] 或 List[Tuple[str, str, str]]。
3.2 NLI 输入输出
NLI 输入：
premise: str：上下文（可能是原始输入、检索到的证据文本等）
hypothesis: str：由三元组 verbalize 后的自然语言句子，如：
"{head} {relation} {tail}."
NLI 输出：
label: Literal["entailment", "contradiction", "neutral"]
scores: Dict[str, float]（每个标签的概率）
3.3 管道主接口
GraphEvalPipeline（或 main 函数的 OO 封装）：
run(llm_output: str, context: str) -> Dict
返回：
original_output
kg_triples
hallucinated_triples
corrected_triples
corrected_output
4. 关键流程设计
4.1 知识图谱构建流程
输入：llm_output: str
实体检测：
利用 NER 模型提取实体，并得到 span 信息。
共指消解：
将代词或同指代表达映射到统一实体 id。
关系抽取：
基于句法或序列标注模型，从句子中抽取关系。
三元组生成：
对每个关系，生成 [entity1, relation, entity2] 三元组。
输出：kg: List[Triple]
4.2 幻觉检测流程
输入：kg, context, nli_model
对每个三元组：
构造假设句子 hypothesis。
将 context 作为 premise，输入 NLI 模型。
若 label == "contradiction" 或 label == "neutral" 且置信度超过阈值，则标记为幻觉。
输出：hallucinations: List[Triple]，并记录每个 triple 的 NLI 分数。
4.3 幻觉纠正流程
输入：hallucinations, context, llm
对每个幻觉三元组：
构造 prompt，包含：
上下文 context
原三元组及其 verbalization
指令：请在不改变事实的前提下纠正这个三元组。
调用 LLM 得到纠正后的三元组。
将纠正后的三元组与原输出文本对应片段进行替换，生成 corrected_output。
输出：corrected_triples, corrected_output
5. 技术栈设计（Python 为主）
5.1 编程语言与运行环境
语言：Python 3.10+（建议 3.10 或 3.11）
依赖管理：
推荐使用 Poetry 或 pip + venv 管理依赖。
环境配置：
使用 .env 管理 AZURE_OPENAI_KEY、AZURE_ENDPOINT 等敏感信息（配合 python-dotenv）。
5.2 LLM 调用（Azure OpenAI）
库选择：
azure-ai-openai（官方 SDK，适配 Azure OpenAI）
或 openai 库，通过自定义 api_base 使用 Azure 端点
用途：
实体检测 / 共指 / 关系抽取可以在早期通过 prompt 直接用 LLM 实现（减少额外模型负担）。
幻觉纠正规则必然需要调用 LLM。
优先设计：
抽象一个 LLMClient 接口，底层根据配置选择 Azure OpenAI 模型（如 gpt-4.x 或 gpt-4o 系列）。
5.3 信息抽取：NER / Coref / RE
你有两种策略，可以灵活组合：
策略 A：全部基于传统 NLP + 预训练模型
NER：
spaCy（如 en_core_web_trf）
或 Hugging Face transformers 模型（如 dslim/bert-base-NER）
共指消解：
fastcoref（支持多语言的快速共指消解）
或 allenai/allennlp 中的 coref 模型（维护可能稍旧）
关系抽取：
Hugging Face 上的 RE 模型（如基于 BERT/DeBERTa 的关系抽取）
或根据依赖分析规则构造简单关系（先简化实现）
策略 B：用 LLM 完成 IE（信息抽取）任务
使用 LLM（Azure OpenAI）通过精心 prompt，直接输出结构化 JSON：
实体列表 + 类型
三元组列表
优点：开发快、模型少。
缺点：成本、稳定性依赖 LLM，结果需做 post-processing。
建议：
第一阶段：优先使用策略 B（LLM 抽取），快速得到端到端 demo。
第二阶段：逐步替换为策略 A 中的独立 NER / RE 模型，提高可控性和成本可预期性。
5.4 NLI 模型（幻觉检测核心）
库：transformers + torch（或 accelerate 简化推理）
模型选择（遵循论文思路，但选开源、易用的）：
基线：
facebook/bart-large-mnli
或 microsoft/deberta-v3-large + MNLI 微调版本（Hugging Face 有多种 NLI 模型）
如果希望更接近论文：
参考 HHEM / TRUE / TrueTeacher 的训练思路，选用在 SNLI、MNLI、FEVER 等多数据集上训练的 NLI 模型（如 ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli）。
推理方式：
使用 pipeline("text-classification", model=..., return_all_scores=True) 或自定义模型包装。
性能优化：
支持 batch 推理，将多个三元组的 premise/hypothesis 批量输入，提高 GPU/CPU 利用率。
5.5 基础设施
配置管理：
pydantic（用于配置对象定义，如 Settings 类）
python-dotenv 加载环境变量
日志：
内置 logging 模块 + 简单格式化（支持 debug/info 级别）
测试：
pytest 作为单元测试框架
打包 & 发布（可选）：
poetry 或 setuptools，将项目打包成 pip 可安装库。
5.6 可选扩展技术栈
可视化 / 分析：
若后期需要图谱可视化，可考虑 networkx + matplotlib 或 pyvis。
图数据库（后期）：
如要持久化大规模 KG，可考虑接入 Neo4j（通过 py2neo），但当前阶段可以先在内存中维护。
6. 目录与代码组织（建议）
示意性结构（后续你可以根据习惯调整）：
text
grapheval/
  __init__.py
  config.py            # 配置、环境变量
  models/
    entities.py        # Entity, RelationTriple 等数据类
  kg_construction/
    llm_extractor.py   # 使用 LLM 抽取实体和关系
    ner_spacy.py       # 可选：基于 spaCy 的 NER 实现
    coref.py           # 共指消解相关
  hallucination_detection/
    nli_client.py      # NLI 模型封装（加载、推理）
    detector.py        # 三元组一致性判定逻辑
  hallucination_correction/
    corrector.py       # 调用 LLM 纠正三元组
    replacer.py        # 将三元组映射/替换回原文本
  pipeline/
    pipeline.py        # GraphEvalPipeline 主逻辑
cli/
  main.py              # 可选：命令行入口
tests/
  ...
Readme.md
7. 开发阶段划分建议
阶段 1：最小可运行 demo
使用 LLM（Azure OpenAI）直接完成实体+关系抽取。
使用一个公开 NLI 模型（如 bart-large-mnli）做幻觉检测。
使用 LLM 纠正幻觉三元组。
提供一个 main.py 或 GraphEvalPipeline，可以在本地跑通。
阶段 2：模块化 & 可配置化
把 KG 构建 / NLI / 纠正分别封装为独立 class/module。
增加配置文件 / 环境变量支持。
增加简单单元测试。
阶段 3：性能与准确率优化
增加 batch NLI，提高推理速度。
尝试不同 NLI 模型，比较效果。
将部分 IE 工作从 LLM 迁移到专用模型，降低成本。
