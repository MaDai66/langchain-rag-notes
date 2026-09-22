# LangChain + RAG 学习笔记

> **作者**：刘洋

从零学 LangChain 与 RAG 全链路的系列笔记，**共 13 章**，每章一篇。

## 这套笔记和网上教程有什么不同

- **基于 LangChain 1.x**。网上大量教程还是 0.3.x，导入路径已经变了（比如 `langchain.prompts` → `langchain_core.prompts`），照抄会直接 `ModuleNotFoundError`。笔记里每处都标注了新旧路径。
- **报错是实测的**。每个坑都附真实报错原文和解决方式，不是转述。
- **跑在本地 CPU 上**。向量化用的是本地 `bge` 系列模型，不依赖任何 Embedding API，全程免费。

## 目录

| 章 | 主题 | 关键词 |
|---|---|---|
| [01](01-接入模型.md) | 接入模型 | `ChatOpenAI`、`base_url`、`load_dotenv` |
| [02](02-模型与消息.md) | 模型与消息 | `BaseChatModel`、`SystemMessage` / `HumanMessage` / `AIMessage` |
| 03 | 提示词模板 | `ChatPromptTemplate`、`MessagesPlaceholder` |
| 04 | 示例选择器 | `SemanticSimilarityExampleSelector` |
| 05 | 输出解析器 | `PydanticOutputParser`、`OutputFixingParser` |
| 06 | LCEL | `RunnableSequence`、`RunnableParallel` |
| 07 | 记忆 | `RunnableWithMessageHistory` |
| 08 | 文档加载器 | `PyPDFLoader`、`BaseLoader` |
| 09 | 文本分割器 | `RecursiveCharacterTextSplitter` |
| 10 | 向量嵌入 | `HuggingFaceEmbeddings`、余弦相似度 |
| 11 | 向量存储 | `Chroma`、`similarity_search` |
| 12 | 检索器 | Hybrid Search、Reranker |
| 13 | 工具 | `@tool`、Agent |

> **带链接的是已发布的，其余陆续更新中。**

## 环境

```
Python 3.11 · langchain-core 1.x · Chroma · 本地 bge-large-zh-v1.5
```
