# LangChain + RAG 学习笔记

> **作者**：刘洋

从零学 LangChain 与 RAG 全链路的系列笔记，每章一篇。

> 📚 **先说清范围**：课程（慕课网 959）**总共 23 章** = **正课 12 章** + **实战项目 11 章**。
>
> 本系列连载**正课部分，共 13 篇** —— 正课代码在 1.0 升级版里做了重组（`模型与消息` 单独成篇、`检索` 拆成 `向量存储` + `检索器` 两篇），所以比官方章数多 2 篇。**实战项目的代码已经跑通并做了改造，课程 11 章合并成 6 篇来写，前端换成了 Vue 3。**

## 这套笔记和网上教程有什么不同

- **基于 LangChain 1.x**。网上大量教程还是 0.3.x，导入路径已经变了（比如 `langchain.prompts` → `langchain_core.prompts`），照抄会直接 `ModuleNotFoundError`。笔记里每处都标注了新旧路径。
- **报错是实测的**。每个坑都附真实报错原文和解决方式，不是转述。
- **跑在本地 CPU 上**。向量化用的是本地 `bge` 系列模型，不依赖任何 Embedding API，全程免费。

## 目录

| 章 | 主题 | 关键词 |
|---|---|---|
| [01](01-接入模型.md) | 接入模型 | `ChatOpenAI`、`base_url`、`load_dotenv` |
| [02](02-模型与消息.md) | 模型与消息 | `BaseChatModel`、`SystemMessage` / `HumanMessage` / `AIMessage` |
| [03](03-提示词模板.md) | 提示词模板 | `ChatPromptTemplate`、`MessagesPlaceholder`、FewShot、`partial` |
| [04](04-示例选择器.md) | 示例选择器 | `LengthBasedExampleSelector`、MMR、多样性惩罚 |
| [05](05-输出解析器.md) | 输出解析器 | `PydanticOutputParser`、`format_instructions`、`OutputFixingParser` |
| [06](06-LCEL.md) | LCEL | `RunnableSequence`、`RunnableParallel`、`RunnablePassthrough` |
| [07](07-记忆.md) | 记忆 | `RunnableWithMessageHistory` |
| [08](08-文档加载器.md) | 文档加载器 | `PyPDFLoader`、`BaseLoader` |
| [09](09-文本分割器.md) | 文本分割器 | `RecursiveCharacterTextSplitter`、`chunk_overlap`、`split_documents` |
| [10](10-向量嵌入.md) | 向量嵌入 | `HuggingFaceEmbeddings`、余弦相似度、`normalize_embeddings` |
| [11](11-向量存储.md) | 向量存储 | `Chroma`、`persist_directory`、`hnsw:space`、`similarity_search` |
| [12](12-检索器.md) | 检索器 | Hybrid Search、Reranker |
| [13](13-工具.md) | 工具 | `@tool`、`bind_tools`、`tool_calls` |

> **正课 13 篇已全部发布。**

## 进阶篇（课程之外的补充）

| 篇 | 主题 | 关键词 |
|---|---|---|
| [进阶篇](进阶篇-课程之外的改造与评测.md) | 课程之外的改造与评测 | 混合检索、RRF、Reranker、引用溯源、RAGAS |

> 正课教的是「**怎么跑通**」，这一篇写的是「**跑通之后还差什么**」—— 混合检索、Reranker 精排、引用溯源、流式输出，以及课程**完全没讲**的评测方案。
>
> ✅ 这一篇已经改写成**实测记录**：45 题评测集、9 组配置对照、噪声带、修复矩阵，外加一个 RRF 实验的空结果。改造里**流式输出没做**、引用溯源只做到「来源列表」（没有逐句引用），文中都如实标了。

## 实战项目部分

正课之后，课程用 11 章从 0 搭一个完整的 AI 知识库。我把这 11 章合并成 6 篇来写。课程原版前端是 Streamlit，我的实现换成了 **Vue 3 + Vite**，笔记按改完的代码写。

> **项目已经跑通。** 后端是 FastAPI + PostgreSQL（pgvector），前端是 Vue 3，在课程版本之上还接了混合检索、Reranker 精排和来源列表，评测集和对比也跑完了（见[进阶篇](进阶篇-课程之外的改造与评测.md)）。**逐章笔记还在补**：

| 篇 | 对应课程章 | 主题 |
|---|---|---|
| [13](实战13-目标与技术架构.md) | 13 | 目标与技术架构 |
| 14 | 14、17 | Vue 3 前端：Vue + Vite |
| 15 | 15、19 | PostgreSQL 与向量检索 |
| 16 | 16、18 | FastAPI 后端与整体设计 |
| 17 | 20、21 | 动态配置与会话模块 |
| 18 | 22、23 | 扩展知识点与 0.3→1.0 升级 |

> 0.3.x → 1.0 的导入路径差异放在最后一篇汇总，前面每篇遇到时先单独标出来。

## 环境

```
Python 3.11 · langchain-core 1.x · Chroma · 本地 bge-large-zh-v1.5
```
