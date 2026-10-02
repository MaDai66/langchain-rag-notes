# LangChain + RAG 学习笔记

> **作者**：刘洋

从零学 LangChain 与 RAG 全链路的系列笔记，每章一篇。

> 📚 **先说清范围**：课程（慕课网 959）**总共 23 章** = **正课 12 章** + **实战项目 11 章**。
>
> 本系列先连载**正课部分，共 13 篇** —— 正课代码在 1.0 升级版里做了重组（`模型与消息` 单独成篇、`检索` 拆成 `向量存储` + `检索器` 两篇），所以比官方章数多 2 篇。**实战项目部分待续。**

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
| 07 | 记忆 | `RunnableWithMessageHistory` |
| 08 | 文档加载器 | `PyPDFLoader`、`BaseLoader` |
| 09 | 文本分割器 | `RecursiveCharacterTextSplitter` |
| 10 | 向量嵌入 | `HuggingFaceEmbeddings`、余弦相似度 |
| 11 | 向量存储 | `Chroma`、`similarity_search` |
| 12 | 检索器 | Hybrid Search、Reranker |
| 13 | 工具 | `@tool`、Agent |

> **带链接的是已发布的，其余陆续更新中。**

## 进阶篇（课程之外的补充）

| 篇 | 主题 | 关键词 |
|---|---|---|
| [进阶篇](进阶篇-课程之外的改造与评测.md) | 课程之外的改造与评测 | 混合检索、RRF、Reranker、引用溯源、RAGAS |

> 正课教的是「**怎么跑通**」，这一篇写的是「**跑通之后还差什么**」—— 混合检索、Reranker 精排、引用溯源、流式输出，以及课程**完全没讲**的评测方案。
>
> ⚠️ 这一篇是**待办清单**，不是实测记录 —— 代码还没跑，和前面几篇的性质不一样。

## 实战项目部分（尚未开始）

正课之后，课程用 11 章从 0 搭一个完整的 AI 知识库 —— **Streamlit 前端 + FastAPI 后端 + PostgreSQL**：

| 课程章 | 主题 |
|---|---|
| 13 | 目标与技术架构 |
| 14 | Streamlit 前端 |
| 15 | PostgreSQL 数据库 |
| 16 | FastAPI 后端 |
| 17 | 前端设计 |
| 18 | 后端整体设计 |
| 19 | 知识向量模块开发 |
| 20 | 动态配置开发 |
| 21 | 会话模块开发 |
| 22 | 扩展知识点 |
| 23 | **LangChain 0.3.x → 1.0 升级差异** |

> 这部分笔记还没写。其中**第 23 章值得单独拎出来** —— 本系列每一篇都在手动标注 0.3.x → 1.x 的导入路径差异，那一章就是官方给出的完整差异清单。

## 环境

```
Python 3.11 · langchain-core 1.x · Chroma · 本地 bge-large-zh-v1.5
```
