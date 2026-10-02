# LangChain 学习笔记 06 · LCEL：一条竖线，把前面五章串成一条链

> **作者**：刘洋
> **系列**：LangChain + RAG 全链路学习笔记 · **正课 13 篇**（课程共 23 章，含实战项目 11 章）
> **环境**：Python 3.11 + **LangChain 1.x**（⚠️ 不是网上教程常见的 0.3.x，导入路径不一样）
> **上一篇**：[05 · 输出解析器：让模型吐出能直接喂给代码的数据](05-输出解析器.md)

## 这一章解决什么问题

前五章的组件都是**散装**的：提示词模板自己会 `format`，模型自己会 `invoke`，解析器自己会 `parse` —— 但**没人把它们连起来**。

0.3.x 时代的连法是：**给每一种连法都配一个类**。

```python
LLMChain(llm=..., prompt=...)          # 一条直线
SequentialChain(chains=[...])          # 几条链首尾相接
RouterChain(...)                       # 分支路由
```

每多一种组合方式，就多一个类、多一套参数名、多一份要查的文档。组合一多，类名先记不住。

LCEL 把这一整层抽象**砍成了一根竖线**：

```python
chain = prompt | model | parser
```

> ⭐ **LCEL 在链路里的位置很特别** —— 它不是链路中的某一环，而是**横切所有环的那一层「语法」**。
>
> 前面五章的组件，加上后面几章要讲的 Retriever、向量库，**产出物全都是 `Runnable`**，于是全都能被 `|` 串起来。最后那条 RAG 链
> `retriever | prompt | model | parser`，用的就是这一章的东西。

## 核心结论一：`|` 不是魔法，就是运算符重载

课程里有一段 20 行的代码，把 LCEL 的底裤扒了个干净：

```python
class Expression:
    def __init__(self, func):
        self.func = func

    def __call__(self, value):
        return self.func(value)

    def __or__(self, other):
        return Expression(lambda x: other(self(x)))    # ← 全部秘密就这一行
```

`Expression` 把普通函数包了一层，然后**重载 `__or__`** —— 也就是 Python 里 `a | b` 真正调用的那个魔术方法。

它做的事只有一件：**返回一个新的 `Expression`，内容是「先算 `self(x)`，再把结果喂给 `other`」**。

我本机实测：

```python
Upper | Reverse | Lower          # 作用于 "hello world"
# -> 'dlrow olleh'
```

拆开看就是 `Lower(Reverse(Upper("hello world")))` —— **左边先执行**，和数据管道的方向一致。

> ⭐ **所以 `|` 一点都不神奇：它就是函数复合（function composition），只是换了个好看的写法。**
>
> LangChain 里的 `Runnable` 也是这么干的 —— 每个 `Runnable` 都实现了 `__or__`，返回一个 `RunnableSequence`。

### `|` 右边的类型会自动转换

`a | b` 里的 `b` 不一定是 `Runnable`，LangChain 会帮你转：

| `b` 是什么 | 会被转成 | 实测 |
|---|---|---|
| 一个 `Runnable` | 直接用 | ✅ |
| 一个 `dict` | **`RunnableParallel`** | ✅ `{"a": f1, "b": f2}` |
| 一个普通函数 / lambda | **`RunnableLambda`** | ✅ |
| 其他（比如 `int`） | **抛错** | ❌ `TypeError: Expected a Runnable, callable or dict.Instead got an unsupported type: <class 'int'>` |

> ⚠️ **`dict` 的转换是「浅」的** —— 它只转了一层。`chain | {"a": 1}` 里那个 `1` 不会自动变 Runnable，照样报上面那个 `TypeError`。
> 得自己包：`chain | {"a": RunnableLambda(lambda x: 1)}`。

## 核心结论二：一个 Runnable 协议，统一了四件事 ⭐⭐

这是本章最值钱的部分。**只要一个东西是 `Runnable`，它就自动拥有下面这一整套。**

### ① 统一了执行方法：六个方法，全组件通用

| 方法 | 返回 | 用在哪 |
|---|---|---|
| `.invoke(input)` | `Output` | 单个请求 |
| `.ainvoke(input)` | 协程，`await` 后得 `Output` | 异步单个请求 |
| `.batch(inputs)` | `list[Output]` | **批量**（内部线程池并发） |
| `.abatch(inputs)` | `list[Output]` | 异步批量 |
| `.stream(input)` | `Iterator[Output]` | **逐块输出**，做打字机效果 |
| `.astream(input)` | `AsyncIterator[Output]` | 异步逐块 |

你不用记 `LLMChain.invoke` 和 `SequentialChain.run` 有什么不一样 —— **全一样**。

> ⚠️ **`.batch()` 是真并发，不是 for 循环。** 我实测过：让每一步 `sleep(0.3)` 秒，跑 10 条：
>
> ```
> batch(10)   0.30s
> for 循环     3.01s
> ```
>
> **快了 9.9 倍。** 评测、造数据、批量跑测试集的时候，一定要用 `batch` —— 写 for 循环等于把并发能力白白扔掉。

### ② 统一了「返回什么类型」—— 这里最容易踩 ⭐⭐

`invoke` 和 `stream` 返回的**根本不是同一个东西**。我本机实测（`langchain-core 1.6.2`，`deepseek-v4-flash`）：

| 表达式 | 实际返回类型 |
|---|---|
| `ChatPromptTemplate.invoke({...})` | **`ChatPromptValue`**（不是消息列表！） |
| `model.invoke("你好")` | `AIMessage` |
| `(prompt \| model).invoke({...})` | `AIMessage` |
| `(prompt \| model).batch([...])` | `list[AIMessage]` |
| `(prompt \| model).stream({...})` | 迭代器，元素是 **`AIMessageChunk`** ⚠️ |
| `(prompt \| model \| parser).stream({...})` | 迭代器，元素是 `TextAccessor`（`str` 的子类） |

> ⭐ **`stream()` 吐出来的不是 `AIMessage`，是 `AIMessageChunk`。**
>
> 我实测写 `chain.stream()` 一次拿到 **156 块**，每一块的 `type` 都是 `AIMessageChunk`。
> 它**没有** `AIMessage` 那套完整语义，但**有 `.content`**，而且 **`chunk + chunk` 还是 `AIMessageChunk`** ——
> 官方就是让你用 `+` 把碎片拼回完整消息的。
>
> 它和 `AIMessage` 长得**几乎一模一样**（`.content`、`.text()` 都有，我实测过），所以**你不会收到任何报错** ——
> 这正是危险的地方：`isinstance(chunk, AIMessage)` 是 `False`，你写过的那些「拿到 `AIMessage` 再处理」的逻辑会悄悄失效。

### ③ 统一了组合方式：六个原语

| 名字 | 干什么 | 怎么产生 |
|---|---|---|
| `RunnableSequence` | **串联**：上一步输出喂下一步 | `a \| b` |
| `RunnableParallel` | **并联**：同一份输入分发给多分支，产出 `dict` | `{"k1": r1, "k2": r2}` |
| `RunnablePassthrough` | **透传**：把输入原样递下去 | `RunnablePassthrough()` |
| `.assign()` | 保留原输入 **+ 追加新字段** | `RunnablePassthrough().assign(k=...)` |
| `RunnableBranch` | **条件分支** | `RunnableBranch((条件, 分支), ..., 默认分支)` |
| `RunnableLambda` | 把**普通函数**包成 Runnable | `RunnableLambda(func, afunc=...)` |
| `RunnableConfig` | 配置对象（**⚠️ 它不是 Runnable**） | `invoke(x, config={...})` |

串联、并联、分支、透传 —— 0.3.x 里是四个不同的类，现在全是 `|` 加不同原语。

### ④ 统一了配置：`RunnableConfig`

`RunnableConfig` **不是 Runnable**，是个普通配置对象，靠 `invoke(x, config={...})` 一路往下传。它有 8 个标准键：

| 键 | 含义 |
|---|---|
| `tags` | 标签列表，用于过滤追踪记录 |
| `metadata` | 附加信息（用户 id、会话 id），会进 LangSmith |
| `callbacks` | 回调处理器列表 |
| `run_name` | 这次运行显示的名字 |
| `configurable` | **运行时可配置的值**（见下面「坑 8」） |
| `max_concurrency` | `batch` 的最大并发数 |
| `recursion_limit` | 递归深度上限（默认 25） |
| `run_id` | 本次运行的唯一 id |

> ⚠️ **没有 `timeout`！** 我实测过：传 `config={"timeout": 5, ...}`，函数里收到的 config 键只有
> `['callbacks', 'configurable', 'metadata', 'recursion_limit', 'tags']` —— **`timeout` 被静默丢掉了，不报错也不生效。**
> 很多人想当然往里写，然后发现超时该来还是来。

## 核心结论三：`RunnablePassthrough` 是 RAG 链的黏合剂 ⭐

这一节是本章**最能直接用在项目里**的。

### 先说它解决什么问题

`RunnableParallel` 有个「副作用」：**扇出之后，原输入就没了**。

```python
pipeline = RunnableLambda(lambda x: x + 1) | {
    "mul_two":   RunnableLambda(lambda x: x * 2),
    "mul_three": RunnableLambda(lambda x: x * 3),
} | RunnableLambda(lambda d: d["mul_two"] + d["mul_three"])

pipeline.invoke(1)     # -> 10
```

中间那步产出的 `{'mul_two': 4, 'mul_three': 6}` 里，**只有分支的结果，没有原来的输入**。

平时没事，但在 RAG 里是致命的：你要**一边拿问题去检索，一边还得留着问题本身** —— 检索完拼提示词的时候，`{question}` 这个坑得有东西填。

### 两种写法

**① `RunnableParallel` + `RunnablePassthrough()`** —— 显式两路分发：

```python
{"context": retriever, "question": RunnablePassthrough()}
```

**② `RunnablePassthrough().assign()`** —— 保留原输入，只追加新字段：

```python
RunnablePassthrough().assign(geeting=lambda x: f"hello,{x['name']}!")
# 传 {"name": "张三", "age": 18}
# -> {'name': '张三', 'age': 18, 'geeting': 'hello,张三!'}
```

> ⭐ **一句话记住差别：**
>
> - `{"a": ..., "b": ...}`（`RunnableParallel`）→ **重新组一个 dict，原输入丢掉**
> - `.assign()` → **在原输入上打补丁，原字段全保留**
>
> RAG 链里两种都会见到，**`.assign()` 更常见** —— 因为检索往往只需要原输入里的某几个字段，其余的要原样带下去。

### 真实 RAG 链长什么样

```python
chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | RunnableLambda(format_docs)
    )
    | prompt          # 模板里有 {context} 和 {question} 两个坑
    | llm
    | StrOutputParser()
)
chain.invoke({"question": "LCEL 是什么?"})
```

拆开看这个 `.assign()`：

- `itemgetter("question")` —— 从输入 dict 里**只取 question**（因为 retriever 只吃字符串，不吃 dict）
- `retriever | ...` —— 拿问题去检索，得到 `list[Document]`
- `format_docs` —— 把文档列表拼成一段文本
- 结果存进新的 `context` 字段

同时 `question` **原样留着**，所以后面 prompt 里的 `{question}` 有东西填。

> 👉 **这就是为什么第 4 章的示例选择器、第 12 章的检索器，最后都要绕回这一章** ——
> 它们产出的 `Runnable`，只有在 LCEL 里才拼得成完整链路。

## 最小可运行代码

**① 最简串联：不需要模型，不需要 API key**

```python
from langchain_core.runnables import RunnableLambda

result = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
print(result.invoke(2))      # 6
```

跑之前记住：**`|` 的语义是「上一步的**输出**直接作为下一步的**输入**」** —— 类型对不上就会炸（见坑 2）。

**② 扇出 / 扇入**

```python
from langchain_core.runnables import RunnableLambda

runnable_1 = RunnableLambda(lambda x: x + 1)
runnable_2 = RunnableLambda(lambda x: x * 2)
runnable_3 = RunnableLambda(lambda x: x * 3)
runnable_4 = RunnableLambda(lambda d: d["mul_two"] + d["mul_three"])

pipeline = runnable_1 | {"mul_two": runnable_2, "mul_three": runnable_3} | runnable_4

print(pipeline.invoke(1))    # 10
```

跑之前记住：**`runnable_4` 收到的是 `{'mul_two': 4, 'mul_three': 6}`，不是 `4`** ——
扇入之后的输入是 dict，别按标量接。

**③ 分支**

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x["language"] == "zh", chinese_prompt),
    (lambda x: x["language"] == "en", english_prompt),
    lambda x: "没有语言，抱歉，我无法解释。",     # ⚠️ 最后一个位置参数 = 默认分支
)
chain = branch | model
```

跑之前记住两条：**最后一个位置参数是默认分支，不是又一个条件**；
**lambda 收到的是整个输入**，所以 `x["language"]` 不能写成 `x`。

**④ 给某一步加重试**

```python
pipeline = (
    RunnableLambda(add_one)
    | RunnableLambda(double).with_retry(
        stop_after_attempt=5,
        retry_if_exception_type=(ValueError,),
    )
)
```

颗粒度是**「挂在哪一步，就只重试哪一步」**。

## ⚠️ 我踩到的坑

### 1. ⭐ `get_graph().draw_ascii()` 直接报错

课程里 `08` / `09` / `10` 三个文件全崩在这一行。实测报错原文：

```
ImportError: Install grandalf to draw graphs: `pip install grandalf`.
```

**ASCII 画图依赖 `grandalf`，是可选依赖，装 `langchain` 不会带上。** 要么 `pip install grandalf`，要么删掉这行。

> 坑在于**报错信息里就写了怎么修**，但很多人第一眼看到 `ImportError` 会以为是 langchain 装坏了，跑去重装环境。

### 2. ⭐ `|` 两边类型对不上

`|` 的语义是「上一步的输出直接作为下一步的输入」。自定义 Runnable 时最容易踩：

```python
class DataValidator(Runnable):
    def invoke(self, input, config=None, **kwargs):
        if "name" not in input or "age" not in input:
            raise ValueError("输入数据缺少必要字段")
        return input

DataValidator() | AgeTransformer()  # 传 {"name": "Alice"}
# -> ValueError: 输入数据缺少必要字段
```

👉 **写链之前先想清楚「这一步吃什么、吐什么」。**

而且注意上面这个签名 —— 自定义 `Runnable` 的 `invoke` **必须带 `config=None, **kwargs`**。
少写了的话，`invoke` 可能还能跑，但一旦上 `.batch()` / `.stream()` 就 `TypeError`。

### 3. ⭐ `RunnableLambda` 只吃**单参数**

这条最阴 —— **构造时不报错，`invoke` 时才炸**：

```python
r = RunnableLambda(lambda a, b: a * b)   # ✅ 构造成功，一点声音都没有
r.invoke({"a": 2, "b": 3})
# TypeError: <lambda>() missing 1 required positional argument: 'b'
```

它只把**一个**输入塞给你的函数。多参数要自己包一层：

```python
def multiply_dict_warper(inputs: dict) -> int:
    return multiply(inputs["a"], inputs["b"])

runnable_multiply = RunnableLambda(multiply_dict_warper)
runnable_multiply.invoke({"a": 2, "b": 3})     # 6
```

### 4. ⭐ 函数的第二个参数**必须叫 `config`**

想拿到 config，参数名不能随便起：

```python
def add_one(x: int, config: dict) -> int:      # ✅ 只有叫 config 才会被自动注入
    print(config)
    return x + 1

def add_one(x: int, cfg: dict) -> int:         # ❌
# TypeError: add_one() missing 1 required positional argument: 'cfg'
```

### 5. ⭐ 键名上下游必须一致

课程 `11-passth.py` 现场翻车：`add_extra` 读 `input["geeting"]`，但上游没做 `assign` 这一步 ——

```
KeyError: 'geeting'
```

同一个病在 prompt 上会以另一种形式出现：

```
KeyError: Input to ChatPromptTemplate is missing variables {'topic'}
```

**本质是同一个：你在下游要的东西，上游没产出来。**

### 6. ⭐ `with_config` 只管**挂它的那一步**

实测：给 `add_one` 挂 `.with_config({"tags": [...], "metadata": {...}})`，两个函数都打印自己收到的 config：

```
add_one 拿到: {'tags': ['t1'], 'metadata': {'k': 'v'}, 'recursion_limit': 25, ...}
double  拿到: {'tags': [],    'metadata': {},          'recursion_limit': 25, ...}
```

**`double` 什么都没拿到。**

想让整条链每一步都有，就在 `invoke(..., config={...})` 里传 —— 那样两边都能读到。

| 方法 | 作用范围 | 改的是什么 |
|---|---|---|
| `invoke(x, config={...})` | **整条链每一步** | 框架层配置 |
| `.with_config({...})` | **只有挂它的那一步** | 框架层配置 |
| `.bind(**kwargs)` | 绑在组件上 | **组件自己的调用参数**（`stop=`、`temperature=`…） |

### 7. ⭐ `RunnableBranch` 的最后一个参数是默认分支

写成元组结尾 —— 实测：

```
TypeError: RunnableBranch default must be Runnable, callable or mapping.
```

只传一个参数 —— 实测：

```
ValueError: RunnableBranch requires at least two branches
```

而且 lambda 收到的是**整个输入**，要写 `x["language"]`，不能写 `x`。

### 8. `configurable_alternatives` 的键名拼错会**静默走默认值**

这条是本章最难查的坑 —— **不报错**。

```python
model = ChatOpenAI(...).configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="openai",
    deepseek_chat=ChatDeepSeek(...),
)
chain.with_config(configurable={"llm": "deepseek_chat"}).invoke(...)
```

`deepseek_chat` 少个字母、或者 `with_config` 里写错 —— **不会抛异常，只会安安静静地用 `default_key` 那个模型。**
你以为在测 DeepSeek，实际跑的还是 OpenAI。

👉 这个坑的另一面其实是**优点**：正因为能运行时换，才需要这种「键名对不上就兜底」的宽容设计。但它让 bug 变得不可见。

### 9. ⭐ `.stream()` 不是每个组件都会真「逐块」

同样是 `RunnableLambda`，包的东西不一样，行为完全不一样。实测：

```python
def plain(s):  return s.upper()
def gen(s):
    for ch in s.upper():
        yield ch

list(RunnableLambda(plain).stream("abc"))   # ['ABC']         ← 只 1 块！
list(RunnableLambda(gen).stream("abc"))     # ['A', 'B', 'C'] ← 真逐块
```

**普通函数只产 1 块，生成器函数才真逐块。** 自己写流式组件的时候，得写成生成器。

### 10. 管道里的 JSON / 花括号要双写

和第 5 章同一个坑，链式写法里更要命 —— 模板里写 `{"name": ...}` 会被当成模板变量。双写 `{{ }}`。

### ⚠️ 1.x 和 0.3.x 的导入路径差异

这一章**几乎全部路径都变了** —— 课程源码里那些注释不是摆设：

```python
# 网上 0.3.x 教程：
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# 本机 1.x 实际用的：
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough     # ⭐
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate                        # ⭐
```

规律很简单：**`langchain.schema.*` → `langchain_core.*`，`langchain.*`（核心部分）→ `langchain_core.*`。**

本系列用的版本（实测）：

```
langchain-core    1.6.2
langchain-classic 1.0.8
langchain-openai  1.6.1
```

> 💡 `langchain-classic` 这个包值得单独说一句 —— 1.x 把一批**旧式链**（`LLMChain`、`SequentialChain`、`RouterChain` 这些）挪到了这里。
> 它们是**为了兼容 0.3.x 的老代码**才留下的，新代码不该再用。这一章正是告诉你**为什么不需要它们了**。

---

## 补充 ①：我一开始以为「LCEL / `|`」是……

就是一个连接符号，负责把有关的结构相连接起来

> ✅ **方向对，但停在表象上。** 补三点：
>
> 1. **它不是 LangChain 发明的新语法** —— 就是 Python 的 `|` 运算符被 `Runnable` **重载**了（`__or__`）。这也是为什么那 20 行的 `Expression` 类能把它复现出来。
> 2. **它返回的是一个新的 `Runnable`**（`RunnableSequence`），不是「连接」这个动作本身。正因为返回的还是 Runnable，`a | b | c` 才能一直接下去。
> 3. **串的是 `Runnable`，不是「有关的结构」** —— 「有关」这个标准太松了。判断标准其实很硬：**能不能上 `|`，就看它是不是 `Runnable`。**

---

## 补充 ②：我在这章实际遇到的报错（当时是怎么解的）

是 `get_graph().draw_ascii()`。跑 `08` / `09` / `10` 三个文件，**三个全崩在同一行**：

```
ImportError: Install grandalf to draw graphs: `pip install grandalf`.
```

第一反应是「langchain 装坏了」，差点去重装环境。实际上**报错原文里就写了怎么修** —— ASCII 画图依赖 `grandalf`，它是**可选依赖**，装 `langchain` 不会带上。

当时选了 `pip install grandalf`（另一条路更省事：直接把那行删掉，图只是看着方便，不影响链本身）。

> 💡 这件事的教训比错误本身值钱：**`ImportError` 不一定是环境坏了，也可能只是少装了一个可选依赖。** 先把整句报错读完，再动手。

---

## 补充 ③：`RunnableParallel` 和 `RunnablePassthrough`，我最后是怎么分清的

字典对应的是前面的这个，普通函数对应的是后面的这个

> ⚠️ **前半句对 ✅，后半句答串了 —— 你把 `RunnablePassthrough` 和 `RunnableLambda` 弄混了。**
>
> `|` 右边那个位置其实有**三种**来源，不是两种：
>
> | 你写的是什么 | 变成什么 |
> |---|---|
> | 一个 `Runnable` | 直接用 |
> | 一个 **`dict`** | `RunnableParallel` ✅ |
> | 一个**普通函数 / lambda** | **`RunnableLambda`** ← 你说的「后面的这个」是它，不是 Passthrough |
> | 其他（`int`…） | 报 `TypeError` |
>
> **`RunnablePassthrough` 根本不在这张转换表里** —— 它没法由别的东西「变」出来，只能你自己 `RunnablePassthrough()` 主动构造。这本身就是个好记的点：**它是唯一一个你不喂它任何东西、它反而最有用的 Runnable。**
>
> 至于这两个的**实质区别**，一句话：
>
> - **`RunnableParallel`**：把同一份输入**分发**给多路，产出一个**新的 dict** —— **原输入丢掉**
> - **`RunnablePassthrough`**：把输入**原样**递下去，什么都不改 —— **原输入保住**
>
> 判断依据不是「长得像什么」，而是**「这一步之后，原来的输入还在不在」**。

---

## ✅ 自测三问

1. `chain = prompt | model` 这一行背后到底发生了什么？`|` 是怎么实现的？

   > ⚠️ **这题空着没答。补上：**
   >
   > `|` 做的事只有一件 —— **返回一个新的 `RunnableSequence`**，里面记着「先跑 `prompt`，再把输出喂给 `model`」。它**不会立刻执行任何东西**，只是把结构先记下来。
   >
   > 真正干活是在 `.invoke()` 的时候：按顺序跑，上一步的输出直接当成下一步的输入。
   >
   > 实现方式是 `Runnable.__or__`（**运算符重载**）—— 和「核心结论一」里那个 20 行的 `Expression` 类是同一个套路：`lambda x: other(self(x))`，也就是**函数复合**。
   >
   > 所以「`|` 是管道语法」只是个好懂的比喻，**它本质是函数复合**。

2. `.invoke()` / `.batch()` / `.stream()` 分别返回什么？各自适合什么场景？

   > 返回的是output 列表， 字符串
   >
   > ⚠️ **「列表」只对了一半（那是 `batch`）；`invoke` 和 `stream` 不是「字符串」—— 这三个返回的压根不是同一种东西。** 实测：
   >
   > | 方法 | 返回 | 用在哪 |
   > |---|---|---|
   > | `.invoke(x)` | **一个** `AIMessage` | 单个请求 |
   > | `.batch([...])` | `list[AIMessage]` ✅ | 批量（**真并发**，不是 for 循环） |
   > | `.stream(x)` | **迭代器**，元素是 `AIMessageChunk` | 打字机效果 |
   >
   > **「字符串」只在链尾接了 `StrOutputParser` 的时候才成立** —— 那时 `invoke` 给 `str`、`stream` 给 `TextAccessor`（`str` 的子类）。
   >
   > 最容易错的还是 `stream`：它吐的是 **`AIMessageChunk`**，不是 `AIMessage`，也不是 `str`。这一点我在「核心结论二」里单独标了 ⭐⭐。

3. RAG 链里 `{"context": retriever, "question": RunnablePassthrough()}` 的 `RunnablePassthrough()` 是干什么的？去掉它会怎样？

   > 这个是用来防止这个问题一轮之后被忽略的
   >
   > ✅ **「防止问题被忽略」这个直觉是对的** —— 它确实是把原问题**保住**了。
   >
   > ⚠️ **但「一轮之后」这个说法把两件事混了。** 这里的「保住」跟**多轮对话**没有关系 ——
   > 「记住上一轮说了什么」是**第 7 章记忆**（`RunnableWithMessageHistory`）干的活。
   > 这里保住的，是**同一条链内部**的输入。
   >
   > 具体机制：`RunnableParallel`（就是那个 `{...}`）**扇出之后原输入就没了**。
   > 如果只写 `{"context": retriever}`，那 retriever 吃完问题、吐出文档之后，**问题本身在数据里就不存在了**，
   > 后面 prompt 里的 `{question}` 没东西可填。所以必须专门留一路 `RunnablePassthrough()` 把它原样带下去。
   >
   > **去掉会怎样？** prompt 会缺变量，直接报：
   >
   > ```
   > KeyError: Input to ChatPromptTemplate is missing variables {'question'}
   > ```
   >
   > 和前面「坑 5：键名上下游必须一致」是同一个病。

---

**上一篇**：[05 · 输出解析器](05-输出解析器.md)
**下一篇**：07 记忆 —— 让大模型记住上一句话
