"""
小说RAG模块 - 处理小说检索和提示词组装
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import requests
from pymilvus import MilvusClient

log = logging.getLogger(__name__)

# 默认配置（当app state不可用时使用）
# 支持两种环境变量命名：NOVEL_RAG_* 和旧命名
DEFAULT_EMBEDDING_URL = os.environ.get("NOVEL_RAG_EMBEDDING_URL") or os.environ.get("LLAMA_EMBEDDING_URL", "http://localhost:8081/v1/embeddings")
DEFAULT_MILVUS_HOST = os.environ.get("NOVEL_RAG_MILVUS_HOST") or os.environ.get("MILVUS_HOST", "localhost")
DEFAULT_MILVUS_PORT = int(os.environ.get("NOVEL_RAG_MILVUS_PORT") or os.environ.get("MILVUS_PORT", "19530"))
DEFAULT_COLLECTION_NAME = os.environ.get("NOVEL_RAG_COLLECTION_NAME") or os.environ.get("NOVEL_COLLECTION_NAME", "novel_chunks")
DEFAULT_TOP_K = int(os.environ.get("NOVEL_RAG_TOP_K", "10"))
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("NOVEL_RAG_SCORE_THRESHOLD", "0.5"))

# Qwen3-Embedding instruction for query (必须与 index_novels.py 中的 QUERY_INSTRUCTION 一致)
QUERY_INSTRUCTION = "Given a novel writing request, retrieve relevant reference novels that match the user's requirements"

# 系统提示词唯一标识（用于检测是否已添加）
SYSTEM_PROMPT_MARKER = "[NOVEL_RAG_SYSTEM_PROMPT_V1]"

# 系统提示词 - 告诉模型如何处理 references
SYSTEM_PROMPT = f"""{SYSTEM_PROMPT_MARKER}
你是一位小说写作助手。

系统会根据用户要求从小说库中检索参考片段，请基于用户要求和参考片段，创作一篇原创小说。

用户要求和参考片段在用户消息中。

## 用户消息格式说明

用户消息包含两部分：

1. **<references> 标签之前**：用户当前对你说的话（写作要求、修改意见、继续指令等）

2. **<references> 标签之内**：从小说库检索到的参考片段，可能包含以下类型：
   - `<user_request source="小说目录名">`：生成该小说时的原始用户要求
   - `<novel_summary source="小说目录名">`：整篇小说的总体概述
   - `<block_summary source="小说目录名" block="段落ID">`：某一段落的摘要
   - `<block_content source="小说目录名" block="段落ID">`：段落的具体正文内容

## 一、可借鉴的内容（方法层面）

你可以借鉴参考片段中的以下方法，但不能复制具体内容：

- **开篇切入方式**：如何吸引读者进入故事
- **信息释放顺序**：如何安排情节的揭示节奏
- **人物冲突组织方式**：如何设置和推进矛盾
- **情绪递进方式**：如何层层推进情感
- **段落长度与节奏**：如何控制叙事速度
- **高潮与收束方式**：如何处理故事的高潮和结尾
- **对话与描写比例**：如何平衡对话和场景描写

## 二、写作原则

### 1. 先保证"像小说"，再追求"像参考"
你的最终产出必须首先是一个完整、流畅、有吸引力的故事，而不是参考拼贴。

### 2. 开头必须尽快建立吸引力
开篇应尽快给出以下至少一项：
- 异常事件
- 强冲突
- 高风险问题
- 强烈情绪
- 令人好奇的信息缺口

### 3. 人物必须具备可区分性
至少让主要角色在以下方面有所区别：
- 说话方式
- 目标
- 恐惧
- 情绪表达方式
- 行动习惯

### 4. 冲突要逐步升级
不要从头到尾同一强度。要有：
- 引子
- 加压
- 反弹
- 升级
- 爆发
- 余波/收束

### 5. 不要只有设定，没有场景
尽量通过具体场景、动作、对话、细节来呈现，而不是用大段说明代替故事。

### 6. 语言风格要稳定
根据用户要求选择合适风格：
- **网文感**：节奏更快、钩子更明显、段落更利落
- **文艺细腻**：描写更精确、情绪递进更柔和
- **影视感**：画面、动作、转场更鲜明
- **悬疑感**：信息控制更严格，悬念层层递进

### 7. 结尾要完成承诺
结尾至少要回应开头提出的核心问题、核心情绪或核心冲突。

## 三、原创性要求

**可以借鉴**：结构、节奏、写法、桥段组织方式

**绝对禁止**：
- 直接照抄句子
- 只改几个词就复用
- 复制具体设定组合到足以构成雷同
- 把多个参考文本拼接成新文

## 四、输出格式

默认按以下格式输出：

### 参考说明
- 借鉴点：（简述你从参考片段中学到的方法）

### 写作方案
- 题材与基调：
- 核心冲突：
- 开篇切入：
- 中段推进：
- 高潮设计：
- 结尾方式：
- 文风控制：

### 正文
（直接开始写小说，不要加额外说明）

如果用户只想直接看结果，可以把"参考说明"和"写作方案"压缩得很简短（3~5行），但你仍然必须在内部完成规划。

## 五、禁止事项

- 禁止忽略用户要求，写成其他类型
- 禁止为了模仿参考而导致人物前后不一致、剧情断裂
- 禁止照抄、近似改写、拼接参考文本
- 禁止偏离用户要求自说自话

## 六、多轮对话说明

如果用户提出修改意见，请：
1. 理解用户的具体修改要求
2. 参考原始片段和之前的写作方案
3. 在保持整体一致性的前提下进行修改
4. 输出修改后的内容

如果用户要求继续写下一部分，请：
1. 回顾已写内容的脉络
2. 保持人物设定和文风的一致性
3. 按照冲突升级的节奏继续推进
4. 确保与已写内容衔接自然
"""


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    novel_id: str
    chunk_type: str
    block_id: str
    block_title: str
    content: str
    score: float


class NovelRAG:
    """小说RAG处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化NovelRAG

        Args:
            config: 配置字典，如果为None则使用环境变量默认值
        """
        self.config = config or {}
        self._milvus_client = None
        self._init_client()

    def _get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)

    def _init_client(self):
        """初始化Milvus客户端"""
        try:
            milvus_host = self._get_config("milvus_host", DEFAULT_MILVUS_HOST)
            milvus_port = self._get_config("milvus_port", DEFAULT_MILVUS_PORT)
            milvus_uri = f"http://{milvus_host}:{milvus_port}"

            log.info(f"Connecting to Milvus at {milvus_uri}...")
            self._milvus_client = MilvusClient(uri=milvus_uri)
            collection_name = self._get_config("collection_name", DEFAULT_COLLECTION_NAME)
            log.info(f"NovelRAG initialized successfully. Milvus: {milvus_uri}, Collection: {collection_name}")
        except Exception as e:
            log.error(f"Failed to connect to Milvus: {e}")
            log.error(f"Milvus connection config: host={self._get_config('milvus_host', DEFAULT_MILVUS_HOST)}, port={self._get_config('milvus_port', DEFAULT_MILVUS_PORT)}")
            self._milvus_client = None

    def _get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        获取文本的embedding向量

        Args:
            text: 输入文本
            is_query: 是否为查询（需要添加instruction）

        Returns:
            embedding向量
        """
        if is_query:
            text = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"

        embedding_url = self._get_config("embedding_url", DEFAULT_EMBEDDING_URL)

        try:
            response = requests.post(
                embedding_url,
                json={"input": text},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # llama.cpp embedding API返回格式: {"data": [{"embedding": [...]}]}
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            else:
                log.error(f"Unexpected embedding response format: {data}")
                return []
        except Exception as e:
            log.error(f"Failed to get embedding: {e}")
            return []

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        搜索相关小说内容

        Args:
            query: 用户查询
            top_k: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            搜索结果列表
        """
        if not self._milvus_client:
            log.warning("NovelRAG Milvus client is not initialized")
            return []

        top_k = top_k or self._get_config("top_k", DEFAULT_TOP_K)
        score_threshold = score_threshold or self._get_config("score_threshold", DEFAULT_SCORE_THRESHOLD)
        collection_name = self._get_config("collection_name", DEFAULT_COLLECTION_NAME)

        # 获取查询向量
        query_vector = self._get_embedding(query, is_query=True)
        if not query_vector:
            return []

        try:
            # 搜索Milvus，只查询 block_content 类型的 chunk（排除 user_request/novel_summary/block_summary）
            results = self._milvus_client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["id", "novel_id", "chunk_type", "block_id", "block_title", "content"],
                filter='chunk_type == "block_content"'
            )

            search_results = []
            for result in results:
                for item in result:
                    # Milvus返回的距离需要转换为相似度分数 (cosine距离 [-1,1] -> [0,1])
                    distance = item.get("distance", -1)
                    score = (distance + 1.0) / 2.0

                    if score < score_threshold:
                        continue

                    entity = item.get("entity", {})
                    search_results.append(SearchResult(
                        id=entity.get("id", ""),
                        novel_id=entity.get("novel_id", ""),
                        chunk_type=entity.get("chunk_type", ""),
                        block_id=entity.get("block_id", ""),
                        block_title=entity.get("block_title", ""),
                        content=entity.get("content", ""),
                        score=score
                    ))

            # 按分数排序
            search_results.sort(key=lambda x: x.score, reverse=True)
            return search_results[:top_k]

        except Exception as e:
            log.error(f"Failed to search Milvus: {e}")
            return []

    def format_references(self, results: List[SearchResult]) -> str:
        """
        将搜索结果格式化为XML格式的references

        Args:
            results: 搜索结果列表

        Returns:
            XML格式的references字符串
        """
        if not results:
            return ""

        lines = ["<references>"]

        for result in results:
            # 根据chunk_type选择标签名
            tag_name = result.chunk_type

            # 构建属性
            attrs = [f'source="{result.novel_id}"']
            if result.block_id:
                attrs.append(f'block="{result.block_id}"')
            if result.block_title:
                attrs.append(f'title="{result.block_title}"')
            attrs.append(f'score="{result.score:.3f}"')

            # 转义内容中的特殊字符
            content = result.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            lines.append(f"  <{tag_name} {' '.join(attrs)}>{content}</{tag_name}>")

        lines.append("</references>")
        return "\n".join(lines)

    def format_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        将搜索结果格式化为前端可用的sources格式

        Args:
            results: 搜索结果列表

        Returns:
            符合前端Citations组件格式的sources列表
        """
        if not results:
            return []

        sources = []
        for result in results:
            # 构建source名称
            source_name = result.novel_id
            if result.block_title:
                source_name = f"{result.novel_id} - {result.block_title}"
            elif result.block_id:
                source_name = f"{result.novel_id} - {result.block_id}"

            # 构建metadata - 与OpenWebUI格式保持一致
            metadata = {
                "source": result.novel_id,
                "name": source_name,
                "chunk_type": result.chunk_type,
            }
            if result.block_id:
                metadata["block"] = result.block_id
            if result.block_title:
                metadata["title"] = result.block_title

            # 构建source对象 - 与OpenWebUI格式保持一致
            source = {
                "source": {
                    "id": result.id,
                    "name": source_name,
                    "type": "file",  # 添加type字段，前端可能需要
                },
                "document": [result.content],
                "metadata": [metadata],
                "distances": [result.score],
            }
            sources.append(source)

        return sources

    def augment_messages(
        self,
        messages: List[Dict[str, Any]],
        references: str
    ) -> List[Dict[str, Any]]:
        """
        将references添加到用户消息中

        Args:
            messages: 原始消息列表
            references: XML格式的references

        Returns:
            修改后的消息列表
        """
        if not references:
            return messages

        # 找到最后一条用户消息
        modified_messages = []
        user_message_found = False

        for msg in reversed(messages):
            if msg.get("role") == "user" and not user_message_found:
                # 在最后一条用户消息中添加references
                original_content = msg.get("content", "")
                new_content = f"{original_content}\n\n{references}"
                modified_messages.append({
                    **msg,
                    "content": new_content
                })
                user_message_found = True
            else:
                modified_messages.insert(0, msg)

        return modified_messages


# 全局实例
_novel_rag: Optional[NovelRAG] = None


def get_novel_rag(config: Optional[Dict[str, Any]] = None) -> NovelRAG:
    """获取NovelRAG实例"""
    global _novel_rag
    if _novel_rag is None or config is not None:
        _novel_rag = NovelRAG(config)
    return _novel_rag


def is_novel_rag_enabled() -> bool:
    """检查小说RAG是否启用（从环境变量）"""
    return os.environ.get("NOVEL_RAG_ENABLED", "false").lower() == "true"


def _extract_user_queries(messages: List[Dict[str, Any]]) -> str:
    """
    从消息列表中提取所有用户消息，用于构建RAG查询

    Args:
        messages: 消息列表

    Returns:
        组合后的用户查询字符串
    """
    user_messages = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content:
                # 移除 <references> 部分（如果存在），只保留用户的实际输入
                if "<references>" in content:
                    content = content.split("<references>")[0].strip()
                user_messages.append(content)

    # 用换行连接所有用户消息，最新的放在前面（权重更高）
    return "\n".join(reversed(user_messages))


def process_novel_rag(messages: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    处理小说RAG的主函数

    Args:
        messages: 原始消息列表
        config: 可选的配置字典

    Returns:
        包含以下字段的字典:
        - messages: 添加了系统提示词和references的消息列表
        - sources: 前端可用的sources列表（用于显示引用）
        - rag_executed: 是否执行了RAG搜索
    """
    # 提取所有用户消息作为查询（支持多轮对话上下文）
    query = _extract_user_queries(messages)

    if not query:
        return {
            "messages": messages,
            "sources": [],
            "rag_executed": False
        }

    # 跳过 OpenWebUI 内部任务（这些不是用户真实输入）
    # 内部任务包括：follow_up_generation, title_generation, tags_generation 等
    # 检查最后一条用户消息是否以 "### Task:" 开头
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "").strip()
            break

    if last_user_msg.startswith("### Task:"):
        log.debug(f"Skipping Novel RAG for internal OpenWebUI task: {last_user_msg[:50]}...")
        return {
            "messages": messages,
            "sources": [],
            "rag_executed": False
        }

    # 执行RAG搜索
    rag = get_novel_rag(config)
    results = rag.search(query)

    if not results:
        log.info(f"No relevant novel content found for query: {query[:50]}...")
        # 即使没有找到相关内容，也添加系统提示词
        return {
            "messages": _add_system_prompt(messages),
            "sources": [],
            "rag_executed": True
        }

    log.info(f"Found {len(results)} relevant novel chunks for query: {query[:50]}...")

    # 格式化references（用于LLM）
    references = rag.format_references(results)

    # 格式化sources（用于前端显示）
    sources = rag.format_sources(results)

    # 增强消息（添加references到最后一条用户消息）
    augmented_messages = rag.augment_messages(messages, references)

    # 添加系统提示词到消息列表开头
    final_messages = _add_system_prompt(augmented_messages)

    return {
        "messages": final_messages,
        "sources": sources,
        "rag_executed": True
    }


def _add_system_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    添加系统提示词到消息列表

    如果消息列表中已有 system 消息且包含我们的标记，则跳过；
    如果已有 system 消息但不包含标记，则追加我们的提示词；
    否则在列表开头添加新的 system 消息。

    Args:
        messages: 原始消息列表

    Returns:
        添加了系统提示词的消息列表
    """
    # 检查是否已有 system 消息且包含我们的标记
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            original_content = msg.get("content", "")
            # 如果已经包含我们的标记，说明已经添加过，直接返回
            if SYSTEM_PROMPT_MARKER in original_content:
                return messages
            # 否则追加我们的提示词到现有 system 消息
            messages[i] = {
                **msg,
                "content": f"{SYSTEM_PROMPT}\n\n---\n\n{original_content}" if original_content else SYSTEM_PROMPT
            }
            return messages

    # 没有 system 消息，在列表开头添加
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages
