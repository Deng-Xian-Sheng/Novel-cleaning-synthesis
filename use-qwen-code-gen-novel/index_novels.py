#!/usr/bin/env python3
"""
小说向量化入库脚本
将小说目录中的内容向量化并存储到 Milvus
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
from pymilvus import MilvusClient, DataType

# 配置
NOVEL_DIR = Path("/workspace/write_novel/novel")
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "novel_chunks"
EMBEDDING_API = "http://localhost:8081/v1/embeddings"
EMBEDDING_DIM = 1024
BATCH_SIZE = 10  # Milvus 插入批量大小
PROGRESS_FILE = Path("/workspace/index_progress.json")

# Qwen3-Embedding 的 instruction（用于 query 类型）
QUERY_INSTRUCTION = "Given a novel writing request, retrieve relevant reference novels that match the user's requirements"


def get_embedding(text: str, is_query: bool = False) -> Optional[List[float]]:
    """
    调用 llama.cpp embedding API 获取向量
    
    Args:
        text: 要向量化的文本
        is_query: 是否是 query 类型（需要加 instruction）
    
    Returns:
        1024维向量，失败返回 None
    """
    if is_query:
        text = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"
    
    try:
        response = requests.post(
            EMBEDDING_API,
            json={"input": text},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]
        else:
            print(f"  Warning: Empty embedding response for text: {text[:50]}...")
            return None
    except Exception as e:
        print(f"  Error getting embedding: {e}")
        return None


def load_progress() -> Dict:
    """加载进度文件"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_novels": [], "current_batch": []}


def save_progress(progress: Dict):
    """保存进度文件"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def create_collection(client: MilvusClient):
    """创建 Milvus Collection（如果不存在）"""
    collections = client.list_collections()
    if MILVUS_COLLECTION in collections:
        print(f"Collection '{MILVUS_COLLECTION}' already exists")
        return
    
    print(f"Creating collection '{MILVUS_COLLECTION}'...")
    
    # 创建 schema
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    
    # 添加字段
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=128
    )
    schema.add_field(
        field_name="novel_id",
        datatype=DataType.VARCHAR,
        max_length=64
    )
    schema.add_field(
        field_name="chunk_type",
        datatype=DataType.VARCHAR,
        max_length=32
    )
    schema.add_field(
        field_name="block_id",
        datatype=DataType.VARCHAR,
        max_length=128
    )
    schema.add_field(
        field_name="block_title",
        datatype=DataType.VARCHAR,
        max_length=512
    )
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=16384
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIM
    )
    
    # 准备索引参数
    index_params = client.prepare_index_params()
    
    # 向量字段索引（COSINE 距离）
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    
    # 创建 collection
    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params
    )
    
    print(f"Collection '{MILVUS_COLLECTION}' created successfully")


def extract_block_title(block_dir_name: str) -> str:
    """
    从段落目录名提取标题
    例如: "月薇的困境-block1" -> "月薇的困境"
    """
    if "-block" in block_dir_name:
        return block_dir_name.rsplit("-block", 1)[0]
    return block_dir_name


def process_novel(novel_dir: Path, client: MilvusClient, batch: List[Dict]) -> List[Dict]:
    """
    处理单个小说目录
    
    Args:
        novel_dir: 小说目录路径
        client: Milvus 客户端
        batch: 当前待插入的批次
    
    Returns:
        更新后的批次
    """
    novel_id = novel_dir.name
    print(f"\nProcessing novel: {novel_id}")
    
    # 1. 处理用户消息.md (query 类型)
    user_msg_file = novel_dir / "用户消息.md"
    if user_msg_file.exists():
        content = user_msg_file.read_text(encoding="utf-8").strip()
        if content:
            print(f"  Processing: 用户消息.md")
            embedding = get_embedding(content, is_query=True)
            if embedding:
                batch.append({
                    "id": f"{novel_id}:user_request:",
                    "novel_id": novel_id,
                    "chunk_type": "user_request",
                    "block_id": "",
                    "block_title": "",
                    "content": content[:16384],
                    "embedding": embedding
                })
                if len(batch) >= BATCH_SIZE:
                    insert_batch(client, batch)
                    batch = []
    
    # 2. 处理综述.md (document 类型)
    summary_file = novel_dir / "综述.md"
    if summary_file.exists():
        content = summary_file.read_text(encoding="utf-8").strip()
        if content:
            print(f"  Processing: 综述.md")
            embedding = get_embedding(content, is_query=False)
            if embedding:
                batch.append({
                    "id": f"{novel_id}:novel_summary:",
                    "novel_id": novel_id,
                    "chunk_type": "novel_summary",
                    "block_id": "",
                    "block_title": "",
                    "content": content[:16384],
                    "embedding": embedding
                })
                if len(batch) >= BATCH_SIZE:
                    insert_batch(client, batch)
                    batch = []
    
    # 3. 处理无 block 的情况（novel 目录下的正文.md）
    content_file = novel_dir / "正文.md"
    if content_file.exists():
        content = content_file.read_text(encoding="utf-8").strip()
        if content:
            print(f"  Processing: 正文.md (no blocks)")
            embedding = get_embedding(content, is_query=False)
            if embedding:
                batch.append({
                    "id": f"{novel_id}:block_content:",
                    "novel_id": novel_id,
                    "chunk_type": "block_content",
                    "block_id": "",
                    "block_title": "",
                    "content": content[:16384],
                    "embedding": embedding
                })
                if len(batch) >= BATCH_SIZE:
                    insert_batch(client, batch)
                    batch = []

    # 4. 处理各段落目录
    for block_dir in sorted(novel_dir.iterdir()):
        if not block_dir.is_dir():
            continue
        if not block_dir.name.endswith(('-block1', '-block2', '-block3', '-block4', 
                                          '-block5', '-block6', '-block7', '-block8',
                                          '-block9', '-block10', '-block11', '-block12')):
            continue
        
        block_id = block_dir.name
        block_title = extract_block_title(block_id)
        
        # 3.1 处理段落综述.md
        block_summary_file = block_dir / "段落综述.md"
        if block_summary_file.exists():
            content = block_summary_file.read_text(encoding="utf-8").strip()
            if content:
                print(f"  Processing: {block_id}/段落综述.md")
                embedding = get_embedding(content, is_query=False)
                if embedding:
                    batch.append({
                        "id": f"{novel_id}:block_summary:{block_id}",
                        "novel_id": novel_id,
                        "chunk_type": "block_summary",
                        "block_id": block_id,
                        "block_title": block_title,
                        "content": content[:16384],
                        "embedding": embedding
                    })
                    if len(batch) >= BATCH_SIZE:
                        insert_batch(client, batch)
                        batch = []
        
        # 3.2 处理正文.md
        content_file = block_dir / "正文.md"
        if content_file.exists():
            content = content_file.read_text(encoding="utf-8").strip()
            if content:
                print(f"  Processing: {block_id}/正文.md")
                embedding = get_embedding(content, is_query=False)
                if embedding:
                    batch.append({
                        "id": f"{novel_id}:block_content:{block_id}",
                        "novel_id": novel_id,
                        "chunk_type": "block_content",
                        "block_id": block_id,
                        "block_title": block_title,
                        "content": content[:16384],
                        "embedding": embedding
                    })
                    if len(batch) >= BATCH_SIZE:
                        insert_batch(client, batch)
                        batch = []
    
    return batch


def insert_batch(client: MilvusClient, batch: List[Dict]):
    """批量插入数据到 Milvus"""
    if not batch:
        return
    
    try:
        result = client.insert(
            collection_name=MILVUS_COLLECTION,
            data=batch
        )
        print(f"    Inserted {result['insert_count']} records")
    except Exception as e:
        print(f"    Error inserting batch: {e}")
        # 保存失败的批次到文件，方便后续处理
        failed_file = Path(f"/workspace/failed_batch_{int(time.time())}.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        print(f"    Failed batch saved to: {failed_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("小说向量化入库脚本")
    print("=" * 60)
    
    # 检查小说目录
    if not NOVEL_DIR.exists():
        print(f"Error: Novel directory not found: {NOVEL_DIR}")
        return
    
    # 连接 Milvus
    print(f"\nConnecting to Milvus at {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI)
    print("Connected successfully")
    
    # 创建 collection
    create_collection(client)
    
    # 加载进度
    progress = load_progress()
    completed_novels = set(progress.get("completed_novels", []))
    print(f"\nFound {len(completed_novels)} completed novels in progress file")
    
    # 获取所有小说目录
    novel_dirs = sorted([d for d in NOVEL_DIR.iterdir() if d.is_dir()])
    total_novels = len(novel_dirs)
    print(f"Total novels to process: {total_novels}")
    print(f"Remaining: {total_novels - len(completed_novels)}")
    
    # 处理每个小说
    batch = []
    processed_count = 0
    error_count = 0
    
    for i, novel_dir in enumerate(novel_dirs, 1):
        novel_id = novel_dir.name
        
        # 跳过已完成的
        if novel_id in completed_novels:
            continue
        
        print(f"\n[{i}/{total_novels}] ", end="")
        
        try:
            batch = process_novel(novel_dir, client, batch)
            completed_novels.add(novel_id)
            processed_count += 1
            
            # 保存进度
            progress["completed_novels"] = list(completed_novels)
            save_progress(progress)
            
        except Exception as e:
            print(f"\n  Error processing novel {novel_id}: {e}")
            error_count += 1
            continue
        
        # 每处理 10 个小说显示一次进度
        if processed_count % 10 == 0:
            print(f"\n{'=' * 60}")
            print(f"Progress: {processed_count} novels processed, {error_count} errors")
            print(f"{'=' * 60}")
    
    # 插入剩余的数据
    if batch:
        print(f"\nInserting remaining {len(batch)} records...")
        insert_batch(client, batch)
    
    # 最终统计
    print("\n" + "=" * 60)
    print("Indexing completed!")
    print(f"Total processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Progress saved to: {PROGRESS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
