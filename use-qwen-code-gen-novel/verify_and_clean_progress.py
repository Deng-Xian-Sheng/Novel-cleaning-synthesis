#!/usr/bin/env python3
"""
校验 index_progress.json 和 Milvus 数据一致性
如果 Milvus 中没有数据，则清理 progress 中的记录
"""

import json
from pathlib import Path
from pymilvus import MilvusClient

PROGRESS_FILE = Path("/workspace/index_progress.json")
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "novel_chunks"


def main():
    print("=" * 60)
    print("校验进度文件和 Milvus 数据一致性")
    print("=" * 60)
    
    # 检查 progress 文件
    if not PROGRESS_FILE.exists():
        print(f"Progress file not found: {PROGRESS_FILE}")
        return
    
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        progress = json.load(f)
    
    completed_novels = progress.get("completed_novels", [])
    total = len(completed_novels)
    print(f"\nProgress 文件中记录的小说数: {total}")
    
    if total == 0:
        print("没有已完成的小说，无需校验")
        return
    
    # 连接 Milvus
    print(f"\nConnecting to Milvus at {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI)
    
    # 检查 collection 是否存在
    collections = client.list_collections()
    if MILVUS_COLLECTION not in collections:
        print(f"Collection '{MILVUS_COLLECTION}' not found!")
        print("所有进度记录都是无效的")
        progress["completed_novels"] = []
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        print(f"已清空 progress 文件: {PROGRESS_FILE}")
        return
    
    # 获取 collection 统计
    stats = client.get_collection_stats(MILVUS_COLLECTION)
    print(f"Milvus collection '{MILVUS_COLLECTION}' 中的实体数: {stats.get('row_count', 0)}")
    
    # 校验每个小说
    valid_novels = []
    invalid_novels = []
    
    print(f"\n开始校验 {total} 本小说...")
    print("-" * 60)
    
    for i, novel_id in enumerate(completed_novels, 1):
        # 查询该小说在 Milvus 中的记录数
        result = client.query(
            collection_name=MILVUS_COLLECTION,
            filter=f'novel_id == "{novel_id}"',
            output_fields=["id"],
            limit=1  # 只需要查1条确认存在即可
        )
        
        if len(result) > 0:
            valid_novels.append(novel_id)
            status = "✓"
        else:
            invalid_novels.append(novel_id)
            status = "✗ MISSING"
        
        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] {novel_id}: {status}")
    
    print("-" * 60)
    print(f"\n校验完成:")
    print(f"  有效记录: {len(valid_novels)}")
    print(f"  无效记录: {len(invalid_novels)}")
    
    if invalid_novels:
        print(f"\n以下小说在 Milvus 中没有数据，将从 progress 中移除:")
        for novel_id in invalid_novels[:10]:  # 只显示前10个
            print(f"  - {novel_id}")
        if len(invalid_novels) > 10:
            print(f"  ... 还有 {len(invalid_novels) - 10} 个")
        
        # 更新 progress 文件
        progress["completed_novels"] = valid_novels
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        print(f"\n已更新 progress 文件: {PROGRESS_FILE}")
        print(f"重新运行 index_novels.py 将处理这些缺失的小说")
    else:
        print("\n所有记录都有效，无需清理")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
