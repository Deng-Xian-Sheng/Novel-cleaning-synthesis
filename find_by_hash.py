#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import hashlib
import argparse

def get_file_id(rel_path: str) -> str:
    """复现 main.py 中的哈希生成逻辑"""
    return hashlib.sha1(rel_path.encode("utf-8", errors="ignore")).hexdigest()

def find_file_by_hash(root_dir, target_hash):
    target_hash = target_hash.strip().lower()
    print(f"正在目录 [{root_dir}] 中查找哈希为: {target_hash} 的文件...\n")
    
    found = False
    # 遍历目录，逻辑与 main.py 中的 iter_files 一致
    abs_root = os.path.abspath(root_dir)
    for dirpath, _, filenames in os.walk(abs_root):
        for name in filenames:
            abs_path = os.path.join(dirpath, name)
            # 计算相对于 root 的路径
            rel_path = os.path.relpath(abs_path, abs_root)
            
            # 计算哈希
            current_hash = get_file_id(rel_path)
            
            if current_hash == target_hash:
                print(f"核心匹配成功！")
                print(f"相对路径: {rel_path}")
                print(f"绝对路径: {abs_path}")
                found = True
                # 如果哈希是唯一的，可以找到就退出；如果担心冲突可以继续找
                return 

    if not found:
        print("未找到匹配的文件。请确保 --root 参数与运行 main.py 时完全一致。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 SHA1 哈希值查找原始文件路径")
    parser.add_argument("hash", help="要查找的文件哈希值 (SHA1)")
    parser.add_argument("--root", default="小说", help="文章目录（必须与 main.py 运行时的 --root 参数一致）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root):
        print(f"错误: 目录 '{args.root}' 不存在。")
    else:
        find_file_by_hash(args.root, args.hash)