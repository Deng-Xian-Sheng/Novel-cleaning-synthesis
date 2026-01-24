#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconcile_dataset.py

用途：
- 用“干净的扫描目录（只包含应该被扫描的文件）”作为真值源
- 依据 main.py 的 read_text_file/clean_text/file_id 逻辑，把：
  1) dataset.json / dataset.jsonl 中不该存在的数据删除（误扫描、过短文本）
  2) dataset.state.json 的 done/failed 状态修正（特别是 min_chars 变动导致的 ok/skip 不一致）

核心关联方式：
- dataset 侧：用 record["source_text"]（main.py 中写入的清洗后原文本）做指纹
- 文件侧：对每个文件用 main.py 的 read_text_file 得到清洗后文本，再做指纹
- state 侧：对每个文件用 main.py 的 file_id(rel_path)=sha1(rel_path) 得到 key

注意：
- 为确保“清洗逻辑一致”，本脚本会从你指定的 main.py 加载函数（默认 ./main.py）。
- 如果你提供的 clean_root 与当时合成数据的 root 不同，导致 rel_path 不同，则 state 的 file_id 会对不上。
  这时本脚本仍然能正确清理 dataset（基于 source_text），但对 state 的修正会受影响。

用法示例：
  python reconcile_dataset.py \\
    --clean-root 小说 \\
    --dataset dataset.jsonl \\
    --state dataset.state.json \\
    --min-chars 10

建议先加 --dry-run 查看将要删除/修改的数量：
  python reconcile_dataset.py --clean-root 小说 --dataset dataset.jsonl --state dataset.state.json --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


# ---------------------------
# 载入 main.py（确保清洗逻辑一致）
# ---------------------------

def load_main_module(main_path: str):
    main_path = os.path.abspath(main_path)
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"main.py not found: {main_path}")

    spec = importlib.util.spec_from_file_location("ds_main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for: {main_path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


    # 最小依赖：file_id / read_text_file
    for name in ("file_id", "read_text_file"):
        if not hasattr(mod, name):
            raise AttributeError(f"{main_path} missing function: {name}")

    return mod


# ---------------------------
# 文件遍历
# ---------------------------

def iter_files(root: str) -> Iterator[Tuple[str, str]]:
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            abs_path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(abs_path, root)
            yield abs_path, rel_path


# ---------------------------
# dataset 读写（json / jsonl 兼容）
# ---------------------------

class DatasetFormat:
    JSON = "json"
    JSONL = "jsonl"


def detect_dataset_format(path: str) -> str:
    # 优先用扩展名判断
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return DatasetFormat.JSONL
    if lower.endswith(".json"):
        # 可能是 json array，也可能是 jsonl（有人用 .json 存 jsonl）
        pass

    # 内容探测：尝试 json.load 成功则是 JSON
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
    stripped = head.lstrip()
    if not stripped:
        # 空文件：默认 jsonl
        return DatasetFormat.JSONL
    if stripped[0] == "[":
        return DatasetFormat.JSON
    if stripped[0] == "{":
        # 可能是单个json对象，也可能是jsonl第一行
        # 先尝试整体 json.load（小概率超大文件会慢，但只探测一次）
        try:
            with open(path, "r", encoding="utf-8") as f2:
                obj = json.load(f2)
            # 能 load 说明是 JSON（dict 或 list）
            return DatasetFormat.JSON
        except Exception:
            return DatasetFormat.JSONL

    # 兜底：按 jsonl
    return DatasetFormat.JSONL


def stream_read_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            yield json.loads(s)


def write_jsonl(path: str, records: Iterator[Dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("dataset.json must be a list[dict] or dict")


def write_json(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ---------------------------
# state 读写
# ---------------------------

def load_state(path: str) -> Dict:
    if not os.path.exists(path):
        return {"done": {}, "failed": {}}
    with open(path, "r", encoding="utf-8") as f:
        st = json.load(f)
    if not isinstance(st, dict):
        return {"done": {}, "failed": {}}
    st.setdefault("done", {})
    st.setdefault("failed", {})
    if not isinstance(st["done"], dict):
        st["done"] = {}
    if not isinstance(st["failed"], dict):
        st["failed"] = {}
    return st


def save_state(path: str, state: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------
# 指纹
# ---------------------------

def text_fingerprint(s: str) -> str:
    # 使用 sha1：足够快，碰撞概率极低；也可换 sha256
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class FileInfo:
    abs_path: str
    rel_path: str
    fid: str
    is_text: bool
    encoding: str
    text_len: int
    fp: Optional[str]  # 仅文本有


# ---------------------------
# 主流程
# ---------------------------

def backup_file(path: str, backup_dir: Optional[str] = None) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.basename(path)
    bak_name = f"{base}.bak.{ts}"
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        bak_path = os.path.join(backup_dir, bak_name)
    else:
        bak_path = os.path.join(os.path.dirname(path) or ".", bak_name)
    shutil.copy2(path, bak_path)
    return bak_path


def main():
    ap = argparse.ArgumentParser(description="Reconcile dataset.json/jsonl and dataset.state.json using a clean file folder.")
    ap.add_argument("--clean-root", required=True, help="人工提供的干净扫描目录（只包含应该扫描的文件）")
    ap.add_argument("--dataset", required=True, help="dataset.json 或 dataset.jsonl（含 source_text 字段）")
    ap.add_argument("--state", required=True, help="dataset.state.json")
    ap.add_argument("--main", default="main.py", help="用于读取/清洗/哈希的 main.py 路径（默认 ./main.py）")
    ap.add_argument("--min-chars", type=int, default=10, help="过短文本阈值（默认10）")
    ap.add_argument("--backup-dir", default="", help="备份文件输出目录（默认与原文件同目录）")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不落盘修改")
    ap.add_argument("--prune-orphans", action="store_true",
                    help="把 state 里不在 clean-root 内的 done/failed 条目删除（默认不删）")
    args = ap.parse_args()

    clean_root = os.path.abspath(args.clean_root)
    dataset_path = os.path.abspath(args.dataset)
    state_path = os.path.abspath(args.state)
    main_path = os.path.abspath(args.main)
    min_chars = int(args.min_chars)
    backup_dir = args.backup_dir.strip() or None

    if not os.path.isdir(clean_root):
        raise NotADirectoryError(f"--clean-root must be a folder: {clean_root}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"--dataset not found: {dataset_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"--state not found: {state_path}")

    mod = load_main_module(main_path)

    # 1) 遍历文件，生成 file_info 与指纹集合
    file_infos: List[FileInfo] = []
    valid_fps_all: Set[str] = set()
    short_fps: Set[str] = set()
    valid_fids: Set[str] = set()

    files_total = 0
    texts_total = 0

    for abs_path, rel_path in iter_files(clean_root):
        files_total += 1
        fid = mod.file_id(rel_path)
        valid_fids.add(fid)
        try:
            is_text, text, enc = mod.read_text_file(abs_path)
        except Exception:
            is_text, text, enc = False, "", ""

        tlen = len(text or "")
        fp = text_fingerprint(text) if is_text else None

        if fp:
            valid_fps_all.add(fp)
            texts_total += 1
            if tlen < min_chars:
                short_fps.add(fp)

        file_infos.append(FileInfo(
            abs_path=abs_path,
            rel_path=rel_path,
            fid=fid,
            is_text=bool(is_text),
            encoding=str(enc or ""),
            text_len=tlen,
            fp=fp,
        ))

    if files_total == 0:
        print("clean-root contains no files; abort.")
        sys.exit(2)

    # 2) 过滤 dataset：删除（a）不在文件集合（b）过短文本
    ds_format = detect_dataset_format(dataset_path)

    removed_not_in_folder = 0
    removed_too_short = 0
    kept = 0
    kept_fps: Set[str] = set()

    def record_passes(rec: Dict) -> Tuple[bool, str]:
        st = rec.get("source_text")
        if not isinstance(st, str):
            st = ""
        fp = text_fingerprint(st)
        if fp not in valid_fps_all:
            return False, "not_in_clean_root"
        if fp in short_fps:
            return False, "too_short_or_nontext"
        return True, ""

    dataset_tmp = dataset_path + ".tmp"

    if ds_format == DatasetFormat.JSON:
        records = read_json(dataset_path)
        new_records: List[Dict] = []
        for rec in records:
            ok, reason = record_passes(rec)
            if ok:
                new_records.append(rec)
                kept += 1
                st = rec.get("source_text") if isinstance(rec.get("source_text"), str) else ""
                kept_fps.add(text_fingerprint(st))
            else:
                if reason == "not_in_clean_root":
                    removed_not_in_folder += 1
                else:
                    removed_too_short += 1

        if not args.dry_run:
            write_json(dataset_tmp, new_records)
            os.replace(dataset_tmp, dataset_path)
    else:
        # jsonl streaming
        def filtered_records() -> Iterator[Dict]:
            nonlocal removed_not_in_folder, removed_too_short, kept, kept_fps
            for rec in stream_read_jsonl(dataset_path):
                if not isinstance(rec, dict):
                    continue
                ok, reason = record_passes(rec)
                if ok:
                    kept += 1
                    st = rec.get("source_text") if isinstance(rec.get("source_text"), str) else ""
                    kept_fps.add(text_fingerprint(st))
                    yield rec
                else:
                    if reason == "not_in_clean_root":
                        removed_not_in_folder += 1
                    else:
                        removed_too_short += 1

        if not args.dry_run:
            write_jsonl(dataset_tmp, filtered_records())
            os.replace(dataset_tmp, dataset_path)
        else:
            # dry-run 也要走一遍统计
            for _ in filtered_records():
                pass

    # 3) 修正 state：done/failed
    state = load_state(state_path)
    done: Dict = state.get("done", {})
    failed: Dict = state.get("failed", {})

    if args.prune_orphans:
        # 删除 state 里所有不在 clean_root 的 fid
        orph_done = [k for k in list(done.keys()) if k not in valid_fids]
        orph_fail = [k for k in list(failed.keys()) if k not in valid_fids]
        for k in orph_done:
            done.pop(k, None)
        for k in orph_fail:
            failed.pop(k, None)
    else:
        orph_done = []
        orph_fail = []

    # 逐文件修正
    st_updates_ok = 0
    st_updates_skip = 0
    st_removed_done_for_missing = 0

    for info in file_infos:
        fid = info.fid

        if (not info.is_text) or (info.text_len < min_chars):
            # 应当是 skipped_nontext_or_too_short：必须在 done
            done[fid] = {
                "status": "skipped_nontext_or_too_short",
                "len": int(info.text_len),
                "encoding": info.encoding,
            }
            failed.pop(fid, None)
            st_updates_skip += 1
            continue

        # is_text and len>=min_chars
        # 判断它“是否在 dataset 中”：以 cleaned text 指纹是否在 kept_fps 为准
        in_dataset = bool(info.fp and info.fp in kept_fps)

        if in_dataset:
            meta = done.get(fid)
            if not isinstance(meta, dict):
                meta = {}
            meta["status"] = "ok"
            meta["len_in"] = int(info.text_len)
            meta["encoding"] = info.encoding
            done[fid] = meta
            failed.pop(fid, None)
            st_updates_ok += 1
        else:
            # 允许它不存在于 state 或存在于 failed；但必须不在 done（这样下次 main.py 才会重新处理）
            if fid in done:
                done.pop(fid, None)
                st_removed_done_for_missing += 1
            # failed 保持原样，不额外添加

    state["done"] = done
    state["failed"] = failed

    # 4) 备份并落盘
    if args.dry_run:
        print("=== DRY RUN === (no files modified)")
    else:
        ds_bak = backup_file(dataset_path, backup_dir=backup_dir)
        st_bak = backup_file(state_path, backup_dir=backup_dir)
        save_state(state_path, state)
        print(f"Backups created:\n  dataset -> {ds_bak}\n  state   -> {st_bak}")

    # 5) 报告
    print("\n=== SUMMARY ===")
    print(f"clean_root: {clean_root}")
    print(f"files_total: {files_total} (text_files: {texts_total})")
    print(f"min_chars: {min_chars}")
    print(f"dataset_format: {ds_format}")
    print(f"dataset_kept: {kept}")
    print(f"dataset_removed_not_in_clean_root: {removed_not_in_folder}")
    print(f"dataset_removed_too_short_or_nontext: {removed_too_short}")
    print(f"state_updates_ok: {st_updates_ok}")
    print(f"state_updates_skip: {st_updates_skip}")
    print(f"state_removed_done_for_missing_in_dataset: {st_removed_done_for_missing}")
    if args.prune_orphans:
        print(f"state_pruned_orphans_done: {len(orph_done)}")
        print(f"state_pruned_orphans_failed: {len(orph_fail)}")
    else:
        print("state_orphan_prune: disabled (use --prune-orphans to prune)")

    # 额外提示：如果 kept 很小，可能 clean_root 与 dataset 不匹配（比如 clean_text 版本不同）
    if kept == 0:
        print("\n[WARN] dataset_kept=0. 这通常意味着：")
        print(" - 你给的 --clean-root 不包含当时数据集对应的文件；或")
        print(" - 合成数据时的 main.py 清洗逻辑与现在的 main.py 不一致（source_text 对不上）；或")
        print(" - dataset 文件不是你想修正的那个。")

if __name__ == "__main__":
    main()
