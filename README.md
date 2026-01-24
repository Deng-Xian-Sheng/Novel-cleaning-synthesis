
能正常运行的参数：
```
python main.py --workers 5 --max-input-chars 1000 --min-delay 2 --max-retries 8 --timeout 300
```

`find_by_hash.py`，当你需要通过哈希寻找某个文件时，使用这个脚本。哈希可以是来自控制台打印或者 `dataset.state.json`


`reconcile_dataset.py`用于：
- 将`dataset.json`中的数据、被扫描的目录、`dataset.state.json`对齐
- 将`dataset.json`中的数据、`dataset.state.json`与过短文本跳过（默认10）对齐

用法：
```
python reconcile_dataset.py \
  --clean-root 小说 \
  --dataset dataset.jsonl \
  --state dataset.state.json \
  --min-chars 10 \
  --prune-orphans
```

输出：
```
((venv) ) likewendy@likewendy-PC:~/Desktop/Novel-cleaning-synthesis$ python reconcile_dataset.py   --clean-root 小说   --dataset dataset.jsonl   --state dataset.state.json   --min-chars 10   --prune-orphans
Backups created:
  dataset -> /home/likewendy/Desktop/Novel-cleaning-synthesis/dataset.jsonl.bak.20260124-153653
  state   -> /home/likewendy/Desktop/Novel-cleaning-synthesis/dataset.state.json.bak.20260124-153653

=== SUMMARY ===
clean_root: /home/likewendy/Desktop/Novel-cleaning-synthesis/小说
files_total: 4963 (text_files: 4818)
min_chars: 10
dataset_format: jsonl
dataset_kept: 4800
dataset_removed_not_in_clean_root: 34
dataset_removed_too_short_or_nontext: 5
state_updates_ok: 4800
state_updates_skip: 162
state_removed_done_for_missing_in_dataset: 0
state_pruned_orphans_done: 34
state_pruned_orphans_failed: 1
```