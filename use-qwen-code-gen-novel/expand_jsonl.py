#!/usr/bin/env python3
"""
将 JSONL 文件展开到文件夹结构中
"""
import json
import os
import re
import sys
from pathlib import Path


def extract_blockquote_content(text: str, block_start: str) -> str:
    """
    从 Markdown 引用块中提取内容
    格式: > **标题**\n>\n> 内容行1\n> 内容行2...
    """
    # 匹配引用块：以 > **block_start** 开头，后面跟着多行以 > 开头的行（包括空引用行 >）
    pattern = rf'> \*\*{re.escape(block_start)}\*\*(?:\s*\n>.*)*'
    match = re.search(pattern, text)
    if not match:
        return ""
    
    block_text = match.group(0)
    lines = []
    for line in block_text.split('\n'):
        line = line.strip()
        if line.startswith('>') and not line.startswith('> **'):
            # 去掉开头的 > 和可能的空格
            clean_line = re.sub(r'^>\s?', '', line)
            lines.append(clean_line)
    
    # 过滤掉开头的空行，但保留内容中的空行
    # 找到第一个非空行的位置
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            start_idx = i
            break
    
    return '\n'.join(lines[start_idx:])


def parse_assistant_content(content: str):
    """
    解析 assistant 的 content，提取综述和各段落
    返回: (综述, [(段落标题, 正文, 前文提要), ...], 无段落正文)
    无段落正文: 当没有段落时，返回剩余的正文内容
    """
    # 提取综述
    summary = extract_blockquote_content(content, "综述")
    
    # 检查是否有段落
    para_pattern = r'## 第\d+段：([^\n]+)\n\n(.*?)(?=\n## 第\d+段：|$)'
    para_matches = list(re.finditer(para_pattern, content, re.DOTALL))
    
    paragraphs = []
    no_para_body = ""
    
    if para_matches:
        # 有段落，按原来的逻辑处理
        for match in para_matches:
            title = match.group(1).strip()
            body_and_recap = match.group(2)
            
            # 分离正文和前文提要
            recap_pattern = r'\n\n> \*\*前文提要\*\*(?:\s*\n>.*)*'
            recap_match = re.search(recap_pattern, body_and_recap)
            
            if recap_match:
                # 有前文提要
                body = body_and_recap[:recap_match.start()].strip()
                recap_text = body_and_recap[recap_match.start():]
                recap = extract_blockquote_content(recap_text, "前文提要")
            else:
                # 没有前文提要（最后一段）
                body = body_and_recap.strip()
                recap = ""
            
            paragraphs.append((title, body, recap))
    else:
        # 没有段落，提取综述后的剩余内容作为正文
        # 找到综述块的结束位置
        summary_pattern = rf'> \*\*综述\*\*(?:\s*\n>.*)*'
        summary_match = re.search(summary_pattern, content)
        if summary_match:
            # 综述后的内容
            remaining = content[summary_match.end():].strip()
            no_para_body = remaining
        else:
            # 没有综述块，整个内容作为正文
            no_para_body = content.strip()
    
    return summary, paragraphs, no_para_body


def sanitize_filename(name: str) -> str:
    """清理文件名，移除非法字符"""
    # 替换常见的非法字符
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    # 去掉首尾空格
    name = name.strip()
    # 限制长度
    if len(name) > 100:
        name = name[:100]
    return name


def process_jsonl(input_file: str, output_dir: str):
    """处理 JSONL 文件，展开到目录结构"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    novel_index = 1
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}", file=sys.stderr)
                continue
            
            # 创建小说目录
            novel_dir = output_path / f"novel{novel_index}"
            novel_dir.mkdir(exist_ok=True)
            novel_index += 1
            
            # 提取 messages
            messages = data.get('messages', [])
            user_content = ""
            assistant_content = ""
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    user_content = content
                elif role == 'assistant':
                    assistant_content = content
            
            # 写入用户消息
            if user_content:
                (novel_dir / "用户消息.md").write_text(user_content, encoding='utf-8')
            
            # 解析 assistant 内容
            if assistant_content:
                summary, paragraphs, no_para_body = parse_assistant_content(assistant_content)
                
                # 写入综述
                (novel_dir / "综述.md").write_text(summary, encoding='utf-8')
                
                if paragraphs:
                    # 有段落，按段落创建目录
                    for idx, (title, body, recap) in enumerate(paragraphs, 1):
                        # 确定文件夹名
                        safe_title = sanitize_filename(title)
                        if safe_title:
                            block_dir_name = f"{safe_title}-block{idx}"
                        else:
                            block_dir_name = f"block{idx}"
                        
                        block_dir = novel_dir / block_dir_name
                        block_dir.mkdir(exist_ok=True)
                        
                        # 写入段落综述（前文提要）- 仅当不为空时
                        if recap:
                            (block_dir / "段落综述.md").write_text(recap, encoding='utf-8')
                        
                        # 写入正文
                        (block_dir / "正文.md").write_text(body, encoding='utf-8')
                else:
                    # 没有段落，正文与综述同级
                    if no_para_body:
                        (novel_dir / "正文.md").write_text(no_para_body, encoding='utf-8')
    
    print(f"处理完成，共处理 {novel_index - 1} 个小说")


def main():
    if len(sys.argv) < 2:
        print("用法: python expand_jsonl.py <input.jsonl> [output_dir]", file=sys.stderr)
        print("默认输出目录: novel", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "novel"
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    process_jsonl(input_file, output_dir)


if __name__ == "__main__":
    main()
