"""
多类事件逐点标签构建脚手架：
- 未来从 data/dreams_project/raw/annotations 下按类别目录读取标注文件
- 当前若无额外标注，脚本仅检测到并提示后退出
"""
import os
import json
from typing import List, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROJECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dreams_project")
RAW_ANN_DIR = os.path.join(PROJECT_DATA_DIR, "raw", "annotations")
POINTSEG_DIR = os.path.join(PROJECT_DATA_DIR, "processed", "pointseg")


def parse_events(file_path: str) -> List[Tuple[str, float, float]]:
    """
    解析单个标注文件，返回 [(excerpt_id, start_sec, end_sec), ...]
    当前实现：示例格式，每行：<excerpt_id> <start_sec> <end_sec>
    未来可根据实际标注格式扩展。
    """
    events = []
    if not os.path.isfile(file_path):
        return events
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            excerpt_id, s, e = parts
            try:
                s = float(s)
                e = float(e)
            except ValueError:
                continue
            events.append((excerpt_id, s, e))
    return events


def build_labels_from_annotations(fs: int = 256):
    if not os.path.isdir(RAW_ANN_DIR):
        print(f"[build_multiclass] 未发现目录 {RAW_ANN_DIR}，仅保留已有 spindle 类，退出。")
        return

    class_dirs = [d for d in os.listdir(RAW_ANN_DIR)
                  if os.path.isdir(os.path.join(RAW_ANN_DIR, d))]
    if not class_dirs:
        print(f"[build_multiclass] {RAW_ANN_DIR} 下未发现类别子目录，暂不更新多类标签。")
        return

    print("[build_multiclass] 发现类别目录：", class_dirs)
    print("[build_multiclass] 当前脚本仅搭建框架，请根据实际标注格式在 parse_events 内实现解析逻辑，并在此函数中将事件投影到逐点标签。")

    # 示例：遍历每个类目录与标注文件
    for cls_name in class_dirs:
        cls_dir = os.path.join(RAW_ANN_DIR, cls_name)
        txt_files = [f for f in os.listdir(cls_dir) if f.endswith(".txt")]
        print(f"[build_multiclass] 类别 {cls_name} 下标注文件数：{len(txt_files)}")
        for txt in txt_files:
            path = os.path.join(cls_dir, txt)
            ev = parse_events(path)
            print(f"[build_multiclass] 解析 {path} 得到 {len(ev)} 个事件（未真正写回 label.npy，仅示例）。")

    print("[build_multiclass] 脚手架执行完毕，未修改任何现有 label.npy。")


def main():
    meta_path = os.path.join(PROJECT_DATA_DIR, "meta.json")
    fs = 256
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fs = int(meta.get("fs", fs))
        except Exception:
            pass
    build_labels_from_annotations(fs=fs)


if __name__ == "__main__":
    main()

