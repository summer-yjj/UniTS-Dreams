import os
import json
import numpy as np
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

SRC_DIR = os.path.join(PROJECT_ROOT, "data", "dreams_units")
OUT_BASE = os.path.join(PROJECT_ROOT, "data", "dreams_project")
POINTSEG_DIR = os.path.join(OUT_BASE, "processed", "pointseg")


def convert_split(split: str, fs: int = 256, window_T: int = 256):
    """
    将 data/dreams_units/{split}.npz (X:(N,256,1), y:(N,)) 转成
    data/dreams_project/processed/pointseg/{split}/{excerpt_id}/signal.npy,label.npy
    """
    src_path = os.path.join(SRC_DIR, f"{split}.npz")
    if not os.path.isfile(src_path):
        print(f"[convert_npz] {src_path} 不存在，跳过该 split={split}")
        return []

    data = np.load(src_path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError(f"{src_path} 期望包含 X 与 y 字段，实际 keys={list(data.keys())}")

    X = data["X"]  # (N, 256, 1)
    y = data["y"]  # (N,)

    if X.ndim != 3 or X.shape[2] != 1:
        raise ValueError(f"{src_path} 中 X 形状期望为 (N,256,1)，实际 {X.shape}")
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"{src_path} 中 y 形状期望为 (N,)，且与 X 第一维一致，实际 X{X.shape}, y{y.shape}")

    out_split_dir = os.path.join(POINTSEG_DIR, split)
    os.makedirs(out_split_dir, exist_ok=True)

    excerpt_ids = []
    N = X.shape[0]
    T = X.shape[1]
    for i in range(N):
        excerpt_id = f"{split}_{i:05d}"
        excerpt_dir = os.path.join(out_split_dir, excerpt_id)
        os.makedirs(excerpt_dir, exist_ok=True)

        sig = X[i, :, 0].astype(np.float32)  # [T]
        label_val = int(y[i])
        label = np.zeros(T, dtype=np.int64)
        if label_val == 1:
            label[:] = 1

        np.save(os.path.join(excerpt_dir, "signal.npy"), sig)
        np.save(os.path.join(excerpt_dir, "label.npy"), label)

        excerpt_ids.append(excerpt_id)

    print(f"[convert_npz] {split}: 生成 {len(excerpt_ids)} 个 excerpt，输出到 {out_split_dir}")
    return excerpt_ids


def write_meta(fs: int = 256, window_T: int = 256):
    meta_path = os.path.join(OUT_BASE, "meta.json")
    os.makedirs(OUT_BASE, exist_ok=True)
    meta = {
        "fs": fs,
        "window_T": window_T,
        "label_map": {
            "0": "background",
            "1": "spindle"
        },
        "source": "data/dreams_units/*.npz (window-level spindle classification)",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "notes": "当前仅包含 spindle 类（1），未来可在 raw/annotations 中增加其它事件标注并重建 label。"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[convert_npz] 写入 meta.json 至 {meta_path}")


def main():
    print("[convert_npz] 源目录：", SRC_DIR)
    print("[convert_npz] 目标目录：", POINTSEG_DIR)
    all_ids = {}
    for split in ["train", "val", "test"]:
        ids = convert_split(split, fs=256, window_T=256)
        all_ids[split] = ids

    # 可选：写 splits/{train,val,test}.txt
    splits_dir = os.path.join(OUT_BASE, "processed", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    for split, ids in all_ids.items():
        split_txt = os.path.join(splits_dir, f"{split}.txt")
        with open(split_txt, "w", encoding="utf-8") as f:
            for eid in ids:
                f.write(eid + "\n")
        print(f"[convert_npz] 写入 {split_txt}，共 {len(ids)} 行")

    write_meta(fs=256, window_T=256)


if __name__ == "__main__":
    main()

