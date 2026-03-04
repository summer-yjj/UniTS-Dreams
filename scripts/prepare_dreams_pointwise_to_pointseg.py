import os
import json
import numpy as np
import mne
from tqdm import tqdm

"""
从 DREAMS 原始 EDF + 专家标注构建逐点标签：
- 每个 excerpt{i}.edf 与 Visual_scoring1_excerpt{i}.txt 对应一条长序列
- 使用 onset/duration 将 [onset, onset+duration) 区间内的采样点标记为 1（spindle）
- 输出格式为：
    data/dreams_pointwise/
      excerpt{i}/
        signal.npy  [N] float32
        label.npy   [N] int64 (0/1)
      splits/
        train.txt   # 每行一个相对路径，例如 excerpt1
        val.txt
        test.txt
      meta.json
该目录可直接被 DreamsPointSegDataset 加载，用于逐点分割 + 事件级评估。
"""


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始 DREAMS 数据位置（与 prepare_dreams.py 保持一致）
RAW_DIR = os.path.join(BASE_DIR, "..", "dataset", "DatabaseSpindles")

# 输出根目录（相对于项目根目录）
OUT_ROOT = os.path.join(BASE_DIR, "data", "dreams_pointwise")

# 采样率
FS = 256

# excerpt 划分（与 prepare_dreams.py 中保持一致，后续可根据需要调整）
TRAIN_IDS = [1, 2, 3, 4, 5]
VAL_IDS = [6]
TEST_IDS = [7, 8]


def load_eeg(edf_path):
    """读取单通道 EEG 信号，返回 1D numpy 数组。"""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    eeg = data[0]  # 使用第一个通道
    return eeg


def load_labels(label_path, T):
    """根据 DREAMS 的 Visual_scoring1 文本，构建长度为 T 的逐点标签数组。"""
    label = np.zeros(T, dtype=np.int64)

    with open(label_path) as f:
        for line in f:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 跳过类似 [vis1_Spindles/C3-A1] 这样的头
            if line.startswith("[") and line.endswith("]"):
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            try:
                onset, duration = map(float, parts)
            except ValueError:
                continue

            start = int(onset * FS)
            end = int((onset + duration) * FS)

            start = max(0, start)
            end = min(T, end)

            label[start:end] = 1

    return label


def process_ids(ids, split_name):
    """
    为给定的一组 excerpt ID 构建逐点信号与标签，并保存为
    OUT_ROOT/excerpt{idx}/{signal,label}.npy。
    返回相对 OUT_ROOT 的目录列表，用于写 splits/<split>.txt。
    """
    rel_dirs = []
    pos_counts = 0
    total_counts = 0

    for idx in tqdm(ids, desc=f"Processing {split_name}"):
        edf_path = os.path.join(RAW_DIR, f"excerpt{idx}.edf")
        lab_path = os.path.join(RAW_DIR, f"Visual_scoring1_excerpt{idx}.txt")

        if not os.path.isfile(edf_path):
            print(f"[prepare_pointwise] EDF not found: {edf_path}, skip.")
            continue
        if not os.path.isfile(lab_path):
            print(f"[prepare_pointwise] Label file not found: {lab_path}, skip.")
            continue

        eeg = load_eeg(edf_path)
        label = load_labels(lab_path, len(eeg))
        assert len(eeg) == len(label), "signal/label length mismatch for excerpt{}".format(idx)

        rel_dir = f"excerpt{idx}"
        out_dir = os.path.join(OUT_ROOT, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "signal.npy"), eeg.astype(np.float32))
        np.save(os.path.join(out_dir, "label.npy"), label.astype(np.int64))

        rel_dirs.append(rel_dir)
        total_counts += label.size
        pos_counts += int(label.sum())

    return rel_dirs, pos_counts, total_counts


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    all_splits = {
        "train": TRAIN_IDS,
        "val": VAL_IDS,
        "test": TEST_IDS,
    }

    splits_dir = os.path.join(OUT_ROOT, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    meta = {
        "fs": FS,
        "splits": {},
        "label_map": {"0": "background", "1": "spindle"},
        "raw_dir": os.path.relpath(RAW_DIR, BASE_DIR),
    }

    for split_name, ids in all_splits.items():
        rel_dirs, pos_counts, total_counts = process_ids(ids, split_name)
        # 写 split 列表（相对 root_path）
        split_txt = os.path.join(splits_dir, f"{split_name}.txt")
        with open(split_txt, "w", encoding="utf-8") as f:
            for d in rel_dirs:
                f.write(d + "\n")
        meta["splits"][split_name] = {
            "ids": ids,
            "n_records": len(rel_dirs),
            "total_points": int(total_counts),
            "pos_points": int(pos_counts),
            "pos_ratio": float(pos_counts / total_counts) if total_counts > 0 else 0.0,
            "split_file": os.path.relpath(split_txt, OUT_ROOT),
        }
        print(f"[prepare_pointwise] {split_name}: {len(rel_dirs)} excerpts, "
              f"pos_ratio={meta['splits'][split_name]['pos_ratio']:.6f}")

    meta_path = os.path.join(OUT_ROOT, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[prepare_pointwise] meta written to {meta_path}")


if __name__ == "__main__":
    main()

