import os
import json
import numpy as np
import mne
from tqdm import tqdm

# =====================
# 基本配置（你只需要改这里）
# =====================
# ===== 路径基准（脚本所在目录）=====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(
    BASE_DIR, "..", "dataset", "DatabaseSpindles"
)

OUT_DIR = os.path.join(
    BASE_DIR, "data", "dreams_units"
)


FS = 256                 # 采样率
WIN_LEN = 256            # 窗口长度（1 秒）
STRIDE = 64              # 步长（0.25 秒）

TRAIN_IDS = [1, 2, 3, 4, 5]
VAL_IDS   = [6]
TEST_IDS  = [7, 8]

# =====================
# 工具函数
# =====================

def load_eeg(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    eeg = data[0]  # 单通道
    return eeg


def load_labels(label_path, T):
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



def make_windows(eeg, label):
    X, y = [], []
    for i in range(0, len(eeg) - WIN_LEN, STRIDE):
        seg_x = eeg[i:i + WIN_LEN]
        seg_y = label[i:i + WIN_LEN]

        X.append(seg_x[:, None])              # (L, 1)
        y.append(int(seg_y.max() > 0))         # 窗口是否包含 spindle

    return np.array(X), np.array(y)


def process_split(ids):
    X_all, y_all = [], []

    for idx in tqdm(ids):
        edf_path = os.path.join(RAW_DIR, f"excerpt{idx}.edf")
        lab_path = os.path.join(
            RAW_DIR, f"Visual_scoring1_excerpt{idx}.txt"
        )

        eeg = load_eeg(edf_path)
        label = load_labels(lab_path, len(eeg))

        X, y = make_windows(eeg, label)

        X_all.append(X)
        y_all.append(y)

    return np.concatenate(X_all), np.concatenate(y_all)


# =====================
# 主流程
# =====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Processing TRAIN...")
    X_train, y_train = process_split(TRAIN_IDS)

    print("Processing VAL...")
    X_val, y_val = process_split(VAL_IDS)

    print("Processing TEST...")
    X_test, y_test = process_split(TEST_IDS)

    np.savez(os.path.join(OUT_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(OUT_DIR, "val.npz"),   X=X_val,   y=y_val)
    np.savez(os.path.join(OUT_DIR, "test.npz"),  X=X_test,  y=y_test)

    meta = {
        "fs": FS,
        "win_len": WIN_LEN,
        "stride": STRIDE,
        "train_ids": TRAIN_IDS,
        "val_ids": VAL_IDS,
        "test_ids": TEST_IDS,
        "pos_ratio_train": float(y_train.mean()),
        "pos_ratio_val": float(y_val.mean()),
        "pos_ratio_test": float(y_test.mean())
    }

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done!")
    print(meta)


if __name__ == "__main__":
    main()
