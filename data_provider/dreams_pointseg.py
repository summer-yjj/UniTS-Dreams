# DREAMS 逐采样点分割 Dataset：滑窗、双格式支持、meta
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def _infer_format(root_path, path_or_dir):
    """推断数据格式：A) 单 .npy [N,2]  B) signal.npy + label.npy。path_or_dir 为文件或目录路径（可为相对 root_path 或已完整路径）。"""
    if not path_or_dir:
        base = root_path
    elif os.path.isabs(path_or_dir) or os.path.isfile(path_or_dir) or os.path.isdir(path_or_dir):
        base = path_or_dir
    else:
        base = os.path.join(root_path, path_or_dir)
    if os.path.isfile(base):
        arr = np.load(base, allow_pickle=True)
        if getattr(arr, "shape", None) is not None and len(arr.shape) == 2 and arr.shape[1] >= 2:
            return "A"
        base = os.path.dirname(base)
    signal_path = os.path.join(base, "signal.npy")
    label_path = os.path.join(base, "label.npy")
    if os.path.isfile(signal_path) and os.path.isfile(label_path):
        return "B"
    for f in (glob.glob(os.path.join(base, "*.npy")) or []):
        try:
            arr = np.load(f, allow_pickle=True)
            if getattr(arr, "shape", None) is not None and len(arr.shape) == 2 and arr.shape[1] >= 2:
                return "A"
        except Exception:
            pass
    return None


def _normalize_signal_shape(sig):
    """Normalize signal shape to [T, C] for UniTS input convention."""
    sig = np.asarray(sig)
    if sig.ndim == 1:
        return sig[:, np.newaxis]
    if sig.ndim != 2:
        raise ValueError("DREAMS pointseg: signal must be 1D or 2D, got shape {}".format(sig.shape))
    return sig


class DreamsPointSegDataset(Dataset):
    """
    逐采样点分割：__getitem__ 返回 (x, y, meta)。
    x: [T, C] FloatTensor,  y: [T] LongTensor,  meta: dict (excerpt_id, window_start, window_end, fs 等)。
    支持滑窗：window_T, stride_T, fs=256。
    数据格式：
      A) 单 .npy 形状 [N, 2]：第 0 列信号，第 1 列逐点 label
      B) signal.npy [N] + label.npy [N]
    若为 B，split 文件里每行一个目录名（相对 root_path）或绝对路径；若为 A，每行一个 .npy 路径或 (id, path)。
    """

    def __init__(
        self,
        root_path,
        flag="train",
        window_T=256,
        stride_T=128,
        fs=256,
        num_classes=2,
        split_files=None,
        split_list=None,
        file_list=None,
        debug=False,
    ):
        self.root_path = os.path.normpath(root_path)
        self.flag = flag
        self.window_T = int(window_T)
        self.stride_T = int(stride_T)
        self.fs = int(fs)
        self.num_classes = int(num_classes)
        self.debug = bool(debug)
        self.windows = []  # list of (excerpt_id, path_or_dir, start_idx, end_idx) or (..., sig_arr, lab_arr)

        if split_list is not None:
            self._build_from_list(split_list)
        elif split_files:
            if isinstance(split_files, str):
                split_files = [split_files]
            for sf in split_files:
                path = os.path.join(self.root_path, sf) if not os.path.isabs(sf) else sf
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    self._build_from_list(lines)
                else:
                    self._build_from_list([path])
        elif file_list is not None:
            self._build_from_list(file_list)
        else:
            # 默认：在 root_path 下找 train/val/test 子目录或单文件
            for name in ["train", "val", "test"]:
                if name != flag:
                    continue
                single = os.path.join(self.root_path, name + ".npy")
                if os.path.isfile(single):
                    self._add_excerpt(name, single)
                d = os.path.join(self.root_path, name)
                if os.path.isdir(d):
                    for sub in os.listdir(d):
                        sub_path = os.path.join(d, sub)
                        rel_path = os.path.join(name, sub)
                        if os.path.isdir(sub_path):
                            self._add_excerpt(sub, rel_path)
                        elif sub.endswith(".npy"):
                            self._add_excerpt(sub.replace(".npy", ""), rel_path)
            if not self.windows:
                for f in glob.glob(os.path.join(self.root_path, "*.npy")):
                    self._add_excerpt(os.path.basename(f).replace(".npy", ""), f)

    def _add_excerpt(self, excerpt_id, path_or_dir):
        path_or_dir = path_or_dir.strip()
        full = os.path.join(self.root_path, path_or_dir) if path_or_dir and not os.path.isabs(path_or_dir) else path_or_dir
        full_exists = os.path.isfile(full) or os.path.isdir(full)
        passed_to_infer = full if full_exists else path_or_dir
        fmt = _infer_format(self.root_path, passed_to_infer)
        if fmt == "B":
            base_dir = full if os.path.isdir(full) else path_or_dir
            sig_path = os.path.join(base_dir, "signal.npy")
            lab_path = os.path.join(base_dir, "label.npy")
            if not os.path.isfile(sig_path) or not os.path.isfile(lab_path):
                return
            sig = np.load(sig_path)
            lab = np.load(lab_path)
        elif fmt == "A":
            path = full if os.path.isfile(full) else None
            if path is None and os.path.isdir(full):
                for f in glob.glob(os.path.join(full, "*.npy")):
                    try:
                        arr = np.load(f, allow_pickle=True)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            path = f
                            break
                    except Exception:
                        pass
            if path is None:
                path = path_or_dir if os.path.isfile(path_or_dir) else full
            if path is None or not os.path.isfile(path):
                return
            arr = np.load(path, allow_pickle=True)
            if arr.ndim != 2 or arr.shape[1] < 2:
                sh = getattr(arr, "shape", None)
                raise ValueError("DREAMS pointseg format A expects .npy shape [N, 2] (signal, label). Got shape {} at {}".format(sh, path))
            sig = np.asarray(arr[:, 0], dtype=np.float64)
            lab = np.asarray(arr[:, 1], dtype=np.int64)
        else:
            raise ValueError(
                "DREAMS pointseg: unknown format. Expect A) one .npy [N,2] or B) signal.npy + label.npy. Path: {}".format(path_or_dir)
            )
        sig = _normalize_signal_shape(sig)
        lab = np.atleast_1d(lab).flatten()
        if sig.shape[0] != len(lab):
            raise ValueError("DREAMS pointseg: signal length {} != label length {} at {}".format(sig.shape[0], len(lab), path_or_dir))
        N = sig.shape[0]
        base_dir = (full if os.path.isdir(full) else path_or_dir) if fmt == "B" else None
        path_a = path if fmt == "A" else None
        for start in range(0, max(0, N - self.window_T) + 1, self.stride_T):
            end = start + self.window_T
            if end > N:
                break
            y_win = lab[start:end]
            has_pos = bool(np.any(y_win > 0))
            self.windows.append((excerpt_id, start, end, fmt, base_dir, path_a, has_pos))

    def _build_from_list(self, lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                excerpt_id, path = line.split("\t", 1)
            else:
                excerpt_id = os.path.basename(line.rstrip("/"))
                path = line
            path = os.path.join(self.root_path, path) if not os.path.isabs(path) else path
            self._add_excerpt(excerpt_id, path)

    def __len__(self):
        return len(self.windows)

    def _load_segment(self, fmt, base_dir, path_a):
        if fmt == "B":
            sig = np.load(os.path.join(base_dir, "signal.npy"))
            lab = np.load(os.path.join(base_dir, "label.npy"))
        else:
            arr = np.load(path_a, allow_pickle=True)
            sig = np.asarray(arr[:, 0], dtype=np.float64)
            lab = np.asarray(arr[:, 1], dtype=np.int64)
        sig = _normalize_signal_shape(sig)
        lab = np.atleast_1d(lab).flatten()
        return sig, lab

    def __getitem__(self, idx):
        item = self.windows[idx]
        excerpt_id, start, end, fmt, base_dir, path_a, _has_pos = item
        sig, lab = self._load_segment(fmt, base_dir, path_a)
        x = sig[start:end].astype(np.float32)
        y = lab[start:end].astype(np.int64)
        x = np.clip(np.nan_to_num(x), -1e9, 1e9)
        if x.ndim != 2:
            raise ValueError("DREAMS pointseg: expected x window [T,C], got shape {}".format(x.shape))
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        meta = {
            "excerpt_id": excerpt_id,
            "window_start": int(start),
            "window_end": int(end),
            "fs": self.fs,
        }
        return x_t, y_t, meta

    def get_window_sample_weights(self, pos_window_weight=3.0):
        """Return per-window sampling weights (positive-window upweighting)."""
        pos_w = max(float(pos_window_weight), 1.0)
        weights = []
        for item in self.windows:
            has_pos = bool(item[6]) if len(item) > 6 else False
            weights.append(pos_w if has_pos else 1.0)
        return weights


def collate_pointseg(batch):
    """batch: list of (x, y, meta). 返回 batch_x [B,T,C], batch_y [B,T], meta list[dict]."""
    xs, ys, metas = zip(*batch)
    batch_x = torch.stack(xs, dim=0)
    batch_y = torch.stack(ys, dim=0)
    return batch_x, batch_y, list(metas)
