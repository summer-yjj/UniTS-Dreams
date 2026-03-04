# 将逐点 mask 转为事件段 (start, end)，支持合并与最短长度
import numpy as np


def mask_to_events(mask_01, min_len=1, merge_gap=0):
    """
    将二值 mask（0/1）转为事件段列表。
    mask_01: 1D array or list of 0/1 (length T)
    min_len: 最短事件长度（点数），短于此的段丢弃
    merge_gap: 两段间隔小于等于此则合并
    Returns: list of (start, end) 左闭右开 [start, end) 索引
    """
    mask = np.asarray(mask_01, dtype=np.int32).flatten()
    events = []
    i = 0
    while i < len(mask):
        if mask[i] != 1:
            i += 1
            continue
        start = i
        while i < len(mask) and mask[i] == 1:
            i += 1
        end = i
        if end - start >= min_len:
            events.append((int(start), int(end)))
    if merge_gap <= 0:
        return events
    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def pred_to_events(mask_01, min_len=1, merge_gap=0):
    """别名：与 mask_to_events 相同。"""
    return mask_to_events(mask_01, min_len=min_len, merge_gap=merge_gap)


def gt_to_events(labels, positive_class=1, min_len=1, merge_gap=0):
    """
    labels: [T] 逐点类别
    positive_class: 视为“事件”的类别
    """
    mask = (np.asarray(labels).flatten() == positive_class).astype(np.int32)
    return mask_to_events(mask, min_len=min_len, merge_gap=merge_gap)


def labels_to_events(labels, class_id, min_len=1, merge_gap=0):
    """
    通用多类接口：从逐点多类标签中抽取某一类的事件段。
    class_id >= 1 通常对应具体事件类别，0 为背景。
    """
    mask = (np.asarray(labels).flatten() == int(class_id)).astype(np.int32)
    return mask_to_events(mask, min_len=min_len, merge_gap=merge_gap)
