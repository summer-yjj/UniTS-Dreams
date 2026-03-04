# 事件级指标：IoU 匹配的 precision/recall/F1，以及边界误差 (ms)
import numpy as np
from utils.event_post import pred_to_events, gt_to_events, labels_to_events


def _iou1d(pred_interval, gt_interval):
    """(start, end) 左闭右开。返回 IoU [0,1]。"""
    ps, pe = pred_interval
    gs, ge = gt_interval
    inter_s = max(ps, gs)
    inter_e = min(pe, ge)
    if inter_e <= inter_s:
        return 0.0
    inter = inter_e - inter_s
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def match_events_by_iou(pred_events, gt_events, iou_thr=0.3):
    """
    贪心 IoU 匹配：对每个 gt 找 IoU>=iou_thr 的 pred，一对多/多对一取最优。
    Returns: list of (pred_idx, gt_idx) 匹配对；未匹配的 pred/gt 不计入。
    """
    pred_events = list(pred_events)
    gt_events = list(gt_events)
    used_pred = set()
    used_gt = set()
    matches = []
    for gi, g in enumerate(gt_events):
        best_pi, best_iou = -1, iou_thr
        for pi, p in enumerate(pred_events):
            if pi in used_pred:
                continue
            iou = _iou1d(p, g)
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_pi >= 0:
            matches.append((best_pi, gi))
            used_pred.add(best_pi)
            used_gt.add(gi)
    return matches


def event_precision_recall_f1(pred_events, gt_events, iou_thr=0.3):
    """
    pred_events / gt_events: list of (start, end).
    Returns: event_precision, event_recall, event_f1 (0~1).
    """
    matches = match_events_by_iou(pred_events, gt_events, iou_thr=iou_thr)
    n_pred = len(pred_events)
    n_gt = len(gt_events)
    tp = len(matches)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def boundary_error_ms(pred_events, gt_events, fs, iou_thr=0.3):
    """
    对 IoU 匹配上的事件对计算 |start_pred - start_gt| 和 |end_pred - end_gt|，转为 ms。
    Returns: mean_boundary_error_ms (start 与 end 误差的平均，再对匹配对求平均)。
    """
    matches = match_events_by_iou(pred_events, gt_events, iou_thr=iou_thr)
    if not matches:
        return float("nan")
    errs = []
    for pi, gi in matches:
        ps, pe = pred_events[pi]
        gs, ge = gt_events[gi]
        err_start = abs(ps - gs) * 1000.0 / fs
        err_end = abs(pe - ge) * 1000.0 / fs
        errs.append((err_start + err_end) / 2.0)
    return float(np.mean(errs))


def compute_event_metrics(pred_mask_01, gt_labels, fs, positive_class=1, min_len=1, merge_gap=0, iou_thr=0.3):
    """
    pred_mask_01: [T] 预测二值
    gt_labels: [T] 真实逐点类别
    Returns: dict with event_precision, event_recall, event_f1, mean_boundary_error_ms
    """
    pred_events = pred_to_events(pred_mask_01, min_len=min_len, merge_gap=merge_gap)
    gt_events = gt_to_events(gt_labels, positive_class=positive_class, min_len=1, merge_gap=0)
    prec, rec, f1 = event_precision_recall_f1(pred_events, gt_events, iou_thr=iou_thr)
    mbe = boundary_error_ms(pred_events, gt_events, fs, iou_thr=iou_thr)
    return {
        "event_precision": prec,
        "event_recall": rec,
        "event_f1": f1,
        "mean_boundary_error_ms": mbe,
        "n_pred_events": len(pred_events),
        "n_gt_events": len(gt_events),
    }


def compute_event_metrics_multiclass(pred_labels, gt_labels, fs, num_classes,
                                     min_len=1, merge_gap=0, iou_thr=0.3):
    """
    多类事件指标：对每个 class_id>=1 计算事件级指标，并给出 macro 平均。
    pred_labels / gt_labels: [T] 或 [B,T] 整型标签。
    """
    pred_arr = np.asarray(pred_labels)
    gt_arr = np.asarray(gt_labels)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr[None, :]
        gt_arr = gt_arr[None, :]

    per_class = {}
    macro_f1_list = []
    for c in range(1, int(num_classes)):
        cls_metrics = {"event_precision": [], "event_recall": [], "event_f1": [], "mean_boundary_error_ms": []}
        for i in range(pred_arr.shape[0]):
            pred_c = (pred_arr[i] == c).astype(np.int32)
            gt_c = gt_arr[i]
            m = compute_event_metrics(pred_c, gt_c, fs, positive_class=c,
                                      min_len=min_len, merge_gap=merge_gap, iou_thr=iou_thr)
            for k in cls_metrics:
                cls_metrics[k].append(m.get(k))
        agg = {}
        for k, vals in cls_metrics.items():
            vals = [v for v in vals if v is not None]
            agg[k] = float(np.nanmean(vals)) if vals else float("nan")
        per_class[str(c)] = agg
        if np.isfinite(agg.get("event_f1", np.nan)):
            macro_f1_list.append(agg["event_f1"])

    overall_macro_f1 = float(np.nanmean(macro_f1_list)) if macro_f1_list else float("nan")
    return {
        "per_class": per_class,
        "macro_event_f1": overall_macro_f1,
    }
