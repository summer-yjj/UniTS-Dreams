# 逐点分割实验：train/vali/test，保存 best、results.json/csv、vis、curves、analysis、cases
from __future__ import division
import os
import sys
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from utils.tools import adjust_learning_rate
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.seg_losses import compute_seg_loss
from utils.event_post import pred_to_events, gt_to_events
from utils.event_metrics import compute_event_metrics, compute_event_metrics_multiclass

try:
    import wandb
except ImportError:
    wandb = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from data_provider.data_factory import data_provider
from utils.ddp import is_main_process, get_world_size

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def _print(msg, path=None):
    print(msg)
    if not path:
        return
    if path.endswith(".log"):
        log_file = path
        log_dir = os.path.dirname(path)
    else:
        log_dir = path
        log_file = os.path.join(log_dir, "finetune_output.log")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def _point_metrics(logits, y, num_classes):
    """logits [B,C,T], y [B,T]. Returns acc, macro_f1, spindle_f1 (class 1)."""
    pred = logits.argmax(dim=1).cpu().numpy().flatten()
    y_np = y.cpu().numpy().flatten()
    valid = (y_np >= 0) & (y_np < num_classes)
    if valid.sum() == 0:
        return 0.0, 0.0, 0.0
    pred = pred[valid]
    y_np = y_np[valid].astype(int)
    acc = accuracy_score(y_np, pred)
    macro_f1 = f1_score(y_np, pred, average="macro", zero_division=0)
    if num_classes >= 2:
        spindle_f1 = f1_score(y_np, pred, average="binary", pos_label=1, zero_division=0)
    else:
        spindle_f1 = macro_f1
    return acc, macro_f1, spindle_f1




def _safe_torch_load(path, map_location="cpu"):
    """Compat loader for PyTorch>=2.6 where torch.load defaults to weights_only=True."""
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "weights_only" in msg:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise



def _normalize_state_dict_keys(state_dict):
    """Strip wrapper prefixes to maximize checkpoint key matching."""
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "student.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def _extract_pretrain_state_dict(raw_obj):
    if isinstance(raw_obj, dict):
        if "student" in raw_obj and isinstance(raw_obj["student"], dict):
            return raw_obj["student"]
        if "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
            return raw_obj["state_dict"]
        if "model" in raw_obj and isinstance(raw_obj["model"], dict):
            return raw_obj["model"]
    return raw_obj

def _event_metrics_agg(metrics_list):
    if not metrics_list:
        return {}
    keys = ["event_precision", "event_recall", "event_f1", "mean_boundary_error_ms"]
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and np.isfinite(m.get(k))]
        out[k] = float(np.nanmean(vals)) if vals else float("nan")
    return out


class Exp_PointSeg:
    def __init__(self, args):
        self.args = args
        with open(args.task_data_config_path, "r") as f:
            config = yaml.safe_load(f)
        task_dataset = config.get("task_dataset", {})
        self.task_data_config_list = [[k, v] for k, v in task_dataset.items()]
        self.task_data_config = task_dataset
        self.device_id = 0
        if getattr(args, "distributed", False):
            import torch.distributed as dist
            self.device_id = dist.get_rank() % torch.cuda.device_count()
        self.model = self._build_model()
        self.path = None

    def _build_model(self):
        import importlib
        module = importlib.import_module("models." + self.args.model)
        model = module.Model(self.args, self.task_data_config_list).to(self.device_id)
        if getattr(self.args, "distributed", False):
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device_id],
                find_unused_parameters=True,
            )
        return model

    def _get_data(self, flag):
        config = self.task_data_config_list[0][1]
        data_set, data_loader = data_provider(self.args, config, flag, ddp=False)
        return data_set, data_loader

    def _seg_loss_cfg(self, class_weights_from_train=None):
        cfg = {
            "seg_loss": getattr(self.args, "seg_loss", "ce_dice"),
            "class_weight": getattr(self.args, "class_weight", "auto"),
            "num_classes": self.task_data_config_list[0][1].get("num_classes", 2),
            "focal_gamma": getattr(self.args, "focal_gamma", 2.0),
            "tversky_alpha": getattr(self.args, "tversky_alpha", 0.7),
            "tversky_beta": getattr(self.args, "tversky_beta", 0.3),
            "bg_keep_prob": getattr(self.args, "bg_keep_prob", 1.0),
        }
        if cfg["class_weight"] == "manual" and hasattr(self.args, "class_weights"):
            cfg["class_weights"] = self.args.class_weights
        if class_weights_from_train is not None:
            cfg["class_weights"] = class_weights_from_train
        return cfg

    def train(self, setting):
        self.path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(self.path, exist_ok=True)
        log_file = os.path.join(self.path, "finetune_output.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("")
        config = self.task_data_config_list[0][1]
        task_id = 0
        fs = config.get("fs", 256)
        min_len = config.get("min_event_len", 1)
        merge_gap = config.get("merge_gap", 0)
        iou_thr = config.get("iou_thr", 0.3)
        train_set, train_loader = self._get_data("train")
        val_set, val_loader = self._get_data("val")
        if len(val_set) == 0:
            val_loader = self._get_data("test")[1]
            val_set = self._get_data("test")[0]

        # Load pretrained weights (Optional)
        if self.args.pretrained_weight is not None:
            pretrain_weight_path = self.args.pretrained_weight
            _print("loading pretrained model: {}".format(pretrain_weight_path), self.path)
            if not os.path.isfile(pretrain_weight_path):
                _print("pretrained model not found: {}".format(pretrain_weight_path), self.path)
            else:
                state_dict = _extract_pretrain_state_dict(_safe_torch_load(pretrain_weight_path, map_location="cpu"))
                if not isinstance(state_dict, dict):
                    _print("pretrained model format invalid (expected dict state_dict)", self.path)
                    state_dict = {}
                ckpt = {}
                for k, v in state_dict.items():
                    if "cls_prompts" not in k:
                        ckpt[k] = v
                ckpt = _normalize_state_dict_keys(ckpt)
                model = self.model.module if hasattr(self.model, "module") else self.model
                model_sd = model.state_dict()
                matched = {}
                skipped = []
                for k, v in ckpt.items():
                    if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
                        matched[k] = v
                    else:
                        skipped.append(k)
                msg = model.load_state_dict(matched, strict=False)
                _print("pretrained matched keys: {} skipped keys: {}".format(len(matched), len(skipped)), self.path)
                if skipped:
                    _print("pretrained first skipped keys: {}".format(skipped[:10]), self.path)
                _print("pretrained load_state_dict: {}".format(msg), self.path)

        results_dir = os.path.join("results", setting)
        curves_dir = os.path.join(results_dir, "curves")
        os.makedirs(curves_dir, exist_ok=True)
        tb_dir = os.path.join(results_dir, "tensorboard")
        if SummaryWriter is not None and is_main_process():
            writer = SummaryWriter(log_dir=tb_dir)
        else:
            writer = None
        metrics_path = os.path.join(results_dir, "metrics.csv")
        metrics_header = ["epoch", "train_loss", "val_acc", "val_macro_f1", "val_spindle_f1", "val_event_f1", "lr"]
        if is_main_process():
            with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(metrics_header)

        num_classes = config.get("num_classes", 2)
        if getattr(self.args, "debug", "") == "enabled":
            bx, by, meta = next(iter(train_loader))
            _print("DEBUG pointseg batch: batch_x {} batch_y {} meta len {}".format(
                bx.shape, by.shape, len(meta)), self.path)
        if self._seg_loss_cfg()["class_weight"] == "auto":
            from collections import Counter
            all_y = []
            for _, (_, y, _) in enumerate(train_loader):
                all_y.extend(y.numpy().flatten().tolist())
            cnt = Counter(all_y)
            total = sum(cnt.values()) or 1
            median_freq = np.median(list(cnt.values())) or 1.0
            class_weights = [median_freq / max(cnt.get(c, 1), 1) for c in range(num_classes)]
            if getattr(self.args, "seg_pos_weight", None) is not None:
                for c in range(1, num_classes):
                    class_weights[c] *= self.args.seg_pos_weight
            self._class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            train_dist = {"count": {str(c): cnt.get(c, 0) for c in range(num_classes)}, "total": total,
                          "pct": {str(c): (cnt.get(c, 0) / total) * 100 for c in range(num_classes)}}
        else:
            self._class_weights_tensor = None
            train_dist = None
        # 统计验证集类别分布并写入 results_dir，用于检查类别不平衡
        if is_main_process():
            from collections import Counter
            all_y_val = []
            for _, (_, y, _) in enumerate(val_loader):
                all_y_val.extend(y.numpy().flatten().tolist())
            cnt_val = Counter(all_y_val)
            total_val = sum(cnt_val.values()) or 1
            val_dist = {"count": {str(c): cnt_val.get(c, 0) for c in range(num_classes)}, "total": total_val,
                        "pct": {str(c): (cnt_val.get(c, 0) / total_val) * 100 for c in range(num_classes)}}
            dist_path = os.path.join(results_dir, "class_distribution.json")
            with open(dist_path, "w", encoding="utf-8") as f:
                json.dump({"train": train_dist, "val": val_dist}, f, indent=2)
            _print("Class distribution: train total={} pct={}".format(
                train_dist["total"] if train_dist else 0, train_dist["pct"] if train_dist else {}), self.path)
            _print("Class distribution: val total={} pct={}".format(
                val_dist["total"], val_dist["pct"]), self.path)

        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=getattr(self.args, "weight_decay", 0))
        scaler = NativeScaler()
        best_val_f1 = -1.0

        debug_enabled = getattr(self.args, "debug", "") == "enabled"
        debug_param = None
        if debug_enabled:
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.dim() >= 2:
                    debug_param = p
                    _print("DEBUG train: monitor param with shape {}".format(tuple(p.shape)), self.path)
                    break

        for epoch in range(self.args.train_epochs):
            adjust_learning_rate(model_optim, epoch, self.args.learning_rate, self.args)
            current_lr = model_optim.param_groups[0]["lr"]
            self.model.train()
            train_loss_sum = 0.0
            n_batch = 0
            if debug_enabled and debug_param is not None:
                w_before = debug_param.detach().clone()
            pbar = tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, self.args.train_epochs), leave=True, disable=not is_main_process())
            for b_idx, (batch_x, batch_y, meta_list) in enumerate(pbar):
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.long().to(self.device_id)
                 # batch 内正类比例（只在前几个 batch 打印，避免刷屏）
                if debug_enabled and epoch == 0 and b_idx < 5:
                    with torch.no_grad():
                        pos_ratio = (batch_y == 1).float().mean().item()
                    _print("DEBUG train batch {} pos_ratio_cls1 {:.6f}".format(b_idx, pos_ratio), self.path)
                cfg = self._seg_loss_cfg(
                    class_weights_from_train=self._class_weights_tensor.tolist() if self._class_weights_tensor is not None else None
                )
                if self._class_weights_tensor is not None:
                    cfg["class_weights"] = self._class_weights_tensor.tolist()
                    cfg["class_weight"] = "auto"
                model_optim.zero_grad()
                logits = self.model(batch_x, None, None, None, task_id=task_id, task_name="point_segmentation")
                loss = compute_seg_loss(logits, batch_y, cfg)
                loss.backward()
                if getattr(self.args, "clip_grad", None) is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                model_optim.step()
                train_loss_sum += loss.item()
                n_batch += 1
                pbar.set_postfix(loss="{:.4f}".format(train_loss_sum / n_batch))
            train_loss_avg = train_loss_sum / max(n_batch, 1)
            _print("Epoch {} train_loss {:.6f}".format(epoch + 1, train_loss_avg), self.path)
            if debug_enabled and debug_param is not None:
                with torch.no_grad():
                    delta = (debug_param - w_before).abs().mean().item()
                _print("DEBUG epoch {} mean|Δw| on monitored param {:.6e}".format(epoch + 1, delta), self.path)

            val_acc, val_macro_f1, val_spindle_f1 = self.vali(val_loader, task_id, num_classes)
            val_event_f1 = self.vali_event_f1(val_loader, task_id, num_classes, fs=fs, min_len=min_len, merge_gap=merge_gap, iou_thr=iou_thr)
            _print("Epoch {} val acc {:.4f} macro_f1 {:.4f} spindle_f1 {:.4f} event_f1 {:.4f}".format(
                epoch + 1, val_acc, val_macro_f1, val_spindle_f1, val_event_f1 if np.isfinite(val_event_f1) else 0), self.path)

            if is_main_process():
                with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([epoch + 1, train_loss_avg, val_acc, val_macro_f1, val_spindle_f1, val_event_f1 if np.isfinite(val_event_f1) else "", current_lr])
                if writer is not None:
                    writer.add_scalar("train/loss", train_loss_avg, epoch + 1)
                    writer.add_scalar("val/acc", val_acc, epoch + 1)
                    writer.add_scalar("val/macro_f1", val_macro_f1, epoch + 1)
                    writer.add_scalar("val/spindle_f1", val_spindle_f1, epoch + 1)
                    writer.add_scalar("val/event_f1", val_event_f1 if np.isfinite(val_event_f1) else 0, epoch + 1)
                    writer.add_scalar("train/lr", current_lr, epoch + 1)
            if wandb and is_main_process():
                wandb.log({"train_loss": train_loss_avg, "val_acc": val_acc, "val_macro_f1": val_macro_f1, "val_spindle_f1": val_spindle_f1, "val_event_f1": val_event_f1, "lr": current_lr})

            if is_main_process():
                state = self.model.state_dict() if not hasattr(self.model, "module") else self.model.module.state_dict()
                last_ckpt_path = os.path.join(self.path, "last.pth")
                torch.save(state, last_ckpt_path)

            if val_spindle_f1 >= best_val_f1:
                best_val_f1 = val_spindle_f1
                ckpt_path = os.path.join(self.path, "best.pth")
                if is_main_process():
                    state = self.model.state_dict() if not hasattr(self.model, "module") else self.model.module.state_dict()
                    torch.save(state, ckpt_path)
                    _print("Saved best checkpoint to {}".format(ckpt_path), self.path)

        if writer is not None:
            writer.close()
        # 训练结束后自动生成曲线图，无需二次调用 plot_curves.py
        if is_main_process():
            try:
                from tools.plot_curves import plot_curves
                plot_curves(results_dir)
                _print("Curves generated at {}".format(os.path.join(results_dir, "curves")), self.path)
            except Exception as e:
                _print("plot_curves failed: {}".format(e), self.path)
        return self.model

    def vali(self, val_loader, task_id, num_classes):
        self.model.eval()
        debug_enabled = getattr(self.args, "debug", "") == "enabled"
        all_logits = []
        all_y = []
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.long().to(self.device_id)
                logits = self.model(batch_x, None, None, None, task_id=task_id, task_name="point_segmentation")
                all_logits.append(logits.cpu())
                all_y.append(batch_y.cpu())
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        if debug_enabled:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                pos_rate_pred = (pred == 1).float().mean().item()
                pos_rate_gt = (y == 1).float().mean().item()
            _print("DEBUG vali: pos_rate pred_cls1 {:.6f}, gt_cls1 {:.6f}".format(
                pos_rate_pred, pos_rate_gt), self.path)
        return _point_metrics(logits, y, num_classes)

    def vali_event_f1(self, val_loader, task_id, num_classes, fs=256, min_len=1, merge_gap=0, iou_thr=0.3):
        """返回验证集上的 macro event F1（多类平均）。"""
        self.model.eval()
        debug_enabled = getattr(self.args, "debug", "") == "enabled"
        all_logits = []
        all_y = []
        all_meta = []
        with torch.no_grad():
            for batch_x, batch_y, meta_list in val_loader:
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.long().to(self.device_id)
                logits = self.model(batch_x, None, None, None, task_id=task_id, task_name="point_segmentation")
                all_logits.append(logits.cpu())
                all_y.append(batch_y.cpu())
                all_meta.extend(meta_list)
        logits = torch.cat(all_logits, dim=0)
        if debug_enabled and num_classes >= 2:
            with torch.no_grad():
                probs_cls1 = torch.softmax(logits, dim=1)[:, 1, :]
                prob_mean = probs_cls1.mean().item()
                prob_max = probs_cls1.max().item()
            _print("DEBUG vali_event: prob_cls1 mean {:.6f}, max {:.6f}".format(
                prob_mean, prob_max), self.path)
        y = torch.cat(all_y, dim=0)
        pred_labels = logits.argmax(dim=1).numpy()
        gt_all = y.numpy()
        macro_f1_list = []
        for i in range(pred_labels.shape[0]):
            meta = all_meta[i] if i < len(all_meta) else {}
            fs_i = meta.get("fs", fs)
            ev = compute_event_metrics_multiclass(
                pred_labels[i], gt_all[i], fs_i, num_classes,
                min_len=min_len, merge_gap=merge_gap, iou_thr=iou_thr)
            if np.isfinite(ev.get("macro_event_f1", np.nan)):
                macro_f1_list.append(ev["macro_event_f1"])
        return float(np.nanmean(macro_f1_list)) if macro_f1_list else float("nan")

    def test(self, setting, load_ckpt=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        config = self.task_data_config_list[0][1]
        task_id = 0
        num_classes = config.get("num_classes", 2)
        fs = config.get("fs", 256)
        min_len = config.get("min_event_len", 1)
        merge_gap = config.get("merge_gap", 0)
        iou_thr = config.get("iou_thr", 0.3)
        test_set, test_loader = self._get_data("test")

        if load_ckpt is None:
            load_ckpt = os.path.join(self.path, "best.pth")
        if os.path.isfile(load_ckpt):
            ckpt = _safe_torch_load(load_ckpt, map_location="cuda:{}".format(self.device_id))
            model = self.model.module if hasattr(self.model, "module") else self.model
            model.load_state_dict(ckpt, strict=False)
            _print("Loaded checkpoint {}".format(load_ckpt), self.path)

        self.model.eval()
        all_logits = []
        all_y = []
        all_meta = []
        first_batch = None
        with torch.no_grad():
            for batch_x, batch_y, meta_list in test_loader:
                if first_batch is None:
                    first_batch = (batch_x.float().to(self.device_id), batch_y, meta_list)
                batch_x = batch_x.float().to(self.device_id)
                logits = self.model(batch_x, None, None, None, task_id=task_id, task_name="point_segmentation")
                all_logits.append(logits.cpu())
                all_y.append(batch_y)
                all_meta.extend(meta_list)
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        acc, macro_f1, spindle_f1 = _point_metrics(logits, y, num_classes)

        # 事件级指标（多类兼容）与逐样本指标（供 analysis / best-worst 使用）
        pred_labels = logits.argmax(dim=1).numpy()
        gt_all = y.numpy()
        event_list = []
        per_sample_rows = []
        from utils.event_metrics import compute_event_metrics_multiclass
        for i in range(logits.shape[0]):
            meta = all_meta[i] if i < len(all_meta) else {}
            fs_i = meta.get("fs", fs)
            ev = compute_event_metrics_multiclass(pred_labels[i], gt_all[i], fs_i, num_classes,
                                                  min_len=min_len, merge_gap=merge_gap, iou_thr=iou_thr)
            event_list.append(ev)
            pc = ev.get("per_class", {})
            event_f1_1 = pc.get("1", {}).get("event_f1", float("nan"))
            mbe_1 = pc.get("1", {}).get("mean_boundary_error_ms", float("nan"))
            per_sample_rows.append({
                "sample_idx": i,
                "excerpt_id": meta.get("excerpt_id", "unknown"),
                "event_f1": ev.get("macro_event_f1", event_f1_1),
                "event_f1_cls1": event_f1_1,
                "mean_boundary_error_ms": mbe_1 if np.isfinite(mbe_1) else "",
            })

        # 聚合 per_class：这里简单平均各样本的 per_class 指标
        per_class_agg = {}
        macro_event_f1_list = []
        for ev in event_list:
            per_c = ev.get("per_class", {})
            for cid, md in per_c.items():
                if cid not in per_class_agg:
                    per_class_agg[cid] = {k: [] for k in md.keys()}
                for k, v in md.items():
                    per_class_agg[cid][k].append(v)
            if np.isfinite(ev.get("macro_event_f1", np.nan)):
                macro_event_f1_list.append(ev["macro_event_f1"])

        per_class_final = {}
        for cid, md in per_class_agg.items():
            per_class_final[cid] = {k: float(np.nanmean(v)) if v else float("nan") for k, v in md.items()}
        overall_macro_event_f1 = float(np.nanmean(macro_event_f1_list)) if macro_event_f1_list else float("nan")

        results = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "spindle_f1": spindle_f1,
            "per_class": per_class_final,
            "overall_macro_event_f1": overall_macro_event_f1,
        }
        _print("Test point: acc {:.4f} macro_f1 {:.4f} spindle_f1 {:.4f}".format(acc, macro_f1, spindle_f1), self.path)
        _print("Test event (macro over classes): macro_event_f1 {:.4f}".format(
            overall_macro_event_f1 if np.isfinite(overall_macro_event_f1) else float("nan")), self.path)

        results_dir = os.path.join("results", setting)
        os.makedirs(results_dir, exist_ok=True)
        vis_dir = os.path.join(results_dir, "vis")
        analysis_dir = os.path.join(results_dir, "analysis")
        cases_best_dir = os.path.join(results_dir, "cases", "best_20")
        cases_worst_dir = os.path.join(results_dir, "cases", "worst_20")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(cases_best_dir, exist_ok=True)
        os.makedirs(cases_worst_dir, exist_ok=True)

        with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump({**results, "config_summary": {"num_classes": num_classes, "fs": fs}}, f, indent=2)
        with open(os.path.join(results_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(results.keys()))
            w.writerow([results[k] for k in results.keys()])

        if per_sample_rows and is_main_process():
            ps_path = os.path.join(results_dir, "per_sample_metrics.csv")
            with open(ps_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["sample_idx", "excerpt_id", "event_f1", "event_f1_cls1", "mean_boundary_error_ms"])
                w.writeheader()
                w.writerows(per_sample_rows)

        if is_main_process():
            np.save(os.path.join(results_dir, "pred_labels.npy"), pred_labels)
            np.save(os.path.join(results_dir, "gt_labels.npy"), gt_all)
            probs_cls1 = torch.softmax(logits, dim=1)[:, 1, :].numpy()
            np.save(os.path.join(results_dir, "probs_cls1.npy"), probs_cls1)

        if first_batch is not None and is_main_process():
            batch_x_vis, batch_y_vis, meta_vis_list = first_batch
            n_vis = min(batch_x_vis.size(0), 20)
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from tools.plot_segmentation import run_visualization
            try:
                run_visualization(
                    self.model, self.device_id,
                    batch_x_vis, batch_y_vis, meta_vis_list,
                    task_id, num_classes=num_classes, fs=fs,
                    min_len=min_len, merge_gap=merge_gap,
                    save_dir=vis_dir, sample_indices=None,
                    max_save=20, compute_saliency=True,
                )
            except Exception as e:
                _print("run_visualization failed: {}".format(e), self.path)
            with open(os.path.join(results_dir, "vis_info.json"), "w", encoding="utf-8") as f:
                json.dump({"n_vis": n_vis, "has_vis_sample_idx_range": [0, n_vis - 1]}, f, indent=2)

        if is_main_process() and per_sample_rows:
            try:
                from tools.analyze_results import run_analysis
                run_analysis(results_dir, num_classes=num_classes, fs=fs)
            except Exception as e:
                _print("analyze_results failed: {}".format(e), self.path)

        _print("Results and vis saved to {}".format(results_dir), self.path)
        return results
