# 分割可视化：信号 + GT + Pred prob/mask + 事件边界；可选 Saliency
from __future__ import division
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 项目根加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.event_post import pred_to_events, gt_to_events


def plot_one(
    signal,
    gt_mask,
    pred_prob,
    pred_mask,
    fs=256,
    window_start=0,
    window_end=None,
    title_prefix="",
    save_path=None,
    pred_events=None,
    gt_events=None,
):
    """
    signal: [T], gt_mask: [T] 0/1, pred_prob: [T] spindle prob, pred_mask: [T] 0/1.
    pred_events / gt_events: list of (s,e) for vertical lines (optional).
    """
    T = len(signal)
    if window_end is None:
        window_end = T
    t = np.arange(T) / fs if fs else np.arange(T)
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    axes[0].plot(t, signal, color="k", linewidth=0.6)
    axes[0].set_ylabel("Signal")
    axes[0].set_title("{} Signal (fs={}, start={}, end={})".format(
        title_prefix, fs, window_start, window_end))
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(t, 0, gt_mask, color="green", alpha=0.5, label="GT")
    axes[1].set_ylabel("GT mask")
    axes[1].set_ylim(-0.1, 1.2)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, pred_prob, color="blue", linewidth=0.8, label="Pred prob(spindle)")
    axes[2].set_ylabel("Pred prob")
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].fill_between(t, 0, pred_mask, color="red", alpha=0.5, label="Pred mask")
    if pred_events:
        for s, e in pred_events:
            axes[3].axvline(s / fs if fs else s, color="red", linestyle="--", alpha=0.8)
            axes[3].axvline((e - 1) / fs if fs else (e - 1), color="red", linestyle="--", alpha=0.8)
    if gt_events:
        for s, e in gt_events:
            axes[3].axvline(s / fs if fs else s, color="green", linestyle=":", alpha=0.8)
            axes[3].axvline((e - 1) / fs if fs else (e - 1), color="green", linestyle=":", alpha=0.8)
    axes[3].set_ylabel("Pred mask")
    axes[3].set_xlabel("Time (s)" if fs else "Sample")
    axes[3].set_ylim(-0.1, 1.2)
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return fig


def plot_saliency(t, signal, saliency, save_path=None, title_prefix="Saliency"):
    """|grad| vs time, same time axis as signal."""
    fig, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(t, signal, color="k", linewidth=0.5, alpha=0.7, label="Signal")
    ax1.set_ylabel("Signal")
    ax2 = ax1.twinx()
    ax2.plot(t, saliency, color="orange", linewidth=0.8, alpha=0.8, label="|grad|")
    ax2.set_ylabel("|grad|")
    ax2.legend(loc="upper right")
    ax1.legend(loc="upper left")
    ax1.set_xlabel("Time (s)" if (t[-1] - t[0]) > 10 else "Sample")
    ax1.set_title(title_prefix)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def run_visualization(
    model,
    device,
    batch_x,
    batch_y,
    meta_list,
    task_id,
    num_classes=2,
    fs=256,
    min_len=1,
    merge_gap=0,
    save_dir=None,
    sample_indices=None,
    max_save=20,
    compute_saliency=True,
):
    """
    batch_x [B,C,T], batch_y [B,T], meta_list list of dict.
    保存至 save_dir，最多 max_save 张图；若 compute_saliency 则对同一样本再存 saliency 图。
    """
    import torch.nn.functional as F

    model.eval()
    B = batch_x.shape[0]
    if sample_indices is None:
        sample_indices = list(range(min(B, max_save)))
    else:
        sample_indices = [i for i in sample_indices if i < B][:max_save]
    if not sample_indices:
        return

    with torch.no_grad():
        logits = model(batch_x, None, None, None, task_id=task_id, task_name="point_segmentation")
    probs = F.softmax(logits, dim=1)
    pred_class = logits.argmax(dim=1)

    os.makedirs(save_dir, exist_ok=True)
    for idx in sample_indices:
        x = batch_x[idx]
        y = batch_y[idx]
        sig = x[0].cpu().numpy() if x.dim() >= 2 else x.cpu().numpy()
        gt = y.cpu().numpy()
        pred = pred_class[idx].cpu().numpy()
        meta = meta_list[idx] if idx < len(meta_list) else {}
        ws = meta.get("window_start", 0)
        we = meta.get("window_end", len(sig))
        excerpt_id = meta.get("excerpt_id", str(idx))
        t = np.arange(len(sig)) / fs

        # num_classes==2: 保持原有表现（类1为 spindle）
        if num_classes <= 2:
            prob = probs[idx, 1].cpu().numpy() if num_classes == 2 else probs[idx, 0].cpu().numpy()
            gt_mask = (gt == 1).astype(np.float32)
            pred_mask = (pred == 1).astype(np.float32)
            pred_ev = pred_to_events(pred_mask, min_len=min_len, merge_gap=merge_gap)
            gt_ev = gt_to_events(gt, positive_class=1, min_len=1, merge_gap=0)
            fname = "seg_{}_{}.png".format(excerpt_id, idx)
            plot_one(
                sig, gt_mask, prob, pred_mask,
                fs=fs, window_start=ws, window_end=we,
                title_prefix="{} idx={}".format(excerpt_id, idx),
                save_path=os.path.join(save_dir, fname),
                pred_events=pred_ev,
                gt_events=gt_ev,
            )
            # Saliency 针对 spindle 类
            if compute_saliency:
                one_x = batch_x[idx : idx + 1].detach().clone().requires_grad_(True)
                logits_g = model(one_x, None, None, None, task_id=task_id, task_name="point_segmentation")
                target = logits_g[:, 1].sum() if num_classes >= 2 else logits_g[:, 0].sum()
                model.zero_grad()
                target.backward()
                if one_x.grad is not None:
                    sal = one_x.grad.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
                    t_sal = np.arange(len(sal)) / fs
                    plot_saliency(
                        t_sal, sig,
                        sal,
                        save_path=os.path.join(save_dir, "saliency_{}_{}.png".format(excerpt_id, idx)),
                        title_prefix="Saliency {} idx={}".format(excerpt_id, idx),
                    )
        else:
            # 多类：对每个 class_id>=1 单独画一张图
            for c in range(1, num_classes):
                gt_mask_c = (gt == c).astype(np.float32)
                pred_mask_c = (pred == c).astype(np.float32)
                prob_c = probs[idx, c].cpu().numpy()
                pred_ev = pred_to_events(pred_mask_c, min_len=min_len, merge_gap=merge_gap)
                gt_ev = gt_to_events(gt, positive_class=c, min_len=1, merge_gap=0)
                fname = "seg_cls{}_{}_{}.png".format(c, excerpt_id, idx)
                plot_one(
                    sig, gt_mask_c, prob_c, pred_mask_c,
                    fs=fs, window_start=ws, window_end=we,
                    title_prefix="class {} {} idx={}".format(c, excerpt_id, idx),
                    save_path=os.path.join(save_dir, fname),
                    pred_events=pred_ev,
                    gt_events=gt_ev,
                )
                if compute_saliency:
                    one_x = batch_x[idx : idx + 1].detach().clone().requires_grad_(True)
                    logits_g = model(one_x, None, None, None, task_id=task_id, task_name="point_segmentation")
                    target = logits_g[:, c].sum()
                    model.zero_grad()
                    target.backward()
                    if one_x.grad is not None:
                        sal = one_x.grad.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
                        t_sal = np.arange(len(sal)) / fs
                        plot_saliency(
                            t_sal, sig,
                            sal,
                            save_path=os.path.join(save_dir, "saliency_cls{}_{}_{}.png".format(c, excerpt_id, idx)),
                            title_prefix="Saliency class {} {} idx={}".format(c, excerpt_id, idx),
                        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task_data_config_path", type=str, default="data_provider/dreams_pointseg.yaml")
    parser.add_argument("--model_id", type=str, default="dreams_pointseg_units_baseline")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/vis_sample.png")
    parser.add_argument("--saliency", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.task_data_config_path, "r") as f:
        config = yaml.safe_load(f)
    task_config = list(config["task_dataset"].values())[0]
    task_config["data"] = "dreams_pointseg"
    from data_provider.data_factory import data_provider
    from exp.exp_pointseg import Exp_PointSeg
    from argparse import Namespace
    a = Namespace(
        batch_size=1, num_workers=0, subsample_pct=None, fix_seed=None,
        task_data_config_path=args.task_data_config_path, debug="disabled",
    )
    _, loader = data_provider(a, task_config, "test")
    batch_x, batch_y, meta_list = next(iter(loader))
    batch_x = batch_x.cuda()
    batch_y = batch_y.cuda()
    from models.UniTS import Model
    from argparse import Namespace as NS
    ar = NS(
        d_model=512, n_heads=8, e_layers=2, prompt_num=5, patch_len=16, stride=16,
        dropout=0.1, enc_in=1, num_class=2,
    )
    configs_list = [[list(config["task_dataset"].keys())[0], task_config]]
    model = Model(ar, configs_list).cuda()
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    task_id = 0
    with torch.no_grad():
        logits = model(batch_x, None, None, None, task_id=task_id, task_name="point_segmentation")
    probs = torch.softmax(logits, dim=1)
    pred_mask = logits.argmax(dim=1)
    idx = args.sample_index
    sig = batch_x[idx, 0].cpu().numpy()
    gt = batch_y[idx].cpu().numpy()
    prob = probs[idx, 1].cpu().numpy()
    pred = pred_mask[idx].cpu().numpy()
    meta = meta_list[idx]
    fs = task_config.get("fs", 256)
    pred_ev = pred_to_events(pred, 1, 0)
    gt_ev = gt_to_events(gt, 1, 1, 0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_one(sig, gt, prob, pred, fs=fs,
             window_start=meta.get("window_start", 0), window_end=meta.get("window_end"),
             title_prefix="{} idx={}".format(meta.get("excerpt_id", idx), idx),
             save_path=args.out, pred_events=pred_ev, gt_events=gt_ev)
    print("Saved", args.out)
    if args.saliency:
        batch_x_g = batch_x.detach().clone().requires_grad_(True)
        logits_g = model(batch_x_g, None, None, None, task_id=task_id, task_name="point_segmentation")
        logits_g[:, 1].sum().backward()
        sal = batch_x_g.grad.abs().sum(dim=1).squeeze(0).cpu().numpy()
        t = np.arange(len(sal)) / fs
        sal_path = args.out.replace(".png", "_saliency.png")
        plot_saliency(t, sig, sal, save_path=sal_path, title_prefix="Saliency")
        print("Saved", sal_path)


if __name__ == "__main__":
    main()
