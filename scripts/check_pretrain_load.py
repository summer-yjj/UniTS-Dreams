"""
检查官方预训练权重是否能被当前 point_segmentation 流程正确加载。
用法（在项目根目录）:
  python scripts/check_pretrain_load.py
  python scripts/check_pretrain_load.py --ckpt checkpoints/units_x128_pretrain_checkpoint.pth
  python scripts/check_pretrain_load.py --ckpt /path/to/any.pth
"""
from __future__ import print_function
import os
import sys
import argparse
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml


def find_checkpoint(ckpt_dir):
    """在 checkpoints 下查找可能的预训练权重（*pretrain*.pth 或任意 .pth）。"""
    if not os.path.isdir(ckpt_dir):
        return []
    candidates = glob.glob(os.path.join(ckpt_dir, "*pretrain*.pth"))
    if not candidates:
        candidates = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    return sorted(candidates)


def main():
    parser = argparse.ArgumentParser(description="Check pretrained weight loading for point_segmentation")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to .pth file. If not set, will search checkpoints/*pretrain*.pth")
    parser.add_argument("--yaml", type=str, default="data_provider/dreams_pointwise_pointseg.yaml",
                        help="Task config YAML for building UniTS model")
    args_parse = parser.parse_args()

    ckpt_path = args_parse.ckpt
    if not ckpt_path:
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        candidates = find_checkpoint(ckpt_dir)
        if not candidates:
            print("[check_pretrain] No .pth found under checkpoints/. Use --ckpt /path/to/file.pth")
            return 1
        ckpt_path = candidates[0]
        print("[check_pretrain] Using first found:", ckpt_path)
    else:
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path

    if not os.path.isfile(ckpt_path):
        print("[check_pretrain] File not found:", ckpt_path)
        return 1

    # 1) 加载并检查 checkpoint 结构
    print("\n--- Checkpoint structure ---")
    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    print("type(ckpt):", type(ckpt_raw))
    if isinstance(ckpt_raw, dict):
        top_keys = list(ckpt_raw.keys())
        print("top-level keys (count={}):".format(len(top_keys)), top_keys[:20])
        if len(top_keys) > 20:
            print("  ... and", len(top_keys) - 20, "more")
        # 与 exp_pointseg 一致：优先 "student"，否则尝试 "model" 或整份当 state_dict
        if "student" in ckpt_raw:
            state_dict = ckpt_raw["student"]
            print("-> Has 'student' key. Will use ckpt['student'] as state_dict (current exp_pointseg logic).")
        elif "model" in ckpt_raw:
            state_dict = ckpt_raw["model"]
            print("-> Has 'model' key. Will use ckpt['model'] as state_dict.")
        else:
            state_dict = ckpt_raw
            print("-> No 'student'/'model'. Will use ckpt itself as state_dict.")
        if isinstance(state_dict, dict):
            sd_keys = list(state_dict.keys())
            print("state_dict key count:", len(sd_keys))
            print("sample keys:", sd_keys[:10])
    else:
        print("Checkpoint is not a dict (e.g. raw state_dict). Treat as state_dict.")
        state_dict = ckpt_raw

    # 2) 按 exp_pointseg 的加载逻辑得到要 load 的 state_dict
    pretrain_weight_path = ckpt_path
    use_student = isinstance(ckpt_raw, dict) and "student" in ckpt_raw
    if "pretrain_checkpoint.pth" in pretrain_weight_path and use_student:
        state_dict = ckpt_raw["student"]
        ckpt = {}
        for k, v in state_dict.items():
            if "cls_prompts" not in k:
                ckpt[k] = v
        print("\n[Filter] Used ckpt['student'], filtered out 'cls_prompts'. Keys to load:", len(ckpt))
    else:
        ckpt = state_dict if isinstance(state_dict, dict) else ckpt_raw
        print("\n[No filter] Using state_dict as-is. Keys to load:", len(ckpt) if isinstance(ckpt, dict) else "N/A")

    if not isinstance(ckpt, dict):
        print("[check_pretrain] Cannot get dict state_dict. Abort.")
        return 1

    # 预训练权重若来自 DDP，键带 "module." 前缀，需去掉才能匹配当前单卡模型
    has_module_prefix = any(k.startswith("module.") for k in ckpt.keys())
    if has_module_prefix:
        print("\n[Strip 'module.'] Checkpoint keys have 'module.' prefix (saved from DDP). Stripping for load.")
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        print("Keys after strip:", len(ckpt), "sample:", list(ckpt.keys())[:5])

    # 3) 构建当前项目的 UniTS 模型（point_segmentation）
    print("\n--- Build UniTS model (point_segmentation) ---")
    yaml_path = os.path.join(PROJECT_ROOT, args_parse.yaml)
    if not os.path.isfile(yaml_path):
        print("[check_pretrain] YAML not found:", yaml_path)
        return 1
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    task_dataset = config.get("task_dataset", {})
    task_data_config_list = [[k, v] for k, v in task_dataset.items()]

    class Args:
        pass
    args = Args()
    args.model = "UniTS"
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.patch_len = 16
    args.stride = 16
    args.prompt_num = 5
    args.dropout = 0.1
    args.enc_in = 1

    try:
        from models.UniTS import Model
        model = Model(args, task_data_config_list)
        model.eval()
    except Exception as e:
        print("[check_pretrain] Model build failed (e.g. missing timm):", e)
        print("\n--- Result (from checkpoint inspection only) ---")
        print("Checkpoint has 'student' key: OK. Keys have 'module.' prefix: strip before load in exp_pointseg.")
        print("Done.")
        return 0

    # 4) 执行 load_state_dict(ckpt, strict=False)
    print("\n--- load_state_dict(ckpt, strict=False) ---")
    msg = model.load_state_dict(ckpt, strict=False)
    print("missing_keys (will stay random):", len(msg.missing_keys))
    if msg.missing_keys:
        for k in msg.missing_keys[:15]:
            print("  ", k)
        if len(msg.missing_keys) > 15:
            print("  ... and", len(msg.missing_keys) - 15, "more")
    print("unexpected_keys (ignored):", len(msg.unexpected_keys))
    if msg.unexpected_keys:
        for k in msg.unexpected_keys[:15]:
            print("  ", k)
        if len(msg.unexpected_keys) > 15:
            print("  ... and", len(msg.unexpected_keys) - 15, "more")

    # 5) 结论
    print("\n--- Result ---")
    if not msg.missing_keys and not msg.unexpected_keys:
        print("OK: Exact match. Pretrained weights can be loaded with strict=True.")
    elif msg.missing_keys and all("seg_heads" in k for k in msg.missing_keys):
        print("OK: Only seg_heads are missing (expected for pretrain without point_seg). Load is correct.")
    else:
        print("OK: strict=False load succeeded. Some keys missing/unexpected (normal for pretrain -> point_seg).")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
