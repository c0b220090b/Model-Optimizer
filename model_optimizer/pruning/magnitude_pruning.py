"""
2-1. Magnitude / SparseGPT Pruning（NVIDIA TensorRT Model Optimizer を使用, 2:4構造化スパース）
nvidia-modelopt の modelopt.torch.sparsity を用いて、4要素ごとに2つをゼロ化する
Ampere+ GPU 向けのハードウェアアクセラレーション対応パターン (2:4 sparsity) を生成する。

--mode
  sparse_magnitude : 重み絶対値のみでマスクを決める（データ不要。キャリブレーションは使わない）
  sparsegpt        : キャリブレーションデータ（WikiText-2 train）の活性化を使ってマスクを決め、
                     残りの重みを補正する（Frantar & Alistarh, 2023）。
評価は WikiText-2 test（キャリブレーションと重ならない）で、seqlen=2048 の Perplexity を測る。
（注）modelopt の重みスパース化は 2:4 パターン固定のため、任意のスパース率は指定できない。
任意スパース率が必要な場合は `--backend custom` で従来の閾値ベース実装にフォールバックする。

事前インストール:
  pip install nvidia-modelopt[hf] datasets

使用例:
  python pruning/magnitude_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --output_dir ./output/tinyllama_v1.1-modelopt-magnitude-2to4

  python pruning/magnitude_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --mode sparsegpt --num_calib_samples 128 \
    --output_dir ./output/tinyllama_v1.1-modelopt-sparsegpt-2to4
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_sparsity, get_linear_sparsity, get_model_size_mb,
                          measure_llm_latency, measure_perplexity_wikitext, get_calib_dataset,
                          save_results, common_output_args, common_eval_args)
import torch
import torch.nn as nn


@torch.no_grad()
def _custom_magnitude_prune_linear(module: nn.Linear, sparsity: float):
    """任意スパース率が必要な場合の従来実装（modeloptは2:4固定のためのフォールバック）"""
    w = module.weight.data
    num_prune = int(w.numel() * sparsity)
    if num_prune <= 0:
        return
    threshold = w.abs().flatten().kthvalue(num_prune).values
    mask = w.abs() > threshold
    module.weight.data = w * mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--backend", type=str, default="modelopt", choices=["modelopt", "custom"])
    p.add_argument("--mode", type=str, default="sparse_magnitude",
                   choices=["sparse_magnitude", "sparsegpt"],
                   help="--backend modelopt 時のみ使用。sparsegpt はキャリブレーションデータを使う")
    p.add_argument("--num_calib_samples", type=int, default=128,
                   help="sparsegpt 用キャリブレーションのサンプル数（各 seqlen トークン）")
    p.add_argument("--sparsity", type=float, default=0.5, help="--backend custom 時のみ使用")
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device
    uses_calib = args.backend == "modelopt" and args.mode == "sparsegpt"

    if args.backend == "modelopt":
        import modelopt.torch.sparsity as mts

        if uses_calib:
            print("🤖 キャリブレーションデータ（WikiText-2 train）を準備中...")
            calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.seqlen,
                                              device=device, seed=args.seed)
            print(f"✅ {args.seqlen} トークンのサンプルを {len(calib_dataset)} 個作成しました。")
            config = {
                "data_loader": calib_dataset,
                # 各バッチ(dict)をそのまま model(**batch) に渡す
                "collect_func": lambda batch: batch,
            }
            print("⚡ SparseGPT (modelopt) でマスク探索と重み補正を実行中...")
            model = mts.sparsify(model, mode="sparsegpt", config=config)
        else:
            # sparse_magnitude は重みの絶対値だけで決まるため、キャリブレーションは不要
            print("⚡ Magnitude (modelopt) で 2:4 マスクを探索中...")
            model = mts.sparsify(model, mode="sparse_magnitude")

        # スパースモジュールのラッパーを外し、マスクを重みに焼き込んで通常の nn.Linear に戻す
        model = mts.export(model)
        method_name = f"modelopt_{args.mode}_2to4"
    else:
        target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
                _custom_magnitude_prune_linear(module, args.sparsity)
        method_name = "custom_magnitude_pruning"

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("📊 性能ベンチマークを測定中（PPL: WikiText-2 test）...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                      max_chunks=args.eval_max_chunks)

    results = {
        "method": method_name,
        "backend": args.backend,
        "mode": args.mode if args.backend == "modelopt" else None,
        "calib_data": f"wikitext2-train x{args.num_calib_samples}" if uses_calib else None,
        "eval_data": f"wikitext2-test seqlen={args.seqlen} max_chunks={args.eval_max_chunks}",
        "actual_sparsity": get_sparsity(model),
        "linear_sparsity": get_linear_sparsity(model),
        "model_size_mb": get_model_size_mb(model),
        "perplexity": ppl,
        **latency,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()