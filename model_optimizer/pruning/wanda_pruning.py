"""
2-2. Wanda Pruning（非構造化 / N:M, 活性化考慮, 再学習不要）
重みの大きさと、対応する入力活性化のノルムの積 (|W| * ||X||) を重要度指標として
使用する枝刈り手法 (Sun et al., 2023)。SparseGPTと異なり再学習・ヘッシアン計算が不要。
キャリブレーションは WikiText-2 train、評価は WikiText-2 test（重ならない）で行う。

（簡易実装の注意）原論文は層を先頭から順に刈り、刈った後の出力を次の層の入力に使う。
本実装は全層の活性化統計を dense モデルで一度に集める one-shot 版。

使用例:
  # 非構造化 50%
  python pruning/wanda_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --sparsity 0.5 \
    --output_dir ./output/tinyllama-wanda-50

  # 2:4（SparseGPT 2:4 と比較する場合）
  python pruning/wanda_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --structure 2:4 \
    --output_dir ./output/tinyllama-wanda-2to4
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_sparsity, get_linear_sparsity, get_model_size_mb,
                          measure_llm_latency, measure_perplexity_wikitext, get_calib_dataset,
                          run_calibration, save_results, common_output_args, common_eval_args)
import torch
import torch.nn as nn

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class ActNormCollector:
    """各 Linear 層の入力について、チャンネルごとの二乗和を fp32 で蓄積する"""
    def __init__(self):
        self.sq_sum = {}
        self.count = {}

    def hook(self, name):
        def fn(module, inp, out):
            # fp16 のまま二乗すると外れ値チャネル（数百〜数千）で 65504 を超えて inf になるため fp32 で計算
            x = inp[0].detach().float()
            x = x.reshape(-1, x.shape[-1])  # [tokens, in_features]
            sq = (x ** 2).sum(dim=0)
            if name in self.sq_sum:
                self.sq_sum[name] += sq
                self.count[name] += x.shape[0]
            else:
                self.sq_sum[name] = sq
                self.count[name] = x.shape[0]
        return fn


def parse_structure(s: str):
    """'unstructured' -> None, '2:4' -> (2, 4)"""
    if s == "unstructured":
        return None
    n, m = s.split(":")
    return int(n), int(m)


@torch.no_grad()
def wanda_prune(model, calib_dataset, sparsity: float, nm=None, target_modules=TARGET_MODULES):
    collector = ActNormCollector()
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            hooks.append(module.register_forward_hook(collector.hook(name)))
    try:
        run_calibration(model, calib_dataset)
    finally:
        for h in hooks:
            h.remove()

    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and name in collector.sq_sum):
            continue
        # 行内の順位だけが重要なので RMS でも L2 ノルムでも結果は同じ
        act_norm = torch.sqrt(collector.sq_sum[name] / max(collector.count[name], 1))  # [in]
        w = module.weight.data  # [out, in]
        importance = w.float().abs() * act_norm.unsqueeze(0).to(w.device)  # |W| * ||X||
        mask = torch.ones_like(w, dtype=torch.bool)

        if nm is None:
            # 出力ニューロンごと(行ごと)に同じ割合を刈る（Wanda の推奨設定）
            num_prune = int(w.shape[1] * sparsity)
            if num_prune <= 0:
                continue
            _, prune_idx = torch.topk(importance, num_prune, dim=1, largest=False)
            mask.scatter_(1, prune_idx, False)
        else:
            # N:M — 入力方向に連続する M 個ごとに、重要度の低い N 個を刈る
            n, m = nm
            if w.shape[1] % m != 0:
                raise ValueError(f"{name}: in_features={w.shape[1]} が M={m} で割り切れません")
            imp = importance.reshape(w.shape[0], -1, m)
            _, prune_idx = torch.topk(imp, n, dim=-1, largest=False)
            mask = mask.reshape(w.shape[0], -1, m)
            mask.scatter_(-1, prune_idx, False)
            mask = mask.reshape_as(w)

        module.weight.data = w * mask
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--structure", type=str, default="unstructured",
                   help="'unstructured'（--sparsity を使用）または '2:4' / '4:8' などの N:M")
    p.add_argument("--sparsity", type=float, default=0.5, help="unstructured 時のみ使用")
    p.add_argument("--num_calib_samples", type=int, default=128)
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()
    nm = parse_structure(args.structure)

    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device

    print("🤖 キャリブレーションデータ（WikiText-2 train）を準備中...")
    calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.seqlen,
                                      device=device, seed=args.seed)
    print(f"⚡ Wanda ({args.structure}) を実行中...")
    wanda_prune(model, calib_dataset, args.sparsity, nm)
    del calib_dataset

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("📊 性能ベンチマークを測定中（PPL: WikiText-2 test）...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                      max_chunks=args.eval_max_chunks)

    results = {
        "method": "wanda",
        "structure": args.structure,
        "target_sparsity": args.sparsity if nm is None else nm[0] / nm[1],
        "calib_data": f"wikitext2-train x{args.num_calib_samples}",
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