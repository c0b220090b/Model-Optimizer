"""
2-3. SparseGPT Pruning（NVIDIA TensorRT Model Optimizer を使用, 2:4構造化スパース）
nvidia-modelopt の modelopt.torch.sparsity ("sparsegpt" モード) を用いる。
ヘッシアン近似(OBS)による重み更新付きの2:4構造化スパース化で、Frantar & Alistarh (2023)
のアルゴリズムを modelopt が実装したもの。
（注）modelopt 版は2:4パターン固定。任意スパース率が必要な場合は `--backend custom` で
Hessian近似ベースの自前実装（ブロック単位, 任意スパース率対応）にフォールバックする。
キャリブレーションは WikiText-2 train、評価は WikiText-2 test（重ならない）で行う。

（custom の注意）全 Linear 層のヘッシアン (in×in, fp32) を同時に GPU に保持する one-shot 実装。
TinyLlama なら約 5GB で収まるが、7B 級では 40GB 近くになる。原論文は層ごとに逐次処理する。

事前インストール:
  pip install nvidia-modelopt[hf] datasets

使用例:
  python pruning/sparsegpt_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --num_calib_samples 128 \
    --output_dir ./output/tinyllama-modelopt-sparsegpt-2to4

  python pruning/sparsegpt_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --backend custom --sparsity 0.5 \
    --output_dir ./output/tinyllama-custom-sparsegpt-50
"""
import sys, os, argparse, math
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_sparsity, get_linear_sparsity, get_model_size_mb,
                          measure_llm_latency, measure_perplexity_wikitext, get_calib_dataset,
                          run_calibration, save_results, common_output_args, common_eval_args)
import torch
import torch.nn as nn

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class _CustomSparseGPTLayerPruner:
    """任意スパース率が必要な場合のフォールバック実装（論文アルゴリズムの簡易再現）"""

    def __init__(self, layer: nn.Linear, blocksize: int = 128, percdamp: float = 0.01):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0
        self.blocksize = blocksize
        self.percdamp = percdamp

    def add_batch(self, inp: torch.Tensor):
        inp = inp.reshape(-1, inp.shape[-1]).float()
        n = inp.shape[0]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = inp * math.sqrt(2 / self.nsamples)
        self.H += inp.t() @ inp

    @torch.no_grad()
    def prune(self, sparsity: float):
        W = self.layer.weight.data.clone().float()
        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        mask_final = torch.zeros_like(W, dtype=torch.bool)
        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            tmp = (W1 ** 2) / (torch.diag(Hinv1).reshape(1, -1) ** 2 + 1e-10)
            thresh_idx = int(count * sparsity)
            mask1 = torch.ones_like(W1, dtype=torch.bool)
            if thresh_idx > 0:
                sorted_idx = torch.argsort(tmp, dim=1)
                mask1.scatter_(1, sorted_idx[:, :thresh_idx], False)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[~mask1[:, i]] = 0
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            mask_final[:, i1:i2] = mask1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        self.layer.weight.data = (W * mask_final).to(self.layer.weight.dtype)


def _custom_sparsegpt(model, calib_dataset, sparsity, target_modules=TARGET_MODULES):
    target_layers = {n: m for n, m in model.named_modules()
                     if isinstance(m, nn.Linear) and any(t in n for t in target_modules)}
    pruners = {n: _CustomSparseGPTLayerPruner(m) for n, m in target_layers.items()}

    def make_hook(name):
        def fn(module, inp, out):
            pruners[name].add_batch(inp[0].detach())
        return fn

    hooks = [m.register_forward_hook(make_hook(n)) for n, m in target_layers.items()]
    try:
        run_calibration(model, calib_dataset)
    finally:
        for h in hooks:
            h.remove()
    for pruner in pruners.values():
        pruner.prune(sparsity)
        torch.cuda.empty_cache()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--backend", type=str, default="modelopt", choices=["modelopt", "custom"])
    p.add_argument("--sparsity", type=float, default=0.5, help="--backend custom 時のみ使用")
    p.add_argument("--num_calib_samples", type=int, default=128)
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device

    print("🤖 キャリブレーションデータ（WikiText-2 train）を準備中...")
    calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.seqlen,
                                      device=device, seed=args.seed)

    if args.backend == "modelopt":
        import modelopt.torch.sparsity as mts
        print("⚡ SparseGPT (modelopt, 2:4) を実行中...")
        # magnitude_pruning.py --mode sparsegpt で動作確認済みの渡し方
        config = {"data_loader": calib_dataset, "collect_func": lambda batch: batch}
        model = mts.sparsify(model, mode="sparsegpt", config=config)
        # ラッパーを外してマスクを重みに焼き込む（save_pretrained / スパース率計測のため）
        model = mts.export(model)
        method_name = "modelopt_sparsegpt_2to4"
    else:
        print(f"⚡ SparseGPT (custom, {args.sparsity:.0%}) を実行中...")
        _custom_sparsegpt(model, calib_dataset, args.sparsity)
        method_name = "custom_sparsegpt"
    del calib_dataset

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
        "target_sparsity": 0.5 if args.backend == "modelopt" else args.sparsity,
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