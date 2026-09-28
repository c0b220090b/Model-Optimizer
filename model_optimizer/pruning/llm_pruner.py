"""
2-4. LLM-Pruner（構造化, 依存関係を考慮したブロック単位の除去）- MLP チャネル版
Ma et al. (2023) の手法に基づき、MLP の中間次元を「依存グループ」単位で評価・除去する。
中間チャネル c のグループ = gate_proj の行 c + up_proj の行 c + down_proj の列 c。
重要度はグループ内の全パラメータの Taylor 1次近似 |w * grad(w)| の和。

除去は「ゼロ埋め」ではなく実際に行列を小さくする（物理削除）。全層で同じ割合を刈るので
config.intermediate_size を更新するだけで save_pretrained / from_pretrained がそのまま通り、
パラメータ数・サイズ・速度が実際に減る。

（簡易実装の注意）
- 原論文の Attention head の刈り込み、先頭/末尾層の除外、LoRA による回復学習は未実装。
- 勾配は fp16 のまま計算するためアンダーフローしないよう loss をスケールし、
  inf/nan が出たサンプルはスケールを下げてやり直す。
キャリブレーションは WikiText-2 train、評価は WikiText-2 test（重ならない）で行う。

使用例:
  python pruning/llm_pruner.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --pruning_ratio 0.2 \
    --output_dir ./output/tinyllama-llmpruner-mlp20
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_num_params, get_model_size_mb, measure_llm_latency,
                          measure_perplexity_wikitext, get_calib_dataset,
                          save_results, common_output_args, common_eval_args)
import torch
import torch.nn as nn


def compute_mlp_group_importance(model, calib_dataset, init_loss_scale: float = 2.0 ** 12):
    """各層の MLP 中間チャネルごとに Σ_samples Σ_group |w * grad| を返す。
    戻り値: list[Tensor[intermediate]]（層インデックス順, fp32, CPU）"""
    layers = model.model.layers
    mlp_params = []
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers:
        for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            lin.weight.requires_grad_(True)
            mlp_params.append(lin.weight)

    importance = [torch.zeros(layer.mlp.gate_proj.out_features, dtype=torch.float32,
                              device=layer.mlp.gate_proj.weight.device) for layer in layers]
    scale = init_loss_scale
    for batch in calib_dataset:
        while True:
            for p in mlp_params:
                p.grad = None
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["input_ids"])
            (out.loss * scale).backward()
            if all(torch.isfinite(p.grad).all() for p in mlp_params):
                break
            scale /= 2
            if scale < 1:
                raise RuntimeError("勾配が inf/nan になりました（loss scale を 1 まで下げても解消せず）")
            print(f"  ⚠ 勾配オーバーフロー → loss scale を {scale:g} に下げて再計算")

        with torch.no_grad():
            for i, layer in enumerate(layers):
                g = layer.mlp.gate_proj.weight
                u = layer.mlp.up_proj.weight
                d = layer.mlp.down_proj.weight
                # グループ c: gate 行 c + up 行 c + down 列 c
                imp = (g.float() * g.grad.float()).abs().sum(dim=1)
                imp += (u.float() * u.grad.float()).abs().sum(dim=1)
                imp += (d.float() * d.grad.float()).abs().sum(dim=0)
                importance[i] += imp / scale

    for p in mlp_params:
        p.grad = None
        p.requires_grad_(False)
    return [imp.cpu() for imp in importance]


def _slice_linear(lin: nn.Linear, idx: torch.Tensor, dim: int) -> nn.Linear:
    """idx のチャネルだけ残した新しい Linear を作る（dim=0: 出力側, dim=1: 入力側）"""
    idx = idx.to(lin.weight.device)
    w = lin.weight.data.index_select(dim, idx)
    new = nn.Linear(w.shape[1], w.shape[0], bias=lin.bias is not None,
                    device=w.device, dtype=w.dtype)
    new.weight.data.copy_(w)
    if lin.bias is not None:
        b = lin.bias.data if dim == 1 else lin.bias.data.index_select(0, idx)
        new.bias.data.copy_(b)
    new.requires_grad_(False)
    return new


@torch.no_grad()
def prune_mlp_channels(model, importance, pruning_ratio: float):
    """各層で重要度の低い中間チャネルを同数ずつ物理削除する"""
    inter = model.config.intermediate_size
    num_keep = inter - int(inter * pruning_ratio)
    for layer, imp in zip(model.model.layers, importance):
        keep = torch.topk(imp, num_keep, largest=True).indices.sort().values
        mlp = layer.mlp
        mlp.gate_proj = _slice_linear(mlp.gate_proj, keep, dim=0)
        mlp.up_proj = _slice_linear(mlp.up_proj, keep, dim=0)
        mlp.down_proj = _slice_linear(mlp.down_proj, keep, dim=1)
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = num_keep
    model.config.intermediate_size = num_keep
    return num_keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--pruning_ratio", type=float, default=0.2,
                   help="MLP 中間次元のうち削除する割合（全層共通）")
    p.add_argument("--num_calib_samples", type=int, default=32)
    p.add_argument("--calib_seqlen", type=int, default=512,
                   help="勾配計算用のキャリブレーション長（逆伝播のメモリを抑えるため評価より短め）")
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
    model.eval()  # dropout を切ったまま勾配だけ計算する
    device = next(model.parameters()).device
    num_params_before = get_num_params(model)

    print("🤖 キャリブレーションデータ（WikiText-2 train）を準備中...")
    calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.calib_seqlen,
                                      device=device, seed=args.seed)
    print("⚡ Taylor 重要度（MLP チャネルグループ）を計算中...")
    importance = compute_mlp_group_importance(model, calib_dataset)
    del calib_dataset
    torch.cuda.empty_cache()

    old_inter = model.config.intermediate_size
    new_inter = prune_mlp_channels(model, importance, args.pruning_ratio)
    print(f"✂ intermediate_size: {old_inter} → {new_inter}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("📊 性能ベンチマークを測定中（PPL: WikiText-2 test）...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                      max_chunks=args.eval_max_chunks)

    num_params = get_num_params(model)
    results = {
        "method": "llm_pruner_taylor_mlp",
        "pruning_ratio": args.pruning_ratio,
        "intermediate_size": new_inter,
        "calib_data": f"wikitext2-train x{args.num_calib_samples} seqlen={args.calib_seqlen}",
        "eval_data": f"wikitext2-test seqlen={args.seqlen} max_chunks={args.eval_max_chunks}",
        "num_params": num_params,
        "param_reduction": 1 - num_params / num_params_before,
        "model_size_mb": get_model_size_mb(model),
        "perplexity": ppl,
        **latency,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()