"""
2-6. Attention Head Pruning（構造化）
各アテンションヘッドの出力ノルム（o_proj への入力）の平均を重要度とし、重要度の低い
ヘッドを「物理的に」削除する（ゼロ埋めではなく行列を小さくする）。

GQA/MQA の扱い:
  - GQA（num_heads > num_kv_heads）: 各 KV グループ内の Query ヘッドを同数ずつ削除する。
    KV ヘッドは残すので、グループの対応関係（Query → KV）が崩れない。
    例: TinyLlama は 32 Query / 4 KV（1グループ 8 ヘッド）。ratio=0.25 → 各グループ 2 ヘッド削除 → 24 Query / 4 KV。
  - MHA（num_heads == num_kv_heads）: Query/Key/Value のヘッドをまとめて削除する。
全層で同数を削るので、config（num_attention_heads, head_dim）を更新すれば
save_pretrained / from_pretrained がそのまま通る。

制約: transformers の LlamaConfig は hidden_size % num_attention_heads == 0 を要求するため、
削除後の Query ヘッド数が hidden_size を割り切る ratio しか物理削除できない
（TinyLlama / Llama-3-8B / Mistral-7B では 0.5 が該当。0.25 は不可）。
任意の ratio を試したい場合は --mode mask（ゼロ埋め。サイズ・速度は変わらない）を使う。
キャリブレーションは WikiText-2 train、評価は WikiText-2 test（重ならない）で行う。

使用例:
  # 物理削除（TinyLlama: 32 → 16 Query ヘッド）
  python pruning/attention_head_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --head_pruning_ratio 0.5 \
    --output_dir ./output/tinyllama-headprune-50

  # ゼロ埋め（任意の ratio）
  python pruning/attention_head_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --mode mask --head_pruning_ratio 0.25 \
    --output_dir ./output/tinyllama-headmask-25
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_num_params, get_model_size_mb, measure_llm_latency,
                          measure_perplexity_wikitext, get_calib_dataset, run_calibration,
                          save_results, common_output_args, common_eval_args)
import torch
import torch.nn as nn


def get_head_dim(config):
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


@torch.no_grad()
def compute_head_importance(model, calib_dataset, num_heads, head_dim):
    """各層・各ヘッドについて、o_proj への入力（= ヘッド出力）の L2 ノルムの平均を返す"""
    importance = {}  # layer_idx -> Tensor[num_heads]

    def make_hook(layer_idx):
        def fn(module, inp, out):
            x = inp[0].detach()
            x = x.reshape(*x.shape[:-1], num_heads, head_dim)
            norm = x.float().norm(dim=-1).reshape(-1, num_heads).mean(dim=0)  # [num_heads]
            importance[layer_idx] = importance.get(layer_idx, 0) + norm.cpu()
        return fn

    hooks = [layer.self_attn.o_proj.register_forward_hook(make_hook(i))
             for i, layer in enumerate(model.model.layers)]
    try:
        run_calibration(model, calib_dataset)
    finally:
        for h in hooks:
            h.remove()
    return importance


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


def _head_rows(heads, head_dim):
    """ヘッド番号のリスト → 重み行列の行（or 列）インデックス"""
    return torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in heads])


def _plan(H, K, ratio):
    """ratio から (削除後の Query ヘッド数, KV ヘッド数, GQA か, グループ内削除数 or 全体削除数) を決める"""
    rep = H // K
    if rep > 1:
        n = int(rep * ratio)
        return K * (rep - n), K, True, n
    n = int(H * ratio)
    return H - n, H - n, False, n


def valid_ratios(config):
    """物理削除できる（削除後のヘッド数が hidden_size を割り切る）ratio の一覧"""
    H, K = config.num_attention_heads, config.num_key_value_heads
    rep = H // K
    steps = rep if rep > 1 else H
    out = []
    for n in range(1, steps):
        r = n / steps
        new_H = _plan(H, K, r)[0]
        if config.hidden_size % new_H == 0:
            out.append(r)
    return out


@torch.no_grad()
def mask_heads(model, importance, pruning_ratio):
    """（--mode mask）重要度の低いヘッドの o_proj 入力列をゼロ埋めする。任意の ratio を使えるが
    行列サイズは変わらないので、パラメータ数・サイズ・速度は減らない。"""
    cfg = model.config
    H = cfg.num_attention_heads
    d = get_head_dim(cfg)
    num_prune = int(H * pruning_ratio)
    if num_prune == 0:
        raise ValueError(f"ratio={pruning_ratio} では 1 ヘッドも削除されません")
    pruned_summary = {}
    for i, layer in enumerate(model.model.layers):
        heads = sorted(torch.topk(importance[i], num_prune, largest=False).indices.tolist())
        pruned_summary[i] = heads
        for h in heads:
            layer.self_attn.o_proj.weight.data[:, h * d:(h + 1) * d] = 0
    return pruned_summary, H, H - num_prune, cfg.num_key_value_heads, cfg.num_key_value_heads


@torch.no_grad()
def prune_heads(model, importance, pruning_ratio):
    cfg = model.config
    H, K = cfg.num_attention_heads, cfg.num_key_value_heads
    d = get_head_dim(cfg)
    rep = H // K
    new_H, new_K, gqa, n = _plan(H, K, pruning_ratio)
    if n == 0 or new_H == 0:
        raise ValueError(f"ratio={pruning_ratio} では削除数が不正です（削除数 {n}）。"
                         f"物理削除できる ratio: {valid_ratios(cfg)}")
    if cfg.hidden_size % new_H != 0:
        raise ValueError(
            f"ratio={pruning_ratio} だと Query ヘッドが {H} → {new_H} になり、hidden_size={cfg.hidden_size} を"
            f"割り切れないため transformers で保存・読み込みできません。\n"
            f"  物理削除できる ratio: {valid_ratios(cfg)}\n"
            f"  任意の ratio を使いたい場合は --mode mask（ゼロ埋め）を指定してください。")
    per_group = n if gqa else None
    num_prune = None if gqa else n

    pruned_summary = {}
    for i, layer in enumerate(model.model.layers):
        imp = importance[i]
        attn = layer.self_attn
        if gqa:
            keep_q = []
            for g in range(K):
                heads = list(range(g * rep, (g + 1) * rep))
                order = sorted(heads, key=lambda h: imp[h].item(), reverse=True)
                keep_q += sorted(order[: rep - per_group])
            keep_kv = list(range(K))
        else:
            keep_q = sorted(torch.topk(imp, new_H, largest=True).indices.tolist())
            keep_kv = keep_q
        pruned_summary[i] = sorted(set(range(H)) - set(keep_q))

        q_idx = _head_rows(keep_q, d)
        attn.q_proj = _slice_linear(attn.q_proj, q_idx, dim=0)
        attn.o_proj = _slice_linear(attn.o_proj, q_idx, dim=1)
        if not gqa:
            kv_idx = _head_rows(keep_kv, d)
            attn.k_proj = _slice_linear(attn.k_proj, kv_idx, dim=0)
            attn.v_proj = _slice_linear(attn.v_proj, kv_idx, dim=0)
        attn.num_key_value_groups = new_H // new_K
        for attr, val in (("num_heads", new_H), ("num_key_value_heads", new_K)):
            if hasattr(attn, attr):
                setattr(attn, attr, val)

    # head_dim を明示しないと hidden_size // num_attention_heads で再計算されて壊れる
    cfg.head_dim = d
    cfg.num_attention_heads = new_H
    cfg.num_key_value_heads = new_K
    return pruned_summary, H, new_H, K, new_K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--head_pruning_ratio", type=float, default=0.5)
    p.add_argument("--mode", type=str, default="remove", choices=["remove", "mask"],
                   help="remove: 物理削除（サイズ・速度が減る） / mask: ゼロ埋め（任意の ratio）")
    p.add_argument("--num_calib_samples", type=int, default=32)
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device
    num_params_before = get_num_params(model)
    num_heads = model.config.num_attention_heads
    head_dim = get_head_dim(model.config)

    if args.mode == "remove":
        new_H = _plan(num_heads, model.config.num_key_value_heads, args.head_pruning_ratio)[0]
        if new_H == 0 or model.config.hidden_size % new_H != 0 or new_H == num_heads:
            raise SystemExit(
                f"❌ ratio={args.head_pruning_ratio} は物理削除できません（Query ヘッド {num_heads} → {new_H}）。\n"
                f"   物理削除できる ratio: {valid_ratios(model.config)}\n"
                f"   任意の ratio を使いたい場合は --mode mask を指定してください。")

    print("🤖 キャリブレーションデータ（WikiText-2 train）を準備中...")
    calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.seqlen,
                                      device=device, seed=args.seed)
    print("⚡ ヘッド重要度を計算中...")
    importance = compute_head_importance(model, calib_dataset, num_heads, head_dim)
    del calib_dataset

    if args.mode == "remove":
        pruned_summary, H, new_H, K, new_K = prune_heads(model, importance, args.head_pruning_ratio)
    else:
        pruned_summary, H, new_H, K, new_K = mask_heads(model, importance, args.head_pruning_ratio)
    print(f"✂ Query heads: {H} → {new_H},  KV heads: {K} → {new_K}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("📊 性能ベンチマークを測定中（PPL: WikiText-2 test）...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                      max_chunks=args.eval_max_chunks)

    num_params = get_num_params(model)
    results = {
        "method": "attention_head_pruning",
        "mode": args.mode,
        "head_pruning_ratio": args.head_pruning_ratio,
        "num_heads_before": H,
        "num_heads_after": new_H,
        "num_kv_heads_before": K,
        "num_kv_heads_after": new_K,
        "pruned_heads_per_layer": pruned_summary,
        "calib_data": f"wikitext2-train x{args.num_calib_samples}",
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