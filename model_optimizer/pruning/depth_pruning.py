"""
2-5. Depth Pruning（構造化, 層除去）
冗長な Transformer 層をまるごと削除する。ShortGPT (Men et al., 2024) の Block Influence
  BI_i = 1 - E_tokens[ cos(x_i, x_{i+1}) ]
を指標に、入力と出力がほぼ同じ（= 恒等写像に近い）層から除去する。
キャリブレーションは WikiText-2 train、評価は WikiText-2 test（重ならない）で行う。

使用例:
  python pruning/depth_pruning.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --num_layers_to_remove 4 \
    --output_dir ./output/tinyllama-depthprune-4layers
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_num_params, get_model_size_mb, measure_llm_latency,
                          measure_perplexity_wikitext, get_calib_dataset, run_calibration,
                          save_results, common_output_args, common_eval_args)
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_block_influence(model, calib_dataset):
    """各デコーダ層の入力と出力のコサイン類似度を「全トークン」で平均し、BI = 1 - cos を返す。
    output_hidden_states は最終層の値に final norm が掛かっている実装があるため使わず、
    各層に直接フックして正規化前の residual stream を比較する。"""
    layers = model.model.layers
    cos_sum = [0.0] * len(layers)
    tok_count = [0] * len(layers)

    def make_hook(i):
        def fn(module, args, kwargs, output):
            h_in = args[0] if args else kwargs["hidden_states"]
            h_out = output[0] if isinstance(output, tuple) else output
            cos = F.cosine_similarity(h_in.float(), h_out.float(), dim=-1)  # [batch, seq]
            cos_sum[i] += cos.sum().item()
            tok_count[i] += cos.numel()
        return fn

    hooks = [layer.register_forward_hook(make_hook(i), with_kwargs=True)
             for i, layer in enumerate(layers)]
    try:
        run_calibration(model, calib_dataset)
    finally:
        for h in hooks:
            h.remove()
    return [1.0 - cos_sum[i] / tok_count[i] for i in range(len(layers))]


def remove_layers(model, layer_indices_to_remove):
    layers = model.model.layers
    keep = [l for i, l in enumerate(layers) if i not in layer_indices_to_remove]
    # KV キャッシュは self_attn.layer_idx で層を引くので、詰め直した番号に振り直す
    # （そのままだと generate 時に存在しないキャッシュ番号を参照して壊れる）
    for new_idx, layer in enumerate(keep):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx
    model.model.layers = torch.nn.ModuleList(keep)
    model.config.num_hidden_layers = len(keep)
    # 新しい transformers は層ごとの種別リスト（full / sliding attention 等）を持つので合わせて削る
    if getattr(model.config, "layer_types", None) is not None:
        model.config.layer_types = [t for i, t in enumerate(model.config.layer_types)
                                    if i not in layer_indices_to_remove]
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_layers_to_remove", type=int, default=4)
    p.add_argument("--num_calib_samples", type=int, default=32)
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device
    num_layers_before = model.config.num_hidden_layers
    num_params_before = get_num_params(model)

    print("🤖 キャリブレーションデータ（WikiText-2 train）を準備中...")
    calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.seqlen,
                                      device=device, seed=args.seed)
    print("⚡ Block Influence を計算中...")
    bi = compute_block_influence(model, calib_dataset)
    del calib_dataset
    for i, v in enumerate(bi):
        print(f"  layer {i:2d}: BI = {v:.4f}")

    # BI が小さい(=恒等写像に近い)層から除去
    remove_idx = set(sorted(range(len(bi)), key=lambda i: bi[i])[:args.num_layers_to_remove])
    print(f"✂ 除去する層のインデックス: {sorted(remove_idx)}")
    remove_layers(model, remove_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("📊 性能ベンチマークを測定中（PPL: WikiText-2 test）...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                      max_chunks=args.eval_max_chunks)

    num_params = get_num_params(model)
    results = {
        "method": "depth_pruning_shortgpt_bi",
        "num_layers_before": num_layers_before,
        "num_layers_removed": args.num_layers_to_remove,
        "removed_layer_indices": sorted(remove_idx),
        "block_influence": bi,
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