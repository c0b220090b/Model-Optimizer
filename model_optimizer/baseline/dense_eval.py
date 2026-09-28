"""
0. ベースライン評価（最適化なしの元モデル）
量子化・プルーニングの各スクリプトと同じ条件（WikiText-2 test, seqlen=2048）で
Perplexity・レイテンシ・メモリを測り、比較の基準値を作る。

使用例:
  python baseline/dense_eval.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --output_dir ./output/tinyllama_v1.1-dense
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_sparsity, get_linear_sparsity, get_model_size_mb,
                          measure_llm_latency, measure_perplexity_wikitext,
                          save_results, common_output_args, common_eval_args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype=args.dtype, device_map="cuda:0")
    model.eval()

    print("📊 性能ベンチマークを測定中（PPL: WikiText-2 test）...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                      max_chunks=args.eval_max_chunks)

    results = {
        "method": f"dense_{args.dtype}",
        "backend": "none",
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
