"""
2-7. 2:4 構造化スパース性（NVIDIA TensorRT Model Optimizer を使用）
nvidia-modelopt の modelopt.torch.sparsity は、そもそも Ampere+ GPU の TensorCore が
アクセラレーションできる 2:4 パターンのみをサポートしている。本スクリプトは
"sparse_magnitude"（軽量・キャリブレーション不要）と "sparsegpt"（ヘッシアン近似で
高精度、キャリブレーション要）の2つのスコアリング方式を同条件で比較する。
キャリブレーションは WikiText-2 train、評価は WikiText-2 test（重ならない）で行う。

事前インストール:
  pip install nvidia-modelopt[hf] datasets

使用例:
  python pruning/07_structured_sparsity_2_4.py \
    --model_path TinyLlama/TinyLlama_v1.1 \
    --output_dir ./output/tinyllama-modelopt-2to4-comparison
"""
import sys, os, argparse, gc
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_sparsity, get_linear_sparsity, get_model_size_mb,
                          measure_llm_latency, measure_perplexity_wikitext, get_calib_dataset,
                          save_results, common_output_args, common_eval_args)
import torch
import modelopt.torch.sparsity as mts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_calib_samples", type=int, default=128)
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()

    all_results = {}
    for scorer in ["sparse_magnitude", "sparsegpt"]:
        print(f"===== {scorer} =====")
        model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
        model.eval()
        device = next(model.parameters()).device

        if scorer == "sparsegpt":
            calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.seqlen,
                                              device=device, seed=args.seed)
            config = {"data_loader": calib_dataset, "collect_func": lambda batch: batch}
            model = mts.sparsify(model, mode="sparsegpt", config=config)
            del calib_dataset
        else:
            model = mts.sparsify(model, mode="sparse_magnitude")
        # ラッパーを外してマスクを重みに焼き込む
        model = mts.export(model)

        sub_dir = os.path.join(args.output_dir, scorer)
        os.makedirs(sub_dir, exist_ok=True)
        model.save_pretrained(sub_dir)
        tokenizer.save_pretrained(sub_dir)

        latency = measure_llm_latency(model, tokenizer)
        ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                          max_chunks=args.eval_max_chunks)
        all_results[scorer] = {
            "calib_data": f"wikitext2-train x{args.num_calib_samples}" if scorer == "sparsegpt" else None,
            "actual_sparsity": get_sparsity(model),
            "linear_sparsity": get_linear_sparsity(model),
            "model_size_mb": get_model_size_mb(model),
            "perplexity": ppl,
            **latency,
        }
        save_results({"method": f"modelopt_{scorer}_2to4", **all_results[scorer]}, sub_dir)
        print(scorer, all_results[scorer])

        del model
        gc.collect()
        torch.cuda.empty_cache()

    save_results({
        "method": "modelopt_2to4_sparsity_comparison",
        "eval_data": f"wikitext2-test seqlen={args.seqlen} max_chunks={args.eval_max_chunks}",
        "results": all_results,
    }, args.output_dir)


if __name__ == "__main__":
    main()