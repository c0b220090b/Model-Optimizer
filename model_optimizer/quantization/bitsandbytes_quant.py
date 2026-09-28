"""
1-1. bitsandbytes INT8 / NF4 量子化（PTQ, 重みのみ）
最も手軽な量子化のベースライン。キャリブレーション不要でロード時に量子化される。

使用例:
  python bitsandbytes_quant.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dtype nf4 \
    --output_dir ./output/llama3.1-8b-bnb-nf4
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_model_size_mb, get_num_params,
                          measure_llm_latency, measure_perplexity,
                          default_calibration_texts, save_results, common_output_args)
import torch
from transformers import BitsAndBytesConfig


def build_bnb_config(dtype: str) -> BitsAndBytesConfig:
    if dtype == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif dtype == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif dtype == "fp4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"unsupported dtype: {dtype}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="nf4", choices=["int8", "nf4", "fp4"])
    common_output_args(p)
    args = p.parse_args()

    bnb_config = build_bnb_config(args.dtype)
    model, tokenizer = load_llm(args.model_path, quantization_config=bnb_config)
    model.eval()

    # bitsandbytes は「保存」して再ロードする形の量子化ではなく実行時量子化のため、
    # 再現性のために設定と tokenizer のみ保存し、評価結果を記録する。
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "quantization_config.json"), "w") as f:
        f.write(bnb_config.to_json_string())

    # --- 【修正】bitsandbytesの評価用（Perplexity用）データも、長い文脈に結合して渡す ---
    print("🤖 bnb評価用データの結合処理を開始します...")
    eval_raw_texts = default_calibration_texts(100)  # 多めにロード
    eval_input_ids = []
    for text in eval_raw_texts:
        if text.strip():
            eval_input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            eval_input_ids.append(tokenizer.eos_token_id)
            
    # 2048トークンずつの評価用チャンクを綺麗に作成
    eval_texts = []
    seqlen = 2048
    for i in range(0, len(eval_input_ids), seqlen):
        chunk = eval_input_ids[i : i + seqlen]
        if len(chunk) == seqlen:
            eval_texts.append(tokenizer.decode(chunk))
            if len(eval_texts) >= 10:  # 10サンプル（約2万トークン）
                break
                
    # 結合された高品質なテキストで正しく Perplexity を測定
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity(model, tokenizer, eval_texts)
    # --------------------------------------------------------------------------------


    results = {
        "method": "bitsandbytes",
        "dtype": args.dtype,
        "num_params": get_num_params(model),
        "model_size_mb": get_model_size_mb(model),
        "perplexity": ppl,
        **latency,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()
