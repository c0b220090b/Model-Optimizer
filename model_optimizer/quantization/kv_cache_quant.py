"""
1-7. KVキャッシュ量子化（NVIDIA TensorRT Model Optimizer を使用）
nvidia-modelopt の mtq.FP8_KV_CFG（または NVFP4_KV_CFG）を、重み量子化の設定と
マージして適用する。KVキャッシュ用の quantizer (*_bmm_quantizer) のみを対象にした
設定のため、ベース重み量子化(例: FP8_DEFAULT_CFG)と組み合わせて使うのが一般的。

事前インストール:
  pip install nvidia-modelopt[hf]

使用例:
  python kv_cache_quant.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --kv_dtype fp8 \
    --context_len 4096 \
    --output_dir ./output/llama3.1-8b-modelopt-kvfp8
"""
import sys, os, argparse, copy
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import load_llm, get_peak_memory_mb, reset_peak_memory, default_calibration_texts, save_results, common_output_args
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def build_long_context_prompt(tokenizer, target_len: int) -> str:
    base = "深層学習モデルの最適化に関する長い技術文書です。 "
    text = base * (target_len // len(base) + 1)
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0][:target_len]
    return tokenizer.decode(ids)


@torch.no_grad()
def measure_generate_memory(model, tokenizer, prompt, max_new_tokens=64):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    reset_peak_memory()
    model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return get_peak_memory_mb()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--kv_dtype", type=str, default="fp8", choices=["fp8", "nvfp4"])
    p.add_argument("--context_len", type=int, default=4096)
    p.add_argument("--num_calib_samples", type=int, default=64)
    common_output_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype="bf16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device
    prompt = build_long_context_prompt(tokenizer, args.context_len)

    # --- ベースライン (KVキャッシュ量子化なし) のメモリ計測 ---
    baseline_mem = measure_generate_memory(model, tokenizer, prompt)

    # --- ベース重み量子化 (FP8) + KVキャッシュ量子化設定をマージして適用 ---
    kv_cfg = mtq.FP8_KV_CFG if args.kv_dtype == "fp8" else mtq.NVFP4_KV_CFG
    merged_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    merged_quant_cfg = merged_cfg.get("quant_cfg", {})
    kv_quant_cfg = kv_cfg.get("quant_cfg", {})
    
    if isinstance(merged_quant_cfg, list) or isinstance(kv_quant_cfg, list):
        # リスト型だった場合は配列として結合する
        m_list = merged_quant_cfg if isinstance(merged_quant_cfg, list) else [merged_quant_cfg]
        k_list = kv_quant_cfg if isinstance(kv_quant_cfg, list) else [kv_quant_cfg]
        merged_cfg["quant_cfg"] = m_list + k_list
    else:
        # 両方辞書型だった場合は元の処理
        merged_cfg["quant_cfg"] = {**merged_quant_cfg, **kv_quant_cfg}


    calib_texts = default_calibration_texts(args.num_calib_samples)

    def forward_loop(m):
        with torch.no_grad():
            for t in calib_texts:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device)
                m(**inputs)

    model = mtq.quantize(model, merged_cfg, forward_loop)

    quant_mem = measure_generate_memory(model, tokenizer, prompt)

    os.makedirs(args.output_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    results = {
        "method": "modelopt_kv_cache_quantization",
        "kv_dtype": args.kv_dtype,
        "context_len": args.context_len,
        "peak_memory_mb_baseline": baseline_mem,
        "peak_memory_mb_kv_quantized": quant_mem,
        "memory_reduction_pct": 100 * (1 - quant_mem / baseline_mem) if baseline_mem else None,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()
