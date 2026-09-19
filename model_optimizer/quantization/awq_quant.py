"""
1-3. AWQ 量子化（NVIDIA TensorRT Model Optimizer を使用）
NVIDIA の nvidia-modelopt ライブラリ (modelopt.torch.quantization) の AWQ 実装
(mtq.INT4_AWQ_CFG, algorithm="awq_lite") を用いて、活性化の大きさに基づき
重要な重みチャネルを保護しながら INT4 量子化する。

事前インストール:
  pip install nvidia-modelopt[hf]

使用例:
  python awq_quant.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --output_dir ./output/llama3.1-8b-modelopt-awq-int4
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_model_size_mb, measure_llm_latency, measure_perplexity,
                          default_calibration_texts, save_results, common_output_args)
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def build_forward_loop(model, tokenizer, calib_texts, device):
    def forward_loop(m):
        with torch.no_grad():
            for t in calib_texts:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device)
                m(**inputs)
    return forward_loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_calib_samples", type=int, default=128)
    common_output_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype="bf16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device

    calib_texts = default_calibration_texts(args.num_calib_samples)
    forward_loop = build_forward_loop(model, tokenizer, calib_texts, device)

    # NVIDIA Model Optimizer の AWQ (INT4, per-group, awq_lite キャリブレーション)
    model = mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop)

    # 評価は export の前に行う。export_hf_checkpoint はレイヤーの resmooth/融合や
    # 重みの実パッキングなど、実際に量子化済みモデルを「書き換える」処理を含むため、
    # export 後に同じ model オブジェクトで forward/generate すると形状不整合等で壊れる。
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity(model, tokenizer, default_calibration_texts(32))

    os.makedirs(args.output_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    results = {
        "method": "modelopt_awq_int4",
        "backend": "nvidia-modelopt (mtq.INT4_AWQ_CFG)",
        "model_size_mb": get_model_size_mb(model),
        "perplexity": ppl,
        **latency,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()
