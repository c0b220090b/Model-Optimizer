"""
1-4. SmoothQuant（NVIDIA TensorRT Model Optimizer を使用）
NVIDIA の nvidia-modelopt (modelopt.torch.quantization) が提供する SmoothQuant 実装
(mtq.INT8_SMOOTHQUANT_CFG) を用いる。活性化の外れ値を重み側に移してから
per-channel/per-tensor の INT8 量子化を行う (Xiao et al., 2023 の手法を modelopt が実装)。

事前インストール:
  pip install nvidia-modelopt[hf]

使用例:
  python smoothquant_quant.py \
    --model_path facebook/opt-1.3b \
    --output_dir ./output/opt1.3b-modelopt-smoothquant-int8
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_model_size_mb, measure_llm_latency, measure_perplexity,
                          default_calibration_texts, save_results, common_output_args)
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_calib_samples", type=int, default=128)
    common_output_args(p)
    args = p.parse_args()

    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device

    calib_texts = default_calibration_texts(args.num_calib_samples)

    def forward_loop(m):
        with torch.no_grad():
            for t in calib_texts:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device)
                m(**inputs)

    # NVIDIA Model Optimizer の SmoothQuant (INT8, 重み+活性化)
    model = mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop)

    # 評価は export の前に行う。export_hf_checkpoint はレイヤーの resmooth/融合や
    # 重みの実パッキングなど、実際に量子化済みモデルを「書き換える」処理を含むため、
    # export 後に同じ model オブジェクトで forward/generate すると形状不整合等で壊れる。
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity(model, tokenizer, default_calibration_texts(32))

    os.makedirs(args.output_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    results = {
        "method": "modelopt_smoothquant_int8",
        "backend": "nvidia-modelopt (mtq.INT8_SMOOTHQUANT_CFG)",
        "model_size_mb": get_model_size_mb(model),
        "perplexity": ppl,
        **latency,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()
