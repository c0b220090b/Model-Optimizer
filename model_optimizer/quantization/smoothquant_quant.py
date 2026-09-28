"""
1-4. SmoothQuant（NVIDIA TensorRT Model Optimizer を使用）- 精度・評価修正版
NVIDIA の nvidia-modelopt (modelopt.torch.quantization) が提供する SmoothQuant 実装
(mtq.INT8_SMOOTHQUANT_CFG) を用いる。活性化の外れ値を重み側に移してから
per-channel/per-tensor の INT8 量子化を行う (Xiao et al., 2023 の手法を modelopt が実装)。
キャリブレーションは WikiText-2 train、評価は WikiText-2 test（重ならない）で行う。

事前インストール:
  pip install nvidia-modelopt[hf] datasets

使用例:
  python quantization/smoothquant_quant.py \
    --model_path facebook/opt-1.3b \
    --output_dir ./output/opt1.3b-modelopt-smoothquant-int8
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_model_size_mb, measure_llm_latency,
                          measure_perplexity_wikitext, get_calib_dataset,
                          save_results, common_output_args, common_eval_args)
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def build_forward_loop(calib_dataset):
    """キャリブレーション用のトークン列（dict のリスト）をそのままモデルに流すフォワードループ。
    SmoothQuant の smoothing 因子と量子化スケールを決めるための活性化統計を集める。"""
    def forward_loop(m):
        with torch.no_grad():
            for batch in calib_dataset:
                m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return forward_loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_calib_samples", type=int, default=256,
                   help="キャリブレーションのサンプル数（各 seqlen トークン）")
    common_output_args(p)
    common_eval_args(p)
    args = p.parse_args()

    # 1. 浮動小数点精度（ベースモデル）のロード
    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device

    # 2. キャリブレーションデータ（WikiText-2 train からランダムに seqlen トークン窓を切り出す）
    print("🤖 キャリブレーションデータ（WikiText-2 train）を準備中...")
    calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.seqlen,
                                      device=device, seed=args.seed)
    print(f"✅ {args.seqlen} トークンのサンプルを {len(calib_dataset)} 個作成しました。")

    # 3. NVIDIA Model Optimizer の SmoothQuant (INT8, 重み+活性化) の実行
    print("⚡ SmoothQuant (modelopt) 量子化（活性化統計の収集）を実行中...")
    model = mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, build_forward_loop(calib_dataset))

    # 4. 性能ベンチマーク（export 前のシミュレーションモデルで測定）
    print("📊 性能ベンチマークを測定中（PPL: WikiText-2 test）...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                      max_chunks=args.eval_max_chunks)

    # 5. ディスクにエクスポート（チェックポイント保存）
    print("💾 SmoothQuant済みチェックポイントをエクスポート中...")
    os.makedirs(args.output_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    results = {
        "method": "modelopt_smoothquant_int8",
        "backend": "nvidia-modelopt (mtq.INT8_SMOOTHQUANT_CFG)",
        "calib_data": f"wikitext2-train x{args.num_calib_samples}",
        "eval_data": f"wikitext2-test seqlen={args.seqlen} max_chunks={args.eval_max_chunks}",
        "model_size_mb": get_model_size_mb(model),
        "perplexity": ppl,
        **latency,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()