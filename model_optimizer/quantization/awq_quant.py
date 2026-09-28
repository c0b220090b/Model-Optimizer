"""
1-3. AWQ 量子化（NVIDIA TensorRT Model Optimizer を使用）- 精度・評価修正版
NVIDIA の nvidia-modelopt ライブラリ (modelopt.torch.quantization) の AWQ 実装
(mtq.INT4_AWQ_CFG, algorithm="awq_lite") を用いて、活性化の大きさに基づき
重要な重みチャネルを保護しながら INT4 量子化する。
キャリブレーションは WikiText-2 train、評価は WikiText-2 test（重ならない）で行う。

評価について:
  export_hf_checkpoint が出力する INT4 パック済みチェックポイントは TensorRT-LLM / vLLM 向けの形式で、
  transformers の from_pretrained ではそのまま読めない。そのため smoothquant_quant.py と同様に、
  export 前の fake-quant（量子化シミュレーション）モデルで Perplexity を測る。
  この PPL は実際の INT4 推論とほぼ同じ数値誤差を再現する。
  一方、レイテンシとメモリは fake-quant のため bf16 相当（高速化・省メモリは反映されない）。
  圧縮後のサイズは checkpoint_size_mb（エクスポート先のディスク上サイズ）で確認する。

事前インストール:
  pip install nvidia-modelopt[hf] datasets

使用例:
  python quantization/awq_quant.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --output_dir ./output/llama3.1-8b-modelopt-awq-int4
"""
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (load_llm, get_model_size_mb, get_checkpoint_size_mb,
                          measure_llm_latency, measure_perplexity_wikitext, get_calib_dataset,
                          save_results, common_output_args, common_eval_args)
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def build_forward_loop(calib_dataset):
    """キャリブレーション用のトークン列（dict のリスト）をそのままモデルに流すフォワードループ。"""
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
    model, tokenizer = load_llm(args.model_path, dtype="bf16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device

    # 2. キャリブレーションデータ（WikiText-2 train からランダムに seqlen トークン窓を切り出す）
    print("🤖 キャリブレーションデータ（WikiText-2 train）を準備中...")
    calib_dataset = get_calib_dataset(tokenizer, args.num_calib_samples, args.seqlen,
                                      device=device, seed=args.seed)
    print(f"✅ {args.seqlen} トークンのサンプルを {len(calib_dataset)} 個作成しました。")

    # 3. NVIDIA Model Optimizer の AWQ (INT4 量子化シミュレーション) の実行
    print("⚡ AWQ (modelopt) 量子化を実行中...")
    model = mtq.quantize(model, mtq.INT4_AWQ_CFG, build_forward_loop(calib_dataset))

    # 4. 性能ベンチマーク（export 前の fake-quant モデルで測定）
    print("📊 性能ベンチマークを測定中（PPL: WikiText-2 test）...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity_wikitext(model, tokenizer, seqlen=args.seqlen,
                                      max_chunks=args.eval_max_chunks)

    # 5. ディスクにエクスポート（INT4 パック済みチェックポイント）
    print("💾 量子化済みチェックポイントをエクスポート中...")
    os.makedirs(args.output_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    results = {
        "method": "modelopt_awq_int4",
        "backend": "nvidia-modelopt (mtq.INT4_AWQ_CFG)",
        "calib_data": f"wikitext2-train x{args.num_calib_samples}",
        "eval_data": f"wikitext2-test seqlen={args.seqlen} max_chunks={args.eval_max_chunks}",
        "eval_model": "fake-quant (export 前)",
        "model_size_mb": get_model_size_mb(model),            # fake-quant のメモリ上サイズ（bf16 相当）
        "checkpoint_size_mb": get_checkpoint_size_mb(args.output_dir),  # INT4 パック後の実サイズ
        "perplexity": ppl,
        **latency,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()