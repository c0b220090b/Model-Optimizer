"""
1-3. AWQ 量子化（NVIDIA TensorRT Model Optimizer を使用）- 精度・評価修正版
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


def build_forward_loop(calib_dataset):
    """
    結合済みのトークンID（dict形式のリスト）をそのままモデルに入力するフォワードループ。
    NVIDIA modelopt のキャリブレーションに必要なテンソルを正確に流します。
    """
    def forward_loop(m):
        with torch.no_grad():
            for batch in calib_dataset:
                # すでに適切なデバイスに載ったテンソルを受け取ってモデルにフォワード
                m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return forward_loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--num_calib_samples", type=int, default=256,
                   help="キャリブレーションに使用する2048トークンの結合サンプル総数")
    common_output_args(p)
    args = p.parse_args()

    # 1. 浮動小数点精度（ベースモデル）のロード
    model, tokenizer = load_llm(args.model_path, dtype="bf16", device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device

    # --- 🛠️ 修正①：短いテキストを 2048 トークンのチャンクに結合（ガッチャンコ）する処理 ---
    print("🤖 キャリブレーションデータの結合処理を開始します...")
    # 2048トークンの塊を必要数(args.num_calib_samples)作るため、多めに生テキストを取得
    raw_texts = default_calibration_texts(args.num_calib_samples * 60)
    
    all_input_ids = []
    for text in raw_texts:
        if text.strip():
            all_input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            all_input_ids.append(tokenizer.eos_token_id)  # EOSを明示的に付与して区切る

    seqlen = 2048
    calib_dataset = []
    for i in range(0, len(all_input_ids), seqlen):
        chunk = all_input_ids[i : i + seqlen]
        if len(chunk) == seqlen:  # 端数は綺麗に切り捨てる
            calib_dataset.append({
                "input_ids": torch.tensor([chunk]).to(device),
                "attention_mask": torch.tensor([1] * seqlen).to(device)
            })
            if len(calib_dataset) >= args.num_calib_samples:
                break

    print(f"✅ 平均コンテキスト長 {seqlen} のサンプルを {len(calib_dataset)} 個作成しました。")
    forward_loop = build_forward_loop(calib_dataset)

    # 2. NVIDIA Model Optimizer の AWQ (INT4量子化シミュレーション) の実行
    print("⚡ AWQ (modelopt) 量子化を実行中...")
    model = mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop)

    # 3. 修正③：量子化済みモデルを一度ディスクにクリーンにエクスポート（保存）する
    print("💾 量子化済みチェックポイントをエクスポート中...")
    os.makedirs(args.output_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # # メモリを解放して競合を防ぐ
    # del model
    # torch.cuda.empty_cache()

    # # 4. 🛠️ 修正③（続き）：保存された完成版の量子化モデルを形状のミスマッチを許容してロード
    # print("🔄 評価のために、エクスポートされた量子化モデルをパッキング形式を考慮して再ロードしています...")
    # from transformers import AutoModelForCausalLM
    # quant_model = AutoModelForCausalLM.from_pretrained(
    #     args.output_dir,
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda:0",
    #     ignore_mismatched_sizes=True, # NVIDIAパッキングによるサイズ縮小のエラーを無視
    #     low_cpu_mem_usage=True
    # )
    # quant_model.eval()


    # # 5. 🛠️ 修正②：評価用（Perplexity用）データも長い文脈（2048トークン）に結合して流す
    # print("🤖 評価用（Perplexity用）データの結合処理を開始します...")
    # eval_raw_texts = default_calibration_texts(100)  # 多めにロード
    # eval_input_ids = []
    # for text in eval_raw_texts:
    #     if text.strip():
    #         eval_input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    #         eval_input_ids.append(tokenizer.eos_token_id)
            
    # eval_texts = []
    # for i in range(0, len(eval_input_ids), seqlen):
    #     chunk = eval_input_ids[i : i + seqlen]
    #     if len(chunk) == seqlen:
    #         eval_texts.append(tokenizer.decode(chunk))
    #         if len(eval_texts) >= 10:  # 評価用には10個（約2万トークン）あれば十分
    #             break

    # # 6. 速度と精度の測定
    # print("📊 性能ベンチマークを測定中...")
    # latency = measure_llm_latency(quant_model, tokenizer)
    # ppl = measure_perplexity(quant_model, tokenizer, eval_texts)

    # results = {
    #     "method": "modelopt_awq_int4",
    #     "backend": "nvidia-modelopt (mtq.INT4_AWQ_CFG)",
    #     "model_size_mb": get_model_size_mb(quant_model),
    #     "perplexity": ppl,
    #     **latency,
    # }
    # save_results(results, args.output_dir)
    # print(results)


if __name__ == "__main__":
    main()
