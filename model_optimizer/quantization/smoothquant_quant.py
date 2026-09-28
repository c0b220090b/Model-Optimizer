"""
1-4. SmoothQuant（NVIDIA TensorRT Model Optimizer を使用）- 精度・評価修正版
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


def build_forward_loop(calib_dataset):
    """
    結合済みのトークンID（dict形式のリスト）をそのままモデルに入力するフォワードループ。
    SmoothQuantの適切な外れ値スケール（Smoothing因子）を計算するためにテンソルを流します。
    """
    def forward_loop(m):
        with torch.no_grad():
            for batch in calib_dataset:
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
    model, tokenizer = load_llm(args.model_path, dtype="fp16", device_map="cuda:0")
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

    # 2. NVIDIA Model Optimizer の SmoothQuant (INT8, 重み+活性化) の実行
    print("⚡ SmoothQuant (modelopt) 量子化（活性化統計のキャッシュ）を実行中...")
    model = mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop)

    # --- 🛠️ 修正②：評価用（Perplexity用）データも長い文脈（2048トークン）に結合して流す ---
    print("🤖 評価用（Perplexity用）データの結合処理を開始します...")
    eval_raw_texts = default_calibration_texts(100)  # 多めにロード
    eval_input_ids = []
    for text in eval_raw_texts:
        if text.strip():
            eval_input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            eval_input_ids.append(tokenizer.eos_token_id)
            
    eval_texts = []
    for i in range(0, len(eval_input_ids), seqlen):
        chunk = eval_input_ids[i : i + seqlen]
        if len(chunk) == seqlen:
            eval_texts.append(tokenizer.decode(chunk))
            if len(eval_texts) >= 10:  # 評価用には10個（約2万トークン）あれば十分
                break

    # 3. 性能ベンチマークを測定（※export前にシミュレーションモデル側で安全に測定）
    print("📊 性能ベンチマークを測定中...")
    latency = measure_llm_latency(model, tokenizer)
    ppl = measure_perplexity(model, tokenizer, eval_texts)

    # 4. ディスクにエクスポート（チェックポイント保存）
    print("💾 SmoothQuant済みチェックポイントをエクスポート中...")
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
