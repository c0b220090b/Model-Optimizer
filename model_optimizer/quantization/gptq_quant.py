"""
1-2. GPTQ 量子化（PTQ, 重みのみ, キャリブレーション要）
誤差補正付きレイヤーごとの量子化。GPTQModel（旧auto-gptqの後継, 能動的にメンテされている
公式実装）を使用する。auto-gptqは開発が停止しており、特に新しいPython/CUDA/torchの
組み合わせではソースビルドに失敗しやすいため、本プロジェクトではGPTQModelを採用する。

事前インストール:
  pip install gptqmodel

推論バックエンドについて:
  GPTQModel.load() は既定(backend="auto")で Marlin / ExLlamaV2 などの高速カーネルを
  自動選択するが、これらは初回ロード時に torch.ops のCUDA拡張をJITビルドする必要があり、
  CUDA Toolkit（nvcc, CUDA_HOMEで参照される完全版。pipの nvidia-cuda-runtime-cu12 等の
  ランタイムのみのパッケージでは不可）が無い環境ではビルドに失敗し、
  `ModuleNotFoundError: ... CUDA_HOME environment variable is not set` で落ちる。
  本スクリプトは既定でロード時のバックエンドを --load_backend torch
  （BACKEND.TORCH, 純PyTorch実装でJITビルド不要）にしており、この問題を回避する。
  量子化自体（from_pretrained/quantize/save）はカーネル選択と無関係のため影響を受けない。
  CUDA_HOMEが設定できる環境でMarlin等の高速カーネルを使いたい場合は
  --load_backend auto を指定する。

使用例:
  python gptq_quant.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.3 \
    --bits 4 \
    --group_size 128 \
    --output_dir ./output/mistral7b-gptqmodel-int4

  # CUDA_HOMEが使えて高速カーネル(Marlin等)を試したい場合:
  python gptq_quant.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.3 \
    --load_backend auto \
    --output_dir ./output/mistral7b-gptqmodel-int4
"""
import sys, os, argparse
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from model_utils import (get_model_size_mb, get_num_params, measure_llm_latency,
                          measure_perplexity, default_calibration_texts,
                          save_results, common_output_args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--desc_act", action="store_true",
                   help="活性化順序でのソートを有効化(精度重視。既定はFalseで速度優先)")
    p.add_argument("--num_calib_samples", type=int, default=512,
                   help="キャリブレーションに使用するサンプル総数（トークン結合後のサンプル数になります）")
    p.add_argument("--load_backend", type=str, default="torch",
                   help="量子化済みモデル再ロード時の推論バックエンド(gptqmodel.BACKEND)。")
    common_output_args(p)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from gptqmodel import GPTQModel, QuantizeConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantize_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
    )

    model = GPTQModel.from_pretrained(args.model_path, quantize_config)

    # --- 【修正】十分な量の生テキストを取得し、2048トークンに結合して必要サンプル数を確保する ---
    print("🤖 キャリブレーションデータの結合・生成を開始します...")
    # 結合して2048トークンにするため、多め（サンプル数 × 60倍）の短いテキストをロード
    raw_texts = default_calibration_texts(args.num_calib_samples * 60)
    
    # すべての短いテキストを一度トークナイズして1つの巨大なトークン配列に結合
    all_input_ids = []
    for text in raw_texts:
        if text.strip():
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_input_ids.extend(ids)
            all_input_ids.append(tokenizer.eos_token_id)  # 区切りとしてEOSを付与

    # 2048トークンずつの綺麗なチャンクに切り出し、目標のサンプル数(args.num_calib_samples)を作成
    seqlen = 2048
    calib_dataset = []
    for i in range(0, len(all_input_ids), seqlen):
        chunk = all_input_ids[i : i + seqlen]
        if len(chunk) == seqlen:  # 端数は切り捨て
            calib_dataset.append({
                "input_ids": torch.tensor([chunk]),
                "attention_mask": torch.tensor([[1] * seqlen])
            })
            if len(calib_dataset) >= args.num_calib_samples:
                break

    print(f"✅ 平均コンテキスト長 {seqlen} のサンプルを {len(calib_dataset)} 個作成しました。")
    
    # 結合済みのデータセットを直接渡して量子化（tokenizer=NoneでOK）
    model.quantize(calibration=calib_dataset, tokenizer=None, batch_size=1)
    # --------------------------------------------------------------------------------

    os.makedirs(args.output_dir, exist_ok=True)
    model.save(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 量子化済みモデルを再ロードして評価
    quant_model = GPTQModel.load(args.output_dir, backend=args.load_backend)
    latency = measure_llm_latency(quant_model, tokenizer)
    
    # --- 【修正】評価（Perplexity測定）に使うテキストも、長い文脈に結合して渡す ---
    print("🤖 評価用（Perplexity用）データの結合処理を開始します...")
    eval_raw_texts = default_calibration_texts(100) # 多めに取得
    eval_input_ids = []
    for text in eval_raw_texts:
        if text.strip():
            eval_input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            eval_input_ids.append(tokenizer.eos_token_id)
            
    # 2048トークンずつの評価用チャンクを数個作成し、それをデコードして生テキストのリストに戻す
    eval_texts = []
    seqlen = 2048
    for i in range(0, len(eval_input_ids), seqlen):
        chunk = eval_input_ids[i : i + seqlen]
        if len(chunk) == seqlen:
            eval_texts.append(tokenizer.decode(chunk))
            if len(eval_texts) >= 10: # 評価用には10サンプル（2万トークン以上）あれば十分です
                break
                
    # 結合された高品質なテキストを Perplexity 測定に渡す
    ppl = measure_perplexity(quant_model, tokenizer, eval_texts)
    # --------------------------------------------------------------------------------


    results = {
        "method": "gptq (gptqmodel)",
        "bits": args.bits,
        "group_size": args.group_size,
        "desc_act": args.desc_act,
        "load_backend": args.load_backend,
        "num_params": get_num_params(quant_model),
        "model_size_mb": get_model_size_mb(quant_model),
        "perplexity": ppl,
        **latency,
    }
    save_results(results, args.output_dir)
    print(results)


if __name__ == "__main__":
    main()
