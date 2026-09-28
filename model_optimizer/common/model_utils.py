"""
共通ユーティリティ
- LLM / 拡散モデルのロード
- モデルサイズ・メモリの計測
- 簡易ベンチマーク（レイテンシ / Perplexity）
- キャリブレーション / 評価データ（WikiText-2: train をキャリブレーション, test を評価に分離）

各最適化スクリプトから import して使う。
"""
from __future__ import annotations

import os
import time
import json
import random
import argparse
import warnings
from typing import Optional

import torch


# ----------------------------------------------------------------------------
# LLM ロード
# ----------------------------------------------------------------------------
def load_llm(model_path: str, dtype: str = "bf16", device_map: str = "auto",
             quantization_config=None, attn_implementation: Optional[str] = None,
             trust_remote_code: bool = True):
    """transformers の causal LM をロードする共通関数"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tokenizer


# ----------------------------------------------------------------------------
# 拡散モデル ロード
# ----------------------------------------------------------------------------
def load_diffusion_pipeline(model_path: str, dtype: str = "fp16", device: str = "cuda"):
    from diffusers import DiffusionPipeline

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype_map.get(dtype, torch.float16))
    pipe = pipe.to(device)
    return pipe


# ----------------------------------------------------------------------------
# サイズ / メモリ計測
# ----------------------------------------------------------------------------
def get_model_size_mb(model: torch.nn.Module) -> float:
    """パラメータ + バッファのメモリ使用量（MB）"""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def get_checkpoint_size_mb(output_dir: str) -> float:
    """保存済みチェックポイントの重みファイル（*.safetensors / *.bin / *.pt）のディスク上サイズ（MB）。
    fake-quant モデルはメモリ上では fp16/bf16 のままなので、量子化の圧縮効果はこちらで見る。"""
    total = 0
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 2)


def get_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_sparsity(model: torch.nn.Module) -> float:
    """モデル全体の重みゼロ率（embedding / norm / lm_head も分母に含む）"""
    total, zeros = 0, 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def get_linear_sparsity(model: torch.nn.Module, exclude: tuple[str, ...] = ("lm_head",)) -> float:
    """Transformer ブロック内の nn.Linear 重みだけのゼロ率（2:4 なら 0.5 になるはず）"""
    total, zeros = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and not any(e in name for e in exclude):
            total += m.weight.numel()
            zeros += (m.weight == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def get_peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ----------------------------------------------------------------------------
# 簡易ベンチマーク
# ----------------------------------------------------------------------------
@torch.no_grad()
def measure_llm_latency(model, tokenizer, prompt: str = "The quick brown fox",
                         max_new_tokens: int = 64, num_runs: int = 5, warmup: int = 2) -> dict:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    reset_peak_memory()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.time() - t0)

    gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    avg_time = sum(times) / len(times)
    return {
        "avg_latency_sec": avg_time,
        "tokens_per_sec": gen_tokens / avg_time if avg_time > 0 else 0,
        "peak_memory_mb": get_peak_memory_mb(),
    }


@torch.no_grad()
def measure_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    """（旧）簡易 Perplexity 計測。後方互換のため残す。
    新しいスクリプトでは measure_perplexity_wikitext() を使うこと。"""
    device = next(model.parameters()).device
    nlls, total_tokens = [], 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        out = model(input_ids, labels=input_ids)
        nlls.append(out.loss.float() * (input_ids.shape[1] - 1))
        total_tokens += input_ids.shape[1] - 1
    if total_tokens == 0:
        return float("nan")
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()


# ----------------------------------------------------------------------------
# キャリブレーション / 評価データ（WikiText-2）
#   キャリブレーション = train split, 評価 = test split に分けてリークを防ぐ
# ----------------------------------------------------------------------------
WIKITEXT_REPO = "Salesforce/wikitext"  # 新しい huggingface_hub では "wikitext" 単体の名前は解決できない


def load_wikitext(split: str):
    from datasets import load_dataset
    return load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split=split)


def load_wikitext_token_ids(tokenizer, split: str) -> torch.Tensor:
    """WikiText-2 の split 全体を "\\n\\n" で結合し、1本のトークン列 (1, N) にする（GPTQ/SparseGPT 論文と同じ作法）"""
    ds = load_wikitext(split)
    return tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids


def get_calib_dataset(tokenizer, num_samples: int = 128, seqlen: int = 2048,
                      device="cuda", seed: int = 0) -> list[dict]:
    """train split からランダム位置で seqlen トークンの窓を num_samples 個切り出す。
    戻り値は modelopt の forward_loop / data_loader にそのまま渡せる dict のリスト。"""
    ids = load_wikitext_token_ids(tokenizer, "train")
    n_tokens = ids.shape[1]
    if n_tokens <= seqlen:
        raise ValueError(f"train split のトークン数 {n_tokens} が seqlen {seqlen} 以下です")

    rng = random.Random(seed)
    dataset = []
    for _ in range(num_samples):
        start = rng.randint(0, n_tokens - seqlen - 1)
        chunk = ids[:, start : start + seqlen].to(device)
        dataset.append({
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk),
        })
    return dataset


@torch.no_grad()
def run_calibration(model, calib_dataset):
    """get_calib_dataset() の各バッチをモデルに流す（フックで統計を取る手法用）"""
    for batch in calib_dataset:
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


@torch.no_grad()
def measure_perplexity_wikitext(model, tokenizer, seqlen: int = 2048,
                                max_chunks: Optional[int] = None, split: str = "test") -> float:
    """WikiText-2 test を seqlen ごとの重ならない窓に分けて Perplexity を計測する。
    トークン ID を直接モデルに入れるので、デコード→再トークナイズや truncation によるズレがない。
    max_chunks=None で test 全体（論文値と比較可能）。"""
    device = next(model.parameters()).device
    ids = load_wikitext_token_ids(tokenizer, split)
    n_chunks = ids.shape[1] // seqlen
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)

    nll_sum, n_tokens = 0.0, 0
    for i in range(n_chunks):
        chunk = ids[:, i * seqlen : (i + 1) * seqlen].to(device)
        loss = model(chunk, labels=chunk).loss.float()
        nll_sum += loss.item() * (seqlen - 1)
        n_tokens += seqlen - 1
    if n_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(nll_sum / n_tokens)))


def default_calibration_texts(n: int = 128) -> list[str]:
    """（旧）簡易キャリブレーション用テキスト。後方互換のため残す。
    新しいスクリプトでは get_calib_dataset() を使うこと。"""
    try:
        ds = load_wikitext("train")
        texts = [t for t in ds["text"] if len(t.strip()) > 20][:n]
        return texts
    except Exception as e:
        warnings.warn(
            f"wikitext の読み込みに失敗したため、ダミーの3文を繰り返したデータで代用します: {e!r}\n"
            "この結果はキャリブレーション・評価ともに信頼できません。"
        )
        base = [
            "人工知能技術は近年急速に発展しており、様々な産業への応用が進んでいる。",
            "The transformer architecture has become the foundation of modern NLP systems.",
            "深層学習モデルの推論コストを削減することは、実運用において重要な課題である。",
        ]
        return (base * (n // len(base) + 1))[:n]


# ----------------------------------------------------------------------------
# 結果保存
# ----------------------------------------------------------------------------
def save_results(results: dict, output_dir: str, filename: str = "result.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")
    return path


def common_output_args(parser: argparse.ArgumentParser):
    parser.add_argument("--output_dir", type=str, required=True, help="出力先ディレクトリ")
    return parser


def common_eval_args(parser: argparse.ArgumentParser):
    """評価条件を全スクリプトで揃えるための共通引数"""
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="キャリブレーション・評価のコンテキスト長")
    parser.add_argument("--eval_max_chunks", type=int, default=None,
                        help="PPL 評価に使う test 窓の上限（None で test 全体）")
    parser.add_argument("--seed", type=int, default=0)
    return parser