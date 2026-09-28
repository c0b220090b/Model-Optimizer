"""
共通ユーティリティ
- LLM / 拡散モデルのロード
- モデルサイズ・メモリの計測
- 簡易ベンチマーク（レイテンシ / Perplexity）

各最適化スクリプトから import して使う。
"""
from __future__ import annotations

import os
import time
import json
import argparse
from dataclasses import dataclass, asdict
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
        torch_dtype=torch_dtype,
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


def get_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_sparsity(model: torch.nn.Module) -> float:
    """モデル全体の重みゼロ率"""
    total, zeros = 0, 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
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
    """簡易 Perplexity 計測（複数テキストの平均）"""
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


def default_calibration_texts(n: int = 128) -> list[str]:
    """簡易キャリブレーション/評価用データ（wikitext がある場合はそちらを推奨）"""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 20][:n]
        return texts
    except Exception:
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
