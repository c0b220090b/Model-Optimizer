# Model-Optimizer

LLM（大規模言語モデル）および拡散モデル（Diffusion Model）に対して、量子化・枝刈り（プルーニング）・蒸留などの最適化手法を適用し、精度と速度・メモリのトレードオフを検証するためのツールキットです。

各最適化手法は `` 配下に**実行可能なスクリプトとして実装済み**です（雛形ではありません）。

## NVIDIA TensorRT Model Optimizer (nvidia-modelopt) の利用について

本プロジェクトは、NVIDIA が公開している量子化・枝刈り・蒸留・投機的デコーディングの統合ライブラリ
[**NVIDIA TensorRT Model Optimizer**](https://github.com/NVIDIA/TensorRT-Model-Optimizer)（PyPIパッケージ名: `nvidia-modelopt`, import名: `modelopt`）を、**該当する手法では実装の主軸として採用**しています。

modelopt が公式にサポートする範囲は以下の通りで、**この範囲内の手法は modelopt の API (`modelopt.torch.quantization`, `.sparsity`, `.distill`, `.speculative`) を直接呼び出す実装**に更新しました。

| modelopt モジュール | 本プロジェクトで使用している箇所 |
| --- | --- |
| `modelopt.torch.quantization` (`mtq`) | AWQ, SmoothQuant, FP8, KVキャッシュ量子化, 拡散モデル(U-Net/TextEncoder/VAE)のPTQ, 複合適用 |
| `modelopt.torch.sparsity` (`mts`) | Magnitude Pruning, SparseGPT, 2:4構造化スパース（modeloptの重みスパース化は**2:4パターン固定**） |
| `modelopt.torch.distill` (`mtd`) | 知識蒸留（Teacher-Student, `kd_loss`モード） |
| `modelopt.torch.speculative` (`mtsp`) | Speculative Decoding（Medusaヘッドの追加・学習） |

一方で、以下の手法は **modelopt の対象外**であるため、従来通り各手法固有のライブラリ・自前実装を使用しています（理由も明記）。

| 手法 | 実装 | modelopt を使わない理由 |
| --- | --- | --- |
| bitsandbytes INT8/NF4 | bitsandbytes | bitsandbytes 独自の量子化フォーマットであり、modeloptとは別プロジェクト |
| GPTQ | auto-gptq | GPTQアルゴリズム自体はmodeloptに実装されていない（modeloptはAWQ/SmoothQuant/maxキャリブレーションを提供） |
| GGUF量子化 | llama.cpp | llama.cpp独自フォーマットであり、CPU/エッジ推論用の別エコシステム |
| LoRAベース QAT | peft + bitsandbytes | QLoRA方式のQATはPEFTの領域。modeloptにもQAT機能はあるがLoRA統合はPEFT側が主流 |
| LLM-Pruner / Depth Pruning / Attention Head Pruning | 自前実装（Taylor近似, コサイン類似度等） | modeloptの構造化枝刈り(`mcore_minitron`)はNVIDIA Megatron-Core形式のモデルが対象で、一般のHuggingFaceモデルには非対応 |
| SVD低ランク分解 | 自前実装 | modeloptのスコープ外（NAS/量子化/スパース化/蒸留/投機的デコーディングが対象） |
| FlashAttention導入, torch.compile | transformers / PyTorch標準機能 | modeloptの対象外（推論エンジン最適化はmodelopt上位のTensorRT-LLM等が担当） |
| 拡散モデルの構造化チャンネルプルーニング | torch-pruning | modeloptのFastNAS構造化枝刈りは画像分類/セグメンテーション向けCNNバックボーンが主対象で、U-Netの条件付き生成アーキテクチャは公式検証範囲外 |
| Attention Block枝刈り, Progressive Distillation, LCM, LCM-LoRA, ADD, サンプラー比較, Token Merging | diffusers / tomesd 等 | 拡散モデル特有の学習不要高速化・蒸留手法で、いずれもmodeloptではなくdiffusersエコシステム側の技術 |

**重要**: `mtq.quantize()` によるPTQは基本的に「疑似量子化(fake-quant)」でキャリブレーション・精度検証を行うものであり、PyTorch上でそのまま実行しても実際の推論速度は変わりません。実際の高速化を得るには、`modelopt.torch.export` を使って ONNX / TensorRT / TensorRT-LLM 形式にエクスポートする必要があります（本プロジェクトのスクリプトはこの手前の「量子化・精度検証」までをカバーしています）。

全32スクリプト中の内訳: **nvidia-modelopt を直接使用 13本**（量子化4, 枝刈り3, その他LLM3, 拡散モデルPTQ3）／ **各手法固有ライブラリ・自前実装 19本**（下記の各表で実装欄に理由を記載）。

## 対象モデル

検証を素早く回せるよう、軽量モデル〜標準的なモデルまでを段階的に対象とします。

### LLM

| モデル | パラメータ数 | 用途・選定理由 |
| --- | --- | --- |
| TinyLlama-1.1B-Chat | 1.1B | 最適化手法のデバッグ・高速イテレーション用ベースライン |
| Phi-3.5-mini-instruct | 3.8B | エッジ／省メモリ環境向け最適化の検証 |
| Mistral-7B-Instruct-v0.3 | 7B | 標準的な Dense Transformer の代表 |
| Llama-3.1-8B-Instruct | 8B | 最も情報・先行事例が多い基準モデル |
| Qwen2.5-7B-Instruct | 7B | 日本語を含む多言語性能の検証、Llama系との比較用 |

### 拡散モデル（Text-to-Image）

| モデル | アーキテクチャ | 用途・選定理由 |
| --- | --- | --- |
| Stable Diffusion 1.5 | U-Net（Latent Diffusion） | 軽量・情報豊富な定番ベースライン |
| Stable Diffusion XL (SDXL) 1.0 | U-Net（大型） | 高解像度・高品質モデルでの最適化効果を検証 |
| SD-Turbo / SDXL-Turbo | 蒸留済み U-Net | 「既に蒸留済みのモデル」への追加最適化の余地を検証 |
| Stable Diffusion 3 Medium | DiT（Diffusion Transformer） | Transformer ベースの拡散モデルでの最適化検証 |
| ControlNet（SD1.5 base） | U-Net + 条件付け分岐 | 条件付き生成モデルでの最適化への影響を確認 |

## ディレクトリ構成

```
Model-Optimizer/
├── README.md
├── requirements.txt
├── output/                                 # 最適化後モデルの出力先
├── results/                                # ベンチマーク結果(json)の集約先
└── model_optimizer/                        # Pythonパッケージ本体 (__init__.py あり)
    ├── __init__.py
    ├── common/
    │   └── model_utils.py                # ロード/計測/評価の共通関数
    ├── quantization/                      # 1. LLM量子化
    │   ├── bitsandbytes_quant.py
    │   ├── gptq_quant.py
    │   ├── awq_quant.py
    │   ├── smoothquant_quant.py
    │   ├── gguf_quant.sh
    │   ├── fp8_quant.py
    │   ├── kv_cache_quant.py
    │   └── qat_lora.py
    ├── pruning/                            # 2. LLM枝刈り
    │   ├── magnitude_pruning.py
    │   ├── wanda_pruning.py
    │   ├── sparsegpt_pruning.py
    │   ├── llm_pruner.py
    │   ├── depth_pruning.py
    │   ├── attention_head_pruning.py
    │   └── structured_sparsity_2_4.py
    ├── llm_optim/                           # 3. LLMその他の最適化
    │   ├── knowledge_distillation.py
    │   ├── low_rank_decomposition.py
    │   ├── speculative_decoding.py
    │   ├── flash_attention.py
    │   ├── torch_compile_opt.py
    │   └── combined_quant_prune.py
    ├── diffusion_quant_prune/                 # 4. 拡散モデル量子化・枝刈り
    │   ├── unet_ptq_int8.py
    │   ├── text_encoder_quant.py
    │   ├── vae_quant.py
    │   ├── unet_channel_pruning.py
    │   └── attention_block_pruning.py
    └── diffusion_distill/                       # 5. 拡散モデル蒸留・高速サンプリング
        ├── progressive_distillation.py
        ├── lcm_distillation.py
        ├── lcm_lora.py
        ├── add_distillation.py
        ├── sampler_comparison.py
        └── token_merging.py
```

すべてのスクリプトは `--output_dir` を必須引数にとり、最適化後モデル・生成画像・`result.json`（精度/速度/メモリの計測結果）をそこに保存します。

## 動作環境

- Python 3.10+
- PyTorch 2.x
- CUDA 11.8+（GPU 使用時、FP8検証には Ada/Hopper世代推奨、2:4スパースはAmpere以降推奨）
- diffusers, transformers, accelerate, peft
- nvidia-modelopt[hf]（AWQ/SmoothQuant/FP8/KVキャッシュ量子化/2:4スパース/蒸留/Speculative Decoding/拡散モデルPTQ で使用）
- （手法により）bitsandbytes, auto-gptq, torch-pruning, tomesd, llama.cpp

## インストール

```bash
cd Model-Optimizer
pip install -r requirements.txt
```

各パッケージディレクトリ（`common/`, `quantization/`, `pruning/`, `llm_optim/`, `diffusion_quant_prune/`, `diffusion_distill/`）には `__init__.py` があり、`model_optimizer` は通常のPythonパッケージとして構成されています。スクリプトは `model_optimizer/` 内から直接実行できます（例: `cd model_optimizer && python quantization/awq_quant.py ...`）。

GGUF変換（`gguf_quant.sh`）のみ、別途 llama.cpp のクローン・ビルドが必要です（スクリプト内コメント参照）。

## 使い方: 手法一覧と実行コマンド

### 1. LLM — 量子化 (`quantization/`)

| 手法 | 種別 | 実装 | スクリプト | 実行例 |
| --- | --- | --- | --- | --- |
| bitsandbytes INT8/NF4 | PTQ（重みのみ） | bitsandbytes | `bitsandbytes_quant.py` | `python quantization/bitsandbytes_quant.py --model_path meta-llama/Llama-3.1-8B-Instruct --dtype nf4 --output_dir ./output/llama3.1-8b-bnb-nf4` |
| GPTQ | PTQ（キャリブレーション要） | auto-gptq | `gptq_quant.py` | `python quantization/gptq_quant.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --bits 4 --group_size 128 --output_dir ./output/mistral7b-gptq-int4` |
| AWQ | PTQ（重要チャネル保護） | **nvidia-modelopt** (`mtq.INT4_AWQ_CFG`) | `awq_quant.py` | `python quantization/awq_quant.py --model_path meta-llama/Llama-3.1-8B-Instruct --output_dir ./output/llama3.1-8b-modelopt-awq-int4` |
| SmoothQuant | PTQ（重み＋活性化） | **nvidia-modelopt** (`mtq.INT8_SMOOTHQUANT_CFG`) | `smoothquant_quant.py` | `python quantization/smoothquant_quant.py --model_path facebook/opt-1.3b --output_dir ./output/opt1.3b-modelopt-smoothquant-int8` |
| GGUF (Q4_K_M等) | PTQ | llama.cpp | `gguf_quant.sh` | `./quantization/gguf_quant.sh ./models/llama3.1-8b ./llama.cpp ./output/llama3.1-8b-gguf Q4_K_M` |
| FP8 (H100/Ada) | PTQ | **nvidia-modelopt** (`mtq.FP8_DEFAULT_CFG`) | `fp8_quant.py` | `python quantization/fp8_quant.py --model_path meta-llama/Llama-3.1-8B-Instruct --output_dir ./output/llama3.1-8b-modelopt-fp8` |
| KVキャッシュ量子化 | PTQ | **nvidia-modelopt** (`mtq.FP8_KV_CFG`/`NVFP4_KV_CFG`) | `kv_cache_quant.py` | `python quantization/kv_cache_quant.py --model_path meta-llama/Llama-3.1-8B-Instruct --kv_dtype fp8 --context_len 4096 --output_dir ./output/llama3.1-8b-modelopt-kvfp8` |
| LoRAベース QAT (QLoRA) | QAT | peft + bitsandbytes | `qat_lora.py` | `python quantization/qat_lora.py --model_path meta-llama/Llama-3.1-8B-Instruct --num_train_steps 200 --output_dir ./output/llama3.1-8b-qlora` |

### 2. LLM — 枝刈り (`pruning/`)

| 手法 | 種別 | 実装 | スクリプト | 実行例 |
| --- | --- | --- | --- | --- |
| Magnitude Pruning | 2:4構造化（既定）/ 任意率（`--backend custom`） | **nvidia-modelopt** (`mts.sparsify`, mode="sparse_magnitude") | `magnitude_pruning.py` | `python pruning/magnitude_pruning.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --output_dir ./output/mistral7b-modelopt-magnitude-2to4` |
| Wanda | 非構造化（活性化考慮） | 自前実装（modelopt未対応の手法） | `wanda_pruning.py` | `python pruning/wanda_pruning.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --sparsity 0.5 --num_calib_samples 128 --output_dir ./output/mistral7b-wanda-50` |
| SparseGPT | 2:4構造化（既定）/ 任意率（`--backend custom`） | **nvidia-modelopt** (`mts.sparsify`, mode="sparsegpt") | `sparsegpt_pruning.py` | `python pruning/sparsegpt_pruning.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --output_dir ./output/mistral7b-modelopt-sparsegpt-2to4` |
| LLM-Pruner | 構造化（Taylor重要度） | 自前実装（modeloptの`mcore_minitron`はMegatron-Core専用のため） | `llm_pruner.py` | `python pruning/llm_pruner.py --model_path meta-llama/Llama-3.1-8B-Instruct --pruning_ratio 0.2 --output_dir ./output/llama3.1-8b-llmpruner-20` |
| Depth Pruning（層除去） | 構造化 | 自前実装（同上） | `depth_pruning.py` | `python pruning/depth_pruning.py --model_path meta-llama/Llama-3.1-8B-Instruct --num_layers_to_remove 4 --output_dir ./output/llama3.1-8b-depthprune-4layers` |
| Attention Head Pruning | 構造化 | 自前実装（同上） | `attention_head_pruning.py` | `python pruning/attention_head_pruning.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --head_pruning_ratio 0.25 --output_dir ./output/mistral7b-headprune-25` |
| 2:4 構造化スパース性 | 構造化（HW対応） | **nvidia-modelopt**（Magnitude/SparseGPT 2スコアラー比較） | `structured_sparsity_2_4.py` | `python pruning/structured_sparsity_2_4.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --output_dir ./output/mistral7b-modelopt-2to4-comparison` |

### 3. LLM — その他の最適化 (`llm_optim/`)

| 手法 | 実装 | スクリプト | 実行例 |
| --- | --- | --- | --- |
| 知識蒸留（Teacher-Student） | **nvidia-modelopt** (`mtd.convert`, mode="kd_loss") | `knowledge_distillation.py` | `python llm_optim/knowledge_distillation.py --teacher_model meta-llama/Llama-3.1-8B-Instruct --student_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --num_train_steps 500 --output_dir ./output/tinyllama-modelopt-distilled` |
| 低ランク分解 (SVD) | 自前実装（modeloptのスコープ外） | `low_rank_decomposition.py` | `python llm_optim/low_rank_decomposition.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --rank_ratio 0.5 --output_dir ./output/mistral7b-svd-r50` |
| Speculative Decoding | **nvidia-modelopt** (`mtsp.convert`, mode="medusa") | `speculative_decoding.py` | `python llm_optim/speculative_decoding.py --model_path meta-llama/Llama-3.1-8B-Instruct --num_medusa_heads 4 --num_train_steps 300 --output_dir ./output/llama3.1-8b-modelopt-medusa` |
| FlashAttention 導入 | transformers標準機能 | `flash_attention.py` | `python llm_optim/flash_attention.py --model_path meta-llama/Llama-3.1-8B-Instruct --output_dir ./output/llama3.1-8b-flashattn-compare` |
| torch.compile / CUDA Graph | PyTorch標準機能 | `torch_compile_opt.py` | `python llm_optim/torch_compile_opt.py --model_path meta-llama/Llama-3.1-8B-Instruct --compile_mode reduce-overhead --output_dir ./output/llama3.1-8b-compile-compare` |
| 量子化×枝刈りの複合適用 | **nvidia-modelopt**（`mts.sparsify` → `mtq.quantize`） | `combined_quant_prune.py` | `python llm_optim/combined_quant_prune.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --sparsify_mode sparse_magnitude --quant_cfg FP8_DEFAULT_CFG --output_dir ./output/mistral7b-modelopt-2to4-fp8` |

### 4. 拡散モデル — 量子化・枝刈り (`diffusion_quant_prune/`)

| 手法 | 実装 | スクリプト | 実行例 |
| --- | --- | --- | --- |
| U-Net の PTQ（INT8/FP8） | **nvidia-modelopt** (`mtq.quantize`) | `unet_ptq_int8.py` | `python diffusion_quant_prune/unet_ptq_int8.py --model_path runwayml/stable-diffusion-v1-5 --output_dir ./output/sd1.5-modelopt-unet-int8` |
| Text Encoder の量子化 | **nvidia-modelopt** (`mtq.quantize`) | `text_encoder_quant.py` | `python diffusion_quant_prune/text_encoder_quant.py --model_path stabilityai/stable-diffusion-xl-base-1.0 --output_dir ./output/sdxl-modelopt-textencoder-int8` |
| VAE の量子化 | **nvidia-modelopt** (`mtq.quantize`) | `vae_quant.py` | `python diffusion_quant_prune/vae_quant.py --model_path runwayml/stable-diffusion-v1-5 --output_dir ./output/sd1.5-modelopt-vae-int8` |
| U-Net チャンネルプルーニング | torch-pruning（modeloptのFastNASはCV分類モデル向けのため） | `unet_channel_pruning.py` | `python diffusion_quant_prune/unet_channel_pruning.py --model_path runwayml/stable-diffusion-v1-5 --pruning_ratio 0.2 --output_dir ./output/sd1.5-unet-channelprune-20` |
| Attention Block の枝刈り | 自前実装（Wanda方式, modelopt未対応） | `attention_block_pruning.py` | `python diffusion_quant_prune/attention_block_pruning.py --model_path runwayml/stable-diffusion-v1-5 --sparsity 0.4 --output_dir ./output/sd1.5-attn-pruned-40` |

### 5. 拡散モデル — ステップ数削減・蒸留系 (`diffusion_distill/`)

modelopt は拡散モデルのステップ蒸留・学習不要高速化を対象としていないため、本カテゴリは全て diffusers / tomesd エコシステム側の実装です。

| 手法 | 実装 | スクリプト | 実行例 |
| --- | --- | --- | --- |
| Progressive Distillation | 自前実装（diffusers） | `progressive_distillation.py` | `python diffusion_distill/progressive_distillation.py --model_path runwayml/stable-diffusion-v1-5 --teacher_steps 32 --num_train_steps 500 --output_dir ./output/sd1.5-progressive-16steps` |
| LCM (Latent Consistency Model) | 自前実装（diffusers） | `lcm_distillation.py` | `python diffusion_distill/lcm_distillation.py --model_path runwayml/stable-diffusion-v1-5 --num_train_steps 500 --output_dir ./output/sd1.5-lcm` |
| LCM-LoRA | diffusers（公開LoRAアダプタの適用） | `lcm_lora.py` | `python diffusion_distill/lcm_lora.py --model_path stabilityai/stable-diffusion-xl-base-1.0 --lcm_lora_path latent-consistency/lcm-lora-sdxl --num_inference_steps 4 --output_dir ./output/sdxl-lcm-lora` |
| ADD（SD-Turbo方式） | 自前実装（diffusers） | `add_distillation.py` | `python diffusion_distill/add_distillation.py --model_path runwayml/stable-diffusion-v1-5 --num_train_steps 500 --output_dir ./output/sd1.5-add-1step` |
| 高速サンプラー比較 | diffusers標準スケジューラ | `sampler_comparison.py` | `python diffusion_distill/sampler_comparison.py --model_path runwayml/stable-diffusion-v1-5 --steps_list 10,20,50 --output_dir ./output/sd1.5-sampler-comparison` |
| Token Merging (ToMe) | tomesd | `token_merging.py` | `python diffusion_distill/token_merging.py --model_path runwayml/stable-diffusion-v1-5 --merge_ratio 0.5 --output_dir ./output/sd1.5-tome-50` |

## 評価軸

各スクリプトは実行後に `<output_dir>/result.json` として以下を記録します。

- **精度**：LLM は Perplexity（簡易wikitext評価）、拡散モデルは PSNR / CLIP Score / 生成画像
- **速度**：レイテンシ、トークン/秒 または 画像生成秒数
- **メモリ**：モデルサイズ(MB)、推論時ピークメモリ(MB)
- **圧縮率 / スパース率**：元モデル比でのパラメータ数・ファイルサイズ・ゼロ率

## 補足

- 共通処理（モデルロード・レイテンシ計測・Perplexity計測・結果保存）は `common/model_utils.py` にまとめてあり、各スクリプトから import して使用します。
- Wanda / SparseGPT / LLM-Pruner / Depth Pruning / Attention Head Pruning / SmoothQuant / 2:4スパースなど、公式実装が公開されていない、または大規模な手法は、論文アルゴリズムに基づく**簡易実装**です。研究目的で厳密な再現性が必要な場合は、各手法の公式リポジトリの利用も検討してください。
- 知識蒸留・Progressive Distillation・LCM蒸留・ADD蒸留の学習系スクリプトは、少数ステップでの動作確認を目的とした最小構成です。実運用レベルの品質を得るには、データセット規模・学習ステップ数を大きくする必要があります。

## ロードマップ

- [ ] 各モデル×各手法の組み合わせでベンチマークを網羅的に実施し `results/` に集約
- [ ] vLLM / TensorRT-LLM / ONNX Runtime / TensorRT へのデプロイ検証（共通デプロイ最適化）
- [ ] 自動最適化パイプライン（精度・速度のトレードオフ探索、Pareto解の自動探索）

## ライセンス

(ライセンスを記載してください)

## 貢献

Issue や Pull Request は歓迎します。貢献方法の詳細は `CONTRIBUTING.md`（準備中）を参照してください。
