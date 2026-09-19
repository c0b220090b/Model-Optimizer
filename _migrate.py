"""scripts/ 配下の実装を model_optimizer/ パッケージへ移植するワンショット移行スクリプト"""
import re
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# (旧パス, 新パス)
MOVES = [
    ("scripts/quantization/01_bitsandbytes_quant.py", "model_optimizer/quantization/bitsandbytes_quant.py"),
    ("scripts/quantization/02_gptq_quant.py", "model_optimizer/quantization/gptq_quant.py"),
    ("scripts/quantization/03_awq_quant.py", "model_optimizer/quantization/awq_quant.py"),
    ("scripts/quantization/04_smoothquant_quant.py", "model_optimizer/quantization/smoothquant_quant.py"),
    ("scripts/quantization/06_fp8_quant.py", "model_optimizer/quantization/fp8_quant.py"),
    ("scripts/quantization/07_kv_cache_quant.py", "model_optimizer/quantization/kv_cache_quant.py"),
    ("scripts/quantization/08_qat_lora.py", "model_optimizer/quantization/qat_lora.py"),

    ("scripts/pruning/01_magnitude_pruning.py", "model_optimizer/pruning/magnitude_pruning.py"),
    ("scripts/pruning/02_wanda_pruning.py", "model_optimizer/pruning/wanda_pruning.py"),
    ("scripts/pruning/03_sparsegpt_pruning.py", "model_optimizer/pruning/sparsegpt_pruning.py"),
    ("scripts/pruning/04_llm_pruner.py", "model_optimizer/pruning/llm_pruner.py"),
    ("scripts/pruning/05_depth_pruning.py", "model_optimizer/pruning/depth_pruning.py"),
    ("scripts/pruning/06_attention_head_pruning.py", "model_optimizer/pruning/attention_head_pruning.py"),
    ("scripts/pruning/07_structured_sparsity_2_4.py", "model_optimizer/pruning/structured_sparsity_2_4.py"),

    ("scripts/other_llm/01_knowledge_distillation.py", "model_optimizer/llm_optim/knowledge_distillation.py"),
    ("scripts/other_llm/02_low_rank_decomposition.py", "model_optimizer/llm_optim/low_rank_decomposition.py"),
    ("scripts/other_llm/03_speculative_decoding.py", "model_optimizer/llm_optim/speculative_decoding.py"),
    ("scripts/other_llm/04_flash_attention.py", "model_optimizer/llm_optim/flash_attention.py"),
    ("scripts/other_llm/05_torch_compile.py", "model_optimizer/llm_optim/torch_compile_opt.py"),
    ("scripts/other_llm/06_combined_quant_prune.py", "model_optimizer/llm_optim/combined_quant_prune.py"),

    ("scripts/diffusion_quant_prune/01_unet_ptq_int8.py", "model_optimizer/diffusion_quant_prune/unet_ptq_int8.py"),
    ("scripts/diffusion_quant_prune/02_text_encoder_quant.py", "model_optimizer/diffusion_quant_prune/text_encoder_quant.py"),
    ("scripts/diffusion_quant_prune/03_vae_quant.py", "model_optimizer/diffusion_quant_prune/vae_quant.py"),
    ("scripts/diffusion_quant_prune/04_unet_channel_pruning.py", "model_optimizer/diffusion_quant_prune/unet_channel_pruning.py"),
    ("scripts/diffusion_quant_prune/05_attention_block_pruning.py", "model_optimizer/diffusion_quant_prune/attention_block_pruning.py"),

    ("scripts/diffusion_distill/01_progressive_distillation.py", "model_optimizer/diffusion_distill/progressive_distillation.py"),
    ("scripts/diffusion_distill/02_lcm_distillation.py", "model_optimizer/diffusion_distill/lcm_distillation.py"),
    ("scripts/diffusion_distill/03_lcm_lora.py", "model_optimizer/diffusion_distill/lcm_lora.py"),
    ("scripts/diffusion_distill/04_add_distillation.py", "model_optimizer/diffusion_distill/add_distillation.py"),
    ("scripts/diffusion_distill/05_sampler_comparison.py", "model_optimizer/diffusion_distill/sampler_comparison.py"),
    ("scripts/diffusion_distill/06_token_merging.py", "model_optimizer/diffusion_distill/token_merging.py"),
]

# sys.path.append(...) + "from model_utils import (...)" ブロックを
# "from model_optimizer.common.model_utils import (...)" に置換する
PATTERN = re.compile(
    r'sys\.path\.append\(os\.path\.join\(os\.path\.dirname\(__file__\), "\.\.", "common"\)\)\n'
    r'from model_utils import (\([^)]*\)|[^\n]*)',
)

def transform(src: str) -> str:
    def repl(m):
        imported = m.group(1)
        return f"from model_optimizer.common.model_utils import {imported}"
    return PATTERN.sub(repl, src)


def main():
    os.makedirs(os.path.join(ROOT, "model_optimizer", "common"), exist_ok=True)
    # common/model_utils.py はそのままコピー
    with open(os.path.join(ROOT, "scripts", "common", "model_utils.py"), encoding="utf-8") as f:
        common_src = f.read()
    with open(os.path.join(ROOT, "model_optimizer", "common", "model_utils.py"), "w", encoding="utf-8") as f:
        f.write(common_src)

    for old, new in MOVES:
        old_path = os.path.join(ROOT, old)
        new_path = os.path.join(ROOT, new)
        with open(old_path, encoding="utf-8") as f:
            src = f.read()
        new_src = transform(src)
        if "sys.path.append" in new_src and "common" in new_src:
            print(f"[警告] 変換しきれていない可能性: {old}")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(new_src)
        print(f"[OK] {old} -> {new}")


if __name__ == "__main__":
    main()
