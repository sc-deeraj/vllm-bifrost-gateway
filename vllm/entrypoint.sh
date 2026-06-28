#!/usr/bin/env bash
#
# entrypoint.sh -- assemble vLLM serve flags from env vars, then launch the
# OpenAI-compatible server.
#
# Lives here (not in docker-compose `command:`) because precision selection
# needs real conditional logic that compose variable-substitution can't
# express: FP16 means *no* --quantization flag at all, while FP8/FP4 each
# need a different one.
#
# Driven by (see .env.example for the full list and defaults):
#   BASE_MODEL, BASE_MODEL_ID
#   VLLM_PRECISION          FP16 | FP8 | FP4
#   VLLM_DTYPE              override the 16-bit dtype (default bf16, Qwen-native)
#   VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION, VLLM_REASONING_PARSER
#   VLLM_LANGUAGE_MODEL_ONLY, VLLM_MAX_IMAGES_PER_PROMPT   (vision)
#   VLLM_ENABLE_LORA + VLLM_MAX_LORAS / _RANK / _CPU_LORAS (optional hot-swap)
#   VLLM_FP4_METHOD, VLLM_EXTRA_ARGS
#
set -euo pipefail

MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
SERVED_NAME="${BASE_MODEL_ID:-qwen3.5-9b}"
PRECISION="${VLLM_PRECISION:-FP16}"

# --- precision -> quantization / dtype -----------------------------------
# Qwen3.5 is bf16-native, so the true "full precision" 16-bit path is
# bfloat16; float16 risks overflow on bf16-trained weights. Set
# VLLM_DTYPE=float16 if you specifically need fp16.
precision_args=()
case "${PRECISION^^}" in
  FP16 | BF16 | "")
    precision_args+=(--dtype "${VLLM_DTYPE:-bfloat16}")
    ;;
  FP8)
    # Online dynamic FP8: works on Ada/Hopper with any bf16 checkpoint.
    precision_args+=(--quantization fp8 --dtype "${VLLM_DTYPE:-auto}")
    ;;
  FP4)
    # NVFP4. Native FP4 compute requires a Blackwell GPU (sm_100+), and
    # modelopt_fp4 expects a pre-quantized FP4 checkpoint. On Ada this will
    # not run -- override VLLM_FP4_METHOD / BASE_MODEL accordingly.
    precision_args+=(--quantization "${VLLM_FP4_METHOD:-modelopt_fp4}")
    [ -n "${VLLM_DTYPE:-}" ] && precision_args+=(--dtype "${VLLM_DTYPE}")
    ;;
  *)
    echo "entrypoint: unknown VLLM_PRECISION='${PRECISION}' (use FP16 | FP8 | FP4)" >&2
    exit 1
    ;;
esac

# --- multimodal / vision -------------------------------------------------
mm_args=()
if [ "${VLLM_LANGUAGE_MODEL_ONLY:-false}" = "true" ]; then
  # Skip the vision encoder to free VRAM (text-only serving).
  mm_args+=(--language-model-only)
else
  mm_args+=(--limit-mm-per-prompt "image=${VLLM_MAX_IMAGES_PER_PROMPT:-2}")
fi

# --- optional LoRA hot-swap ----------------------------------------------
# Off by default: the vision path is the focus, and LoRA + quantization
# (FP8/FP4) has known compatibility edge cases. Safe to enable with FP16.
lora_args=()
if [ "${VLLM_ENABLE_LORA:-false}" = "true" ]; then
  lora_args+=(
    --enable-lora
    --max-loras "${VLLM_MAX_LORAS:-2}"
    --max-lora-rank "${VLLM_MAX_LORA_RANK:-16}"
    --max-cpu-loras "${VLLM_MAX_CPU_LORAS:-4}"
  )
fi

set -x
exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --reasoning-parser "${VLLM_REASONING_PARSER:-qwen3}" \
  "${precision_args[@]}" \
  "${mm_args[@]}" \
  "${lora_args[@]}" \
  ${VLLM_EXTRA_ARGS:-}
