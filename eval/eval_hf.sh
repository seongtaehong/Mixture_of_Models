#!/bin/bash

# HuggingFace 모델 평가 스크립트
export HF_ALLOW_CODE_EVAL=1

# 평가할 모델 목록
models=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Ministral-8B-Instruct-2410"
    "Qwen/Qwen2.5-7B-Instruct"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
    "Qwen/Qwen2.5-Math-7B-Instruct"
    "deepseek-ai/deepseek-math-7b-instruct"
    "BioMistral/BioMistral-7B"
    "nvidia/OpenMath2-Llama3.1-8B"
    "Henrychur/MMed-Llama-3-8B"
    "AdaptLLM/law-chat"
    "Equall/Saul-7B-Instruct-v1"
    "AdaptLLM/medicine-chat"
    "instruction-pretrain/finance-Llama3-8B"
    "AdaptLLM/finance-chat"
    "google/medgemma-4b-it"
    "aaditya/Llama3-OpenBioLLM-8B"
    "MathLLMs/MathCoder-CL-7B"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# 평가 설정
NUM_GPUS=3                  # 동시에 사용할 GPU 수
TASK="mmlu"                 # 평가 태스크 (mmlu, mathqa, medmcqa 등)
OUT_DIR="./output/${TASK}"
mkdir -p "$OUT_DIR"

echo "=== HuggingFace 모델 평가 시작 ==="
echo "태스크: $TASK"
echo "출력 디렉토리: $OUT_DIR"
echo "사용할 GPU 수: $NUM_GPUS"
echo "평가할 모델 수: ${#models[@]}"

pids=()  # 실행 중인 프로세스 ID 배열

for idx in "${!models[@]}"; do
    model="${models[$idx]}"
    gpu_id=$((idx % NUM_GPUS))
    log_file="${OUT_DIR}/$(echo "$model" | tr '/: ' '_').log"
    
    echo ">>> $(date '+%F %T') | 모델 $model을 GPU $gpu_id에서 시작"
    
    # HuggingFace 모델 평가 실행
    CUDA_VISIBLE_DEVICES=$gpu_id \
    lm_eval --model hf \
            --model_args pretrained="$model",max_length=3072 \
            --tasks "$TASK" \
            --batch_size auto \
            --write_out \
            --log_samples \
            --output_path "$OUT_DIR" \
            >"$log_file" 2>&1 &
    
    pids+=($!)
    
    # GPU 수만큼 프로세스가 실행되면 하나가 끝날 때까지 대기
    if (( ${#pids[@]} >= NUM_GPUS )); then
        if wait -n 2>/dev/null; then
            :
        else
            # Bash 4.x 호환성
            wait "${pids[0]}"
        fi
        
        # 완료된 프로세스 제거
        alive=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            fi
        done
        pids=("${alive[@]}")
    fi
done

# 모든 남은 작업 완료 대기
wait
echo "=== 모든 HuggingFace 모델 평가 완료 ==="