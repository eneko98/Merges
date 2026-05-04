#!/bin/bash
#SBATCH --job-name=eval_qwen_flores
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-sxm4
##SBATCH --constraint=a100-pcie
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/evalero/projects/merge_llm/evaluation/logs/out/originals/qwen/qwen3.6_flores_%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/evaluation/logs/err/originals/qwen/qwen3.6_flores_%A_%a.err

set -euo pipefail

############ ENVIRONMENT ############
module purge

LM_HARNESS_VENV="/scratch/evalero/envs/vllm019"
source "${LM_HARNESS_VENV}/bin/activate"

export HF_HOME="/scratch/evalero/huggingface_cache"
# export TRANSFORMERS_CACHE="/scratch/evalero/huggingface_cache/transformers"
export HF_TOKEN="${HF_TOKEN}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

############ MODELS TO EVALUATE ############
# Format:
# model_name|model_path|kind|gpus_needed|tensor_parallel_size|max_model_len|gpu_memory_utilization
MODELS=(
  # ===== 1 GPU =====
  #"Qwen3-8B-Base|Qwen/Qwen3-8B-Base|base|1|1|18192|0.8"
  #"Qwen3-8B|Qwen/Qwen3-8B|instr|1|1|18192|0.8"
  #"Qwen3.5-0.8B-Base|Qwen/Qwen3.5-0.8B-Base|base|1|1|18192|0.8"
  #"Qwen3.5-2B-Base|Qwen/Qwen3.5-2B-Base|base|1|1|18192|0.8"
  #"Qwen3.5-4B-Base|Qwen/Qwen3.5-4B-Base|base|1|1|18192|0.8"
  #"Qwen3.5-9B-Base|Qwen/Qwen3.5-9B-Base|base|1|1|18192|0.8"
  #"Qwen3.5-0.8B|Qwen/Qwen3.5-0.8B|instr|1|1|18192|0.8"
  #"Qwen3.5-2B|Qwen/Qwen3.5-2B|instr|1|1|18192|0.8"
  #"Qwen3.5-4B|Qwen/Qwen3.5-4B|instr|1|1|18192|0.8"
  #"Qwen3.5-9B|Qwen/Qwen3.5-9B|instr|1|1|18192|0.8"
  #"Nemotron-Cascade-8B|nvidia/Nemotron-Cascade-8B|instr|1|1|18192|0.8"
  #"Nemotron-Cascade-8B-Thinking|nvidia/Nemotron-Cascade-8B-Thinking|instr|1|1|18192|0.8"
  #"cat_Qwen3-8B-Base|/data/evalero/models_original/cat_Qwen3-8B-Base|base|1|1|18192|0.8"
  #"es_Qwen3-8B-Base|/data/evalero/models_original/es_Qwen3-8B-Base|base|1|1|18192|0.8"
  #"eu_Qwen3-8B-Base|/data/evalero/models_original/eu_Qwen3-8B-Base|base|1|1|18192|0.8"
  #"gl_Qwen3-8B-Base|/data/evalero/models_original/gl_Qwen3-8B-Base|base|1|1|18192|0.8"
  #"latxa-qwen3:8b|/data/evalero/models_original/latxa-qwen3:8b|instr|1|1|18192|0.8"

  # ===== 2 GPU =====
  #"Nemotron-Cascade-14B-Thinking|nvidia/Nemotron-Cascade-14B-Thinking|instr|2|2|18192|0.8"
  #"Qwen3-14B-Base|Qwen/Qwen3-14B-Base|base|2|2|18192|0.8"
  #"Qwen3-14B|Qwen/Qwen3-14B|instr|2|2|18192|0.8"
  #"cat_Qwen3-14B-Base|/data/evalero/models_original/cat_Qwen3-14B-Base|base|2|2|18192|0.8"
  #"es_Qwen3-14B-Base|/data/evalero/models_original/es_Qwen3-14B-Base|base|2|2|18192|0.8"
  #"eu_Qwen3-14B-Base|/data/evalero/models_original/eu_Qwen3-14B-Base|base|2|2|18192|0.8"
  #"gl_Qwen3-14B-Base|/data/evalero/models_original/gl_Qwen3-14B-Base|base|2|2|18192|0.8"
  #"Qwen3.5-27B|Qwen/Qwen3.5-27B|instr|2|2|18192|0.8"
  #"latxa-qwen3:32b|/data/evalero/models_original/latxa-qwen3:32b|instr|2|2|18192|0.8"
  #"32b.multi.eu.gl.ca|/data/evalero/models_original/32b.multi.eu.gl.ca|instr|2|2|18192|0.8"
  "Qwen3.6-35B-A3B|Qwen/Qwen3.6-35B-A3B|instr|2|2|18192|0.8"
  "Qwen3.6-27B|Qwen/Qwen3.6-27B|instr|2|2|18192|0.8"
)

############ SELECT CURRENT MODEL ############
IDX=${SLURM_ARRAY_TASK_ID:-0}

if [[ "${IDX}" -ge "${#MODELS[@]}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${IDX} is out of range. MODELS has ${#MODELS[@]} entries."
  exit 1
fi

IFS='|' read -r model_name model_path kind gpus_needed tp_size max_model_len gpu_mem_util <<< "${MODELS[$IDX]}"

echo "========================================"
echo "Running FLORES tasks"
echo "Model: ${model_name}"
echo "Path: ${model_path}"
echo "Kind: ${kind}"
echo "GPUs needed: ${gpus_needed}"
echo "Tensor parallel size: ${tp_size}"
echo "Max model len: ${max_model_len}"
echo "GPU memory util: ${gpu_mem_util}"
echo "========================================"

############ SAFETY CHECK ############
allocated_gpus="${SLURM_GPUS_ON_NODE:-1}"
if [[ "${allocated_gpus}" -lt "${gpus_needed}" ]]; then
  echo "ERROR: ${model_name} needs ${gpus_needed} GPUs but this allocation only has ${allocated_gpus}."
  exit 1
fi

############ OUTPUT DIRECTORIES ############
base_dir="/scratch/evalero/projects/merge_llm/evaluation"
results_dir="${base_dir}/results/originals/qwen/${model_name}/flores"
mkdir -p "${results_dir}"

############ FLORES TASKS ############
tasks_selected=(
  "basque_bench_flores_eu-en"
  "basque_bench_flores_en-eu"
  "basque_bench_flores_eu-es"
  "spanish_bench_flores_es-eu"
  "galician_bench_flores_gl-es"
  "spanish_bench_flores_es-gl"
  "galician_bench_flores_gl-en"
  "galician_bench_flores_en-gl"
  "catalan_bench_flores_ca-es"
  "spanish_bench_flores_es-ca"
  "catalan_bench_flores_ca-en"
  "catalan_bench_flores_en-ca"
  "spanish_bench_flores_en-es"
  "spanish_bench_flores_es-en"
)

############ EVALUATION LOOP ############
for task in "${tasks_selected[@]}"; do
  echo "===== Evaluating ${model_name} on ${task} ====="

  num_fewshot=5
  batch_size=auto

  model_args="pretrained=${model_path},dtype=bfloat16,enable_thinking=False,tensor_parallel_size=${tp_size},max_model_len=${max_model_len},gpu_memory_utilization=${gpu_mem_util},max_num_seqs=1,trust_remote_code=True"

  extra_flags=()
  if [[ "$kind" == "instr" ]]; then
    extra_flags+=(--apply_chat_template --fewshot_as_multiturn)
  fi

  srun python3 -m lm_eval \
    --model vllm \
    --model_args "${model_args}" \
    --tasks "${task}" \
    --device cuda \
    --output_path "${results_dir}/${task}.json" \
    --batch_size "${batch_size}" \
    --num_fewshot "${num_fewshot}" \
    --include_path /scratch/evalero/repos/LatxaTxat/evaluation/tasks \
    --log_samples \
    "${extra_flags[@]}"
done