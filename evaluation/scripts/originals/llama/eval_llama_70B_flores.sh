#!/bin/bash
#SBATCH --job-name=eval_llama_70B_flores
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=499G
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100-sxm4
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=/scratch/evalero/projects/merge_llm/evaluation/logs/out/originals/llama/eval_llama_70B_flores_%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/evaluation/logs/err/originals/llama/eval_llama_70B_flores_%A_%a.err

############ ENVIRONMENT ############
module purge
module load zlib/.1.2.13-GCCcore-12.3.0
module load binutils/.2.40-GCCcore-12.3.0
module load Miniforge3/24.11.3-2
module load CUDA/12.1.1
module load GCC/12.3.0
module load GCCcore/12.3.0

LM_HARNESS_VENV="/scratch/evalero/envs/lm_eval_env"
source "${LM_HARNESS_VENV}/bin/activate"

export HF_HOME="/scratch/evalero/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/evalero/huggingface_cache/transformers"
export HF_TOKEN="${HF_TOKEN}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
export VLLM_USE_V1=1

############ MODELS ############
MODELS=(
  "Llama-3.1-70B-Instruct|meta-llama/Llama-3.1-70B-Instruct|instr"
  "Latxa-Llama-3.1-70B-Instruct|HiTZ/Latxa-Llama-3.1-70B-Instruct|instr"
)

############ SELECT CURRENT MODEL ############
IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r model_name model_path kind <<< "${MODELS[$IDX]}"

echo "Running FLORES tasks for model: ${model_name} (${kind}) from ${model_path}"

############ OUTPUT DIR ############
base_dir="/scratch/evalero/projects/merge_llm/evaluation"
results_dir="${base_dir}/results/originals/llama/${model_name}/flores"
mkdir -p "$results_dir"

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

  model_args="pretrained=${model_path},dtype=bfloat16,enable_thinking=False,tensor_parallel_size=4,max_model_len=8192,gpu_memory_utilization=0.9,max_num_seqs=1"

  srun python3 -m lm_eval \
    --model vllm \
    --model_args "${model_args}" \
    --tasks "${task}" \
    --device cuda \
    --output_path "${results_dir}/${task}.json" \
    --batch_size "${batch_size}" \
    --num_fewshot "${num_fewshot}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn
done
