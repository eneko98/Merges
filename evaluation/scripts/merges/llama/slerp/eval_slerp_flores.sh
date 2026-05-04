#!/bin/bash
#SBATCH --job-name=eval_slerp_flores
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=24:00:00
#SBATCH --array=0-0%4
#SBATCH --output=/scratch/evalero/projects/merge_llm/evaluation/logs/out/merges/llama/slerp/slerp_flores_%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/evaluation/logs/err/merges/llama/slerp/slerp_flores_%A_%a.err

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
export HF_TOKEN="${HF_TOKEN}"
export TRANSFORMERS_CACHE="/scratch/evalero/huggingface_cache/transformers"
export VLLM_USE_V1=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false

############ MODELS ############
MODELS=(
  "slerp_EU100_INS100|/data/evalero/projects/merge_llm/merges/models/slerp/llama/slerp_EU100_INS100"
)

############ SELECT MODEL ############
IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r model_name model_path <<< "${MODELS[$IDX]}"

echo "Running FLORES tasks for SLERP merge: ${model_name}"

############ OUTPUT DIRS ############
base_dir="/scratch/evalero/projects/merge_llm/evaluation"
results_dir="${base_dir}/results/merges/llama/slerp/${model_name}/flores"
mkdir -p "$results_dir"

############ FLORES TASKS ############
tasks_selected=(
  "flores_eu-en"
  "flores_en-eu"
  "flores_eu-es"
  "flores_es-eu"
  "flores_gl-es"
  "flores_es-gl"
  "flores_gl-en"
  "flores_en-gl"
  "flores_ca-es"
  "flores_es-ca"
  "flores_ca-en"
  "flores_en-ca"
  "flores_en-es"
  "flores_es-en"
)

############ LOOP ############
for task in "${tasks_selected[@]}"; do
  echo "=== Evaluating ${model_name} on ${task} ==="

  model_args="pretrained=${model_path},dtype=bfloat16,tensor_parallel_size=1,max_model_len=18192,gpu_memory_utilization=0.8,max_num_seqs=1,trust_remote_code=True"

  srun python3 -m lm_eval \
      --model vllm \
      --model_args "${model_args}" \
      --tasks "${task}" \
      --device cuda \
      --output_path "${results_dir}/${task}.json" \
      --apply_chat_template \
      --fewshot_as_multiturn \
      --batch_size auto \
      --num_fewshot 5 \
      --log_samples
done
