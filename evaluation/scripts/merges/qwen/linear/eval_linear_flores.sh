#!/bin/bash
#SBATCH --job-name=eval_linear_flores
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
#SBATCH --output=/scratch/evalero/projects/merge_llm/evaluation/logs/out/merges/qwen/linear/8B/linear_flores_%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/evaluation/logs/err/merges/qwen/linear/8B/linear_flores_%A_%a.err

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
  "eu_Qwen3-8B-Base__Qwen3-8B|/data/evalero/projects/merge_llm/merges/models/linear/qwen/8B/eu_Qwen3-8B-Base__Qwen3-8B"
  #"es_Qwen3-8B-Base__Qwen3-8B|/scratch/evalero/projects/merge_llm/merges/models/linear/qwen/es_Qwen3-8B-Base__Qwen3-8B"
  #"gl_Qwen3-8B-Base__Qwen3-8B|/scratch/evalero/projects/merge_llm/merges/models/linear/qwen/gl_Qwen3-8B-Base__Qwen3-8B"
  #"cat_Qwen3-8B-Base__Qwen3-8B|/scratch/evalero/projects/merge_llm/merges/models/linear/qwen/cat_Qwen3-8B-Base__Qwen3-8B"
  #"eu_Qwen3-14B-Base__Qwen3-14B|/scratch/evalero/projects/merge_llm/merges/models/linear/qwen/eu_Qwen3-14B-Base__Qwen3-14B"
  #"es_Qwen3-14B-Base__Qwen3-14B|/scratch/evalero/projects/merge_llm/merges/models/linear/qwen/es_Qwen3-14B-Base__Qwen3-14B"
  #"gl_Qwen3-14B-Base__Qwen3-14B|/scratch/evalero/projects/merge_llm/merges/models/linear/qwen/gl_Qwen3-14B-Base__Qwen3-14B"
  #"cat_Qwen3-14B-Base__Qwen3-14B|/scratch/evalero/projects/merge_llm/merges/models/linear/qwen/cat_Qwen3-14B-Base__Qwen3-14B"
  #"linear_CAT1_ES1_EU1_GL1_INS1_8B|/data/evalero/projects/merge_llm/merges/models/linear/qwen/8B/linear_CAT1_ES1_EU1_GL1_INS1"
  #"linear_CAT025_ES025_EU025_GL025_INS1_8B|/data/evalero/projects/merge_llm/merges/models/linear/qwen/8B/linear_CAT025_ES025_EU025_GL025_INS1"
  #"linear_CAT025_ES025_EU025_GL025_INS1_14B|/data/evalero/projects/merge_llm/merges/models/linear/qwen/14B/linear_CAT025_ES025_EU025_GL025_INS1"
)

############ SELECT MODEL ############
IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r model_name model_path <<< "${MODELS[$IDX]}"

echo "Running FLORES tasks for LINEAR merge: ${model_name}"

############ OUTPUT DIRS ############
base_dir="/scratch/evalero/projects/merge_llm/evaluation"
results_dir="${base_dir}/results/merges/qwen/linear/${model_name}/flores"
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

############ LOOP ############
for task in "${tasks_selected[@]}"; do
  echo "===== Evaluating ${model_name} on ${task} ====="

  num_fewshot=5
  batch_size=auto

  model_args="pretrained=${model_path},dtype=bfloat16,enable_thinking=True,tensor_parallel_size=1,max_model_len=18192,think_end_token='</think>',gpu_memory_utilization=0.8,max_num_seqs=1,reasoning_parser=qwen3,trust_remote_code=True"

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
