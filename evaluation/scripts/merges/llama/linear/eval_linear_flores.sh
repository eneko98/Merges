#!/bin/bash
#SBATCH --job-name=eval_linear_flores
##SBATCH --account=hitz-exclusive
##SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
##SBATCH --constraint=a100-sxm4
#SBATCH --constraint=a100-pcie
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/evalero/projects/merge_llm/evaluation/logs/out/merges/llama/linear/linear_flores_%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/evaluation/logs/err/merges/llama/linear/linear_flores_%A_%a.err

############ ENVIRONMENT ############
module purge
#module load zlib/.1.2.13-GCCcore-12.3.0
#module load binutils/.2.40-GCCcore-12.3.0
#module load Miniforge3/24.11.3-2
module load CUDA/12.1.1
#module load GCC/12.3.0
#module load GCCcore/12.3.0

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
  #"Latxa3.1_8b_lr1e-5__Llama-3.1-8B-Instruct|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/Latxa3.1_8b_lr1e-5__Llama-3.1-8B-Instruct"
  #"Latxa3.1_8b_lr1e-5__Llama-3.3-8B-Instruct-Unofficial|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/Latxa3.1_8b_lr1e-5__Llama-3.3-8B-Instruct-Unofficial"
  #"Latxa3.1_8b_lr1e-5__Llama-3.3-8B-Instruct-128K|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/Latxa3.1_8b_lr1e-5__Llama-3.3-8B-Instruct-128K"
  #"linear_CAT1_ES1_EU1_GL1_INS1|/data/evalero/projects/merge_llm/merges/models/linear/llama/linear_CAT1_ES1_EU1_GL1_INS1"
  #"linear_CAT025_ES025_EU025_GL025_INS1|/data/evalero/projects/merge_llm/merges/models/linear/llama/linear_CAT025_ES025_EU025_GL025_INS1"

  #"linear_EU0.5_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_EU0.5_INS1.0"
  #"linear_EU1.0_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_EU1.0_INS1.0"
  #"linear_EU1.5_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_EU1.5_INS1.0"
  #"linear_EU2.0_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_EU2.0_INS1.0"

  #"linear_CAT0.5_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_CAT0.5_INS1.0"
  #"linear_CAT1.0_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_CAT1.0_INS1.0"
  #"linear_CAT1.5_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_CAT1.5_INS1.0"
  #"linear_CAT2.0_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_CAT2.0_INS1.0"

  #"linear_ES0.5_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_ES0.5_INS1.0"
  #"linear_ES1.0_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_ES1.0_INS1.0"
  #"linear_ES1.5_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_ES1.5_INS1.0"
  #"linear_ES2.0_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_ES2.0_INS1.0"

  #"linear_GL0.5_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_GL0.5_INS1.0"
  #"linear_GL1.0_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_GL1.0_INS1.0"
  #"linear_GL1.5_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_GL1.5_INS1.0"
  #"linear_GL2.0_INS1.0|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_GL2.0_INS1.0"

  "linear_ta_EU025_GL025_CA025_ES025|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_ta_EU025_GL025_CA025_ES025"
  "linear_ta_EU050_GL050_CA050_ES050|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_ta_EU050_GL050_CA050_ES050"
  "linear_ta_EU075_GL075_CA075_ES075|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_ta_EU075_GL075_CA075_ES075"
  "linear_ta_EU100_GL100_CA100_ES100|/scratch/evalero/projects/merge_llm/merges/models/linear/llama/linear_ta_EU100_GL100_CA100_ES100"

)

############ SELECT MODEL ############
IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r model_name model_path <<< "${MODELS[$IDX]}"

echo "Running FLORES tasks for LINEAR merge: ${model_name}"

############ OUTPUT DIRS ############
base_dir="/scratch/evalero/projects/merge_llm/evaluation"
results_dir="${base_dir}/results/merges/llama/linear/${model_name}/flores"
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
