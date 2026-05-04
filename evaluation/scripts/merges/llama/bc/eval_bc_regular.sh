#!/bin/bash
#SBATCH --job-name=eval_bc_regular
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
#SBATCH --output=/scratch/evalero/projects/merge_llm/evaluation/logs/out/merges/llama/bc/bc_regular_%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/evaluation/logs/err/merges/llama/bc/bc_regular_%A_%a.err

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
export TOKENIZERS_PARALLELISM=false

############ MODELS TO EVALUATE ############
# All merges are INSTRUCT models → no conditional logic needed
# You can append more merges later and just update the array range.

MODELS=(
  #"bc_EU100_INS100|/data/evalero/projects/merge_llm/merges/models/bc/llama/bc_EU100_INS100"
  "bc_CAT100_INS100|/data/evalero/projects/merge_llm/merges/models/bc/llama/bc_CAT100_INS100"
  #"bc_ES100_INS100|/data/evalero/projects/merge_llm/merges/models/bc/llama/bc_ES100_INS100"
  #"bc_GL100_INS100|/data/evalero/projects/merge_llm/merges/models/bc/llama/bc_GL100_INS100"
)

############ SELECT MODEL ############
IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r model_name model_path <<< "${MODELS[$IDX]}"

echo "Running REGULAR tasks for BC merge: ${model_name}"

############ OUTPUT DIRECTORIES ############
base_dir="/scratch/evalero/projects/merge_llm/evaluation"
results_dir="${base_dir}/results/merges/llama/bc/${model_name}"
mkdir -p "$results_dir"

############ TASK LIST ############
tasks_selected=(

  # Basque
  "eus_reading"
  "belebele_eus_Latn"
  "eus_proficiency"
  "eus_trivia"
  "eus_exams_eu"
  "xstorycloze_eu"
  "mgsm_native_cot_eu"
  "bertaqa_eu_global"
  "bertaqa_eu_local"
  "arc_eu_challenge"
  "arc_eu_challenge_mc"
  "openbookqa_eu_mc"

  # Galician
  "belebele_glg_Latn"
  "parafrases_gl"
  "openbookqa_gl"
  "openbookqa_gl_mc"
  "galcola"
  "xstorycloze_gl"
  "mgsm_native_cot_gl"

  # Catalan
  "belebele_cat_Latn"
  "catcola"
  "openbookqa_ca"
  "openbookqa_ca_mc"
  "arc_ca_challenge"
  "arc_ca_challenge_mc"
  "mgsm_native_cot_ca"
  "xstorycloze_ca"

  # English
  "belebele_eng_Latn"
  "bertaqa_en_global"
  "bertaqa_en_local"
  "mgsm_native_cot_en"
  "xstorycloze_en"
  "openbookqa_mc"

  # Spanish
  "belebele_spa_Latn"
  "mgsm_native_cot_es"
  "escola"
  "openbookqa_es"
  "openbookqa_es_mc"
  "xstorycloze_es"
)

############ EVALUATION LOOP ############
for task in "${tasks_selected[@]}"; do
  echo "=== Evaluating ${model_name} on ${task} ==="

  # Few-shot selection
  if [[ "$task" == *mgsm* ]]; then
    num_fewshot=5
  elif [[ "$task" == *xnli* || "$task" == *xstory* ]]; then
    num_fewshot=0
  else
    num_fewshot=5
  fi

  # Batch size policy
  if [[ $task == belebele* || $task == xstory* || "$task" == "eus_reading" || "$task" == "eus_trivia" ]]; then
    batch_size=1
  else
    batch_size=auto
  fi

  # Memory safety
  if [[ "$task" == "eus_reading" ]]; then
    model_args="pretrained=${model_path},dtype=bfloat16,tensor_parallel_size=1,max_model_len=18192,gpu_memory_utilization=0.5,max_num_seqs=1"
  else
    model_args="pretrained=${model_path},dtype=bfloat16,tensor_parallel_size=1,max_model_len=18192,gpu_memory_utilization=0.8,max_num_seqs=1"
  fi

  srun python3 -m lm_eval \
      --model vllm \
      --model_args "${model_args}" \
      --tasks "${task}" \
      --device cuda \
      --output_path "${results_dir}/${task}.json" \
      --apply_chat_template \
      --fewshot_as_multiturn \
      --batch_size "${batch_size}" \
      --num_fewshot "${num_fewshot}" \
      --log_samples \
      --include_path /scratch/evalero/repos/LatxaTxat/evaluation/tasks
done
