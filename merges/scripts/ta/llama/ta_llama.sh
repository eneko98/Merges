#!/bin/bash
#SBATCH --job-name=ta_llama
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=00:30:00
#SBATCH --array=0-0%4
#SBATCH --output=/scratch/evalero/projects/merge_llm/merges/logs/out/ta/llama/%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/merges/logs/err/ta/llama/%A_%a.err

################ ENVIRONMENT ################
module purge
module load Miniforge3/24.11.3-2
module load CUDA/12.1.1

export HF_HOME="/scratch/evalero/huggingface_cache"
export HF_TOKEN="${HF_TOKEN}"

source /scratch/evalero/envs/merge_env/bin/activate

################ MODELS ################
BASE_MODEL="/data/evalero/models_original/cat_Llama-3.1-8B"
INSTRUCT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_FOR_TA="meta-llama/Llama-3.1-8B"

################ WEIGHTS (ARRAY) ################
BASE_WEIGHTS=(1.0)
INSTRUCT_WEIGHT=1.0
BASE_WEIGHT="${BASE_WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

################ NAMING ################
BASE_MODEL_NAME="$(basename "${BASE_MODEL}")"
INSTRUCT_MODEL_NAME="$(basename "${INSTRUCT_MODEL}")"
MODEL_NAME="${BASE_MODEL_NAME}__${INSTRUCT_MODEL_NAME}"

OUT_BASE="/scratch/evalero/projects/merge_llm/merges/models/ta/llama"
OUT_DIR="${OUT_BASE}/${MODEL_NAME}"
mkdir -p "$OUT_DIR"

################ YAML ################
YAML_TMP="${SLURM_TMPDIR:-/tmp}/${MODEL_NAME}.yaml"

cat > "$YAML_TMP" <<EOF
base_model: ${BASE_MODEL_FOR_TA}

models:
  - model: ${BASE_MODEL}
    parameters:
      weight: ${BASE_WEIGHT}
  - model: ${INSTRUCT_MODEL}
    parameters:
      weight: ${INSTRUCT_WEIGHT}

merge_method: task_arithmetic
dtype: bfloat16
EOF

################ MERGE ################
mergekit-yaml "$YAML_TMP" "$OUT_DIR" \
  --copy-tokenizer \
  --cuda \
  --low-cpu-memory \
  --out-shard-size 5B \
  --lazy-unpickle \
  --allow-crimes

################ JSON CONFIG ################
JSON_CONFIG_SRC="/scratch/evalero/json_configs/Llama-3.1"
cp "${JSON_CONFIG_SRC}"/*.json "${OUT_DIR}/"

echo "Task Arithmetic (llama) merge completed: ${MODEL_NAME}"
