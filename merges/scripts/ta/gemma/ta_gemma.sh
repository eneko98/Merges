#!/bin/bash
#SBATCH --job-name=ta_gemma
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=00:30:00
#SBATCH --array=0-0%4
#SBATCH --output=/scratch/evalero/projects/merge_llm/merges/logs/out/ta/gemma/%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/merges/logs/err/ta/gemma/%A_%a.err

################ ENVIRONMENT ################
module purge
module load Miniforge3/24.11.3-2
module load CUDA/12.1.1

export HF_HOME="/scratch/evalero/huggingface_cache"
export HF_TOKEN="${HF_TOKEN}"

source /scratch/evalero/envs/merge_env/bin/activate

################ MODELS ################
BASE_MODEL="google/gemma-3-27b-pt"
MODEL_1="google/gemma-3-27b-it"
MODEL_2="google/translategemma-27b-it"

################ WEIGHTS (ARRAY) ################
MODEL_1_WEIGHTS=(1.0)
MODEL_2_WEIGHT=1.0
BASE_WEIGHT="${MODEL_1_WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

################ NAMING ################
MODEL_1_NAME="$(basename "${MODEL_1}")"
MODEL_2_NAME="$(basename "${MODEL_2}")"
MODEL_NAME="${MODEL_1_NAME}__${MODEL_2_NAME}"

OUT_BASE="/scratch/evalero/projects/merge_llm/merges/models/ta/gemma"
OUT_DIR="${OUT_BASE}/${MODEL_NAME}"
mkdir -p "$OUT_DIR"

################ YAML ################
YAML_TMP="${SLURM_TMPDIR:-/tmp}/${MODEL_NAME}.yaml"

cat > "$YAML_TMP" <<EOF
base_model: ${BASE_MODEL}

models:
  - model: ${MODEL_1}
    parameters:
      weight: ${MODEL_1_WEIGHTS}
  - model: ${MODEL_2}
    parameters:
      weight: ${MODEL_2_WEIGHT}

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
JSON_CONFIG_SRC="/scratch/evalero/json_configs/Gemma3/gemma-3-27b"
cp "${JSON_CONFIG_SRC}"/*.json "${OUT_DIR}/"

echo "Task Arithmetic (gemma) merge completed: ${MODEL_NAME}"