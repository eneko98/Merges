#!/bin/bash
#SBATCH --job-name=linear_llama_weight_sweep
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=/scratch/evalero/projects/merge_llm/merges/logs/out/linear/llama/weight_sweep_%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/merges/logs/err/linear/llama/weight_sweep_%A_%a.err

################ ENVIRONMENT ################
module purge
module load Miniforge3/24.11.3-2
module load CUDA/12.1.1

export HF_HOME="/scratch/evalero/huggingface_cache"
export HF_TOKEN="${HF_TOKEN}"

source /scratch/evalero/envs/merge_env/bin/activate

################ EXPERIMENT CONFIG ################

# ---- MODELS (easy to change later) ----
BASE_MODEL="/data/evalero/models_original/gl_Llama-3.1-8B"
INSTRUCT_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# ---- WEIGHTS ----
BASE_WEIGHTS=(0.5 1.0 1.5 2.0)
#BASE_WEIGHTS=(0.5)
INSTRUCT_WEIGHT=1.0

BASE_WEIGHT="${BASE_WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

# ---- OUTPUT ----
OUT_BASE="/scratch/evalero/projects/merge_llm/merges/models/linear/llama"
MODEL_NAME="linear_GL${BASE_WEIGHT}_INS${INSTRUCT_WEIGHT}"
OUT_DIR="${OUT_BASE}/${MODEL_NAME}"

# ---- TEMP YAML ----
YAML_TMP="/tmp/${MODEL_NAME}.yaml"

# ---- JSON CONFIG ----
JSON_CONFIG_SRC="/scratch/evalero/json_configs/Llama-3.1"

mkdir -p "$OUT_DIR"

################ YAML GENERATION ################
cat > "$YAML_TMP" <<EOF
models:
  - {model: ${BASE_MODEL}, parameters: {weight: ${BASE_WEIGHT}}}
  - {model: ${INSTRUCT_MODEL}, parameters: {weight: ${INSTRUCT_WEIGHT}}}

merge_method: linear
dtype: bfloat16
EOF

echo "========================================"
echo " Running linear merge"
echo " Base model weight:      ${BASE_WEIGHT}"
echo " Instruct model weight:  ${INSTRUCT_WEIGHT}"
echo " Output:                ${OUT_DIR}"
echo "========================================"

################ MERGE ################
mergekit-yaml "$YAML_TMP" "$OUT_DIR" \
  --copy-tokenizer \
  --cuda \
  --low-cpu-memory \
  --out-shard-size 5B \
  --lazy-unpickle \
  --allow-crimes && \
cp "${JSON_CONFIG_SRC}"/*.json "${OUT_DIR}/"

echo "Merge completed for ${MODEL_NAME}"
