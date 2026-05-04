#!/bin/bash
#SBATCH --job-name=linear_from_ta_sweep
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=00:45:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/evalero/projects/merge_llm/merges/logs/out/linear/llama/ta_linear_sweep_%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/merges/logs/err/linear/llama/ta_linear_sweep_%A_%a.err

################ ENVIRONMENT ################
module purge
module load Miniforge3/24.11.3-2
module load CUDA/12.1.1

export HF_HOME="/scratch/evalero/huggingface_cache"
export HF_TOKEN="${HF_TOKEN}"

source /scratch/evalero/envs/merge_env/bin/activate

################ TA MODELS ################
TA_EU="/data/evalero/projects/merge_llm/merges/models/ta/llama/ta_EU100_INS100"
TA_GL="/data/evalero/projects/merge_llm/merges/models/ta/llama/ta_GL100_INS100"
TA_CA="/data/evalero/projects/merge_llm/merges/models/ta/llama/ta_CAT100_INS100"
TA_ES="/data/evalero/projects/merge_llm/merges/models/ta/llama/ta_ES100_INS100"

################ SWEEP WEIGHTS ################
# One job per weight (same weight applied to EU/GL/CA/ES for that run)
MERGE_WEIGHTS=(0.25 0.50 0.75 1.0)
W="${MERGE_WEIGHTS[$SLURM_ARRAY_TASK_ID]}"

# Pretty strings for naming (avoid trailing zeros issues)
W_STR=$(printf "%.2f" "$W")   # e.g. 0.25, 1.00
W_TAG=$(echo "$W_STR" | sed 's/\.//g')  # e.g. 025, 100

################ OUTPUT ################
OUT_BASE="/scratch/evalero/projects/merge_llm/merges/models/linear/llama"
MODEL_NAME="linear_ta_EU${W_TAG}_GL${W_TAG}_CA${W_TAG}_ES${W_TAG}"
OUT_DIR="${OUT_BASE}/${MODEL_NAME}"
mkdir -p "$OUT_DIR"

################ YAML ################
YAML_TMP="${SLURM_TMPDIR:-/tmp}/${MODEL_NAME}.yaml"

cat > "$YAML_TMP" <<EOF
models:
  - model: ${TA_EU}
    parameters:
      weight: ${W}
  - model: ${TA_GL}
    parameters:
      weight: ${W}
  - model: ${TA_CA}
    parameters:
      weight: ${W}
  - model: ${TA_ES}
    parameters:
      weight: ${W}

merge_method: linear
dtype: bfloat16
EOF

echo "========================================"
echo " Linear merge over TA models (sweep)"
echo " Weight (EU/GL/CA/ES): ${W}"
echo " Output: ${OUT_DIR}"
echo " YAML: ${YAML_TMP}"
echo "========================================"

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

echo "Linear-from-TA sweep merge completed: ${MODEL_NAME}"
