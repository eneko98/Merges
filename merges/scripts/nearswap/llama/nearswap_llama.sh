#!/bin/bash
#SBATCH --job-name=nearswap_llama
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=00:30:00
#SBATCH --array=0-0%4
#SBATCH --output=/scratch/evalero/projects/merge_llm/merges/logs/out/nearswap/llama/%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/merges/logs/err/nearswap/llama/%A_%a.err

################ ENVIRONMENT ################
module purge
module load Miniforge3/24.11.3-2
module load CUDA/12.1.1

export HF_HOME="/scratch/evalero/huggingface_cache"
export HF_TOKEN="${HF_TOKEN}"

source /scratch/evalero/envs/merge_env/bin/activate

################ MODELS ################
BASE_MODEL="/data/evalero/models_original/Latxa3.1_8b_lr1e-5"
INSTRUCT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_FOR_NEARSWAP="meta-llama/Llama-3.1-8B-Instruct"

################ NEARSWAP PARAMETERS ################
T_VALUES=(0.5)
T="${T_VALUES[$SLURM_ARRAY_TASK_ID]}"

################ LAYERS ################
LAYER_START=0
LAYER_END=32

################ NAMING ################
BASE_MODEL_NAME="$(basename "${BASE_MODEL}")"
INSTRUCT_MODEL_NAME="$(basename "${INSTRUCT_MODEL}")"
MODEL_NAME="${BASE_MODEL_NAME}__${INSTRUCT_MODEL_NAME}"

OUT_BASE="/scratch/evalero/projects/merge_llm/merges/models/nearswap/llama"
OUT_DIR="${OUT_BASE}/${MODEL_NAME}"

mkdir -p "$OUT_DIR"

################ YAML ################
YAML_TMP="${SLURM_TMPDIR:-/tmp}/${MODEL_NAME}.yaml"

cat > "$YAML_TMP" <<EOF
slices:
  - sources:
      - model: ${INSTRUCT_MODEL}
        layer_range: [${LAYER_START}, ${LAYER_END}]
      - model: ${BASE_MODEL}
        layer_range: [${LAYER_START}, ${LAYER_END}]

merge_method: nearswap

base_model: ${BASE_MODEL_FOR_NEARSWAP}

parameters:
  t: ${T}

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

echo "NearSwap merge completed: ${MODEL_NAME} (t=${T})"
