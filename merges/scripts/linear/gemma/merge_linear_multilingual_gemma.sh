#!/bin/bash
#SBATCH --job-name=linear_multi_llama
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/evalero/projects/merge_llm/merges/logs/out/linear/llama/%j.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/merges/logs/err/linear/llama/%j.err

################ ENVIRONMENT ################
module purge
module load Miniforge3/24.11.3-2
module load CUDA/12.1.1

export HF_HOME="/scratch/evalero/huggingface_cache"
export HF_TOKEN="${HF_TOKEN}"

source /scratch/evalero/envs/merge_env/bin/activate

################ MODELS ################
# Order MUST match TAGS and WEIGHTS

MODELS=(
  "/data/evalero/models_original/cat_Llama-3.1-8B"
  "/data/evalero/models_original/es_Llama-3.1-8B"
  "/data/evalero/models_original/Latxa3.1_8b_lr1e-5"
  "/data/evalero/models_original/gl_Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
)

TAGS=(
  "CAT"
  "ES"
  "EU"
  "GL"
  "INS"
)

WEIGHTS=(
  0.25
  0.25
  0.25
  0.25
  1.0
)

################ SANITY CHECK ################
if [ "${#MODELS[@]}" -ne "${#TAGS[@]}" ] || [ "${#MODELS[@]}" -ne "${#WEIGHTS[@]}" ]; then
  echo "❌ MODELS / TAGS / WEIGHTS length mismatch"
  exit 1
fi

################ NAMING ################
NAME_PARTS=()

for i in "${!TAGS[@]}"; do
  W_INT=$(printf "%.0f" "$(echo "${WEIGHTS[$i]} * 1" | bc)")
  NAME_PARTS+=("${TAGS[$i]}${W_INT}")
done

MODEL_NAME="linear_$(IFS=_; echo "${NAME_PARTS[*]}")"

OUT_BASE="/scratch/evalero/projects/merge_llm/merges/models/linear/llama"
OUT_DIR="${OUT_BASE}/${MODEL_NAME}"
mkdir -p "$OUT_DIR"

echo "🧠 Output model: ${MODEL_NAME}"

################ YAML ################
YAML_TMP="${SLURM_TMPDIR:-/tmp}/${MODEL_NAME}.yaml"

{
  echo "models:"
  for i in "${!MODELS[@]}"; do
    cat <<EOF
  - model: ${MODELS[$i]}
    parameters:
      weight: ${WEIGHTS[$i]}
EOF
  done

  cat <<EOF

merge_method: linear
dtype: bfloat16
EOF
} > "$YAML_TMP"

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

echo "✅ Linear multilingual merge completed:"
echo "   ${MODEL_NAME}"
