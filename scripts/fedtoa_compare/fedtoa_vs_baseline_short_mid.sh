#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../fedtoa/common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

run_level="${RUN_LEVEL:-short}"   # short | mid
dataset="${DATASET:-flickr}"      # flickr | coco

goal="${GOAL:-FedToA-vs-FedCola}"
root="${DATA_ROOT_PREFIX:-}"
out_root="${COMPARE_OUT_DIR:-outputs/fedtoa_compare}"
log_root="${COMPARE_LOG_DIR:-logs/fedtoa_compare}"

ic="${IC:-12}"
tc="${TC:-12}"
mc="${MC:-8}"
cncntrtn="${CNCNTRTN:-0.5}"
c="${C_RATIO:-0.25}"
nt="${NUM_THREAD:-8}"
seed="${SEED:-1}"

loader_workers="${LOADER_NUM_WORKERS:-6}"
loader_prefetch="${LOADER_PREFETCH_FACTOR:-4}"

case "$run_level" in
  short)
    rounds="${ROUNDS:-3}"
    local_epochs="${LOCAL_EPOCHS:-1}"
    ;;
  mid)
    # 不要默认 10x2，先用稳妥中跑
    rounds="${ROUNDS:-5}"
    local_epochs="${LOCAL_EPOCHS:-1}"
    ;;
  *)
    echo "Unsupported RUN_LEVEL=${run_level}. Use short or mid." >&2
    exit 1
    ;;
esac

case "$dataset" in
  flickr)
    dataset_name="Flickr30k"
    datasets=(CIFAR100 AG_NEWS Flickr30k Coco)
    data_paths=("${root}data/cifar100" "${root}data/agnews" "${root}data/flickr30k" "${root}data/coco")
    batch_size="${BATCH_SIZE:-16}"
    eval_batch_size="${EVAL_BATCH_SIZE:-32}"
    ;;
  coco)
    dataset_name="COCO"
    datasets=(CIFAR100 AG_NEWS Coco Coco)
    data_paths=("${root}data/cifar100" "${root}data/agnews" "${root}data/coco" "${root}data/coco")
    batch_size="${BATCH_SIZE:-16}"
    eval_batch_size="${EVAL_BATCH_SIZE:-32}"
    ;;
  *)
    echo "Unsupported DATASET=${dataset}. Use flickr or coco." >&2
    exit 1
    ;;
esac

# 采用你现在已经相对稳定的 FedToA 默认值
beta_topo="${BETA_TOPO:-1e-4}"
gamma_spec="${GAMMA_SPEC:-1.0}"
eta_lip="${ETA_LIP:-0.0}"
warmup_rounds="${WARMUP_ROUNDS:-5}"
warmup_start_beta="${WARMUP_START_BETA:-0.0}"
warmup_mode="${WARMUP_MODE:-linear}"
topk_edges="${TOPK_EDGES:-512}"
var_threshold="${FEDTOA_VAR_THRESHOLD:-0.5}"

mkdir -p "$out_root" "$log_root"
ts="$(fedtoa_ts)"

base_out="${out_root}/fedcola_${dataset}_${run_level}"
fedtoa_out="${out_root}/fedtoa_${dataset}_${run_level}"
mkdir -p "$base_out" "$fedtoa_out"

base_log="${log_root}/fedcola_${dataset}_${run_level}_${ts}.log"
fedtoa_log="${log_root}/fedtoa_${dataset}_${run_level}_${ts}.log"

echo "[COMPARE] launching baseline=fedavg then fedtoa on dataset=${dataset_name} level=${run_level}"
echo "[COMPARE] outputs: baseline=${base_out}, fedtoa=${fedtoa_out}"
echo "[COMPARE] logs: baseline=${base_log}, fedtoa=${fedtoa_log}"

echo "[RUN_CONFIG] script=fedtoa_vs_baseline_short_mid.sh" | tee "$base_log"
echo "[RUN_CONFIG] dataset=${dataset_name}" | tee -a "$base_log"
echo "[RUN_CONFIG] level=${run_level}" | tee -a "$base_log"
echo "[RUN_CONFIG] algorithm=fedavg" | tee -a "$base_log"
echo "[RUN_CONFIG] rounds=${rounds}" | tee -a "$base_log"
echo "[RUN_CONFIG] local_epochs=${local_epochs}" | tee -a "$base_log"
echo "[RUN_CONFIG] batch_size=${batch_size}" | tee -a "$base_log"
echo "[RUN_CONFIG] eval_batch_size=${eval_batch_size}" | tee -a "$base_log"
echo "[RUN_CONFIG] output_dir=${base_out}" | tee -a "$base_log"
echo "[RUN_CONFIG] log_file=${base_log}" | tee -a "$base_log"

python main.py \
  --exp_name FedCola \
  --result_path "$base_out" \
  --log_path "$log_root" \
  --shared_param attn \
  --share_scope modality \
  --colearn_param none \
  --compensation \
  --with_aux \
  --aux_trained \
  --algorithm fedavg \
  --seed "$seed" \
  --multi-task \
  --datasets "${datasets[@]}" \
  --modalities img txt img+txt img+txt \
  --data_paths "${data_paths[@]}" \
  --Ks "$ic" "$tc" "$mc" \
  --Cs "$c" \
  --test_size -1 \
  --split_type diri \
  --cncntrtn "$cncntrtn" \
  --model_name mome_small_patch16 \
  --resize 224 \
  --imnorm \
  --eval_type global \
  --eval_every 1 \
  --eval_metrics acc1 \
  --R "$rounds" \
  --E "$local_epochs" \
  --B "$batch_size" \
  --beta1 0 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr_decay 0.99 \
  --lr_decay_step 1 \
  --criterion CrossEntropyLoss \
  --num_thread "$nt" \
  --loader_num_workers "$loader_workers" \
  --loader_pin_memory \
  --loader_persistent_workers \
  --loader_prefetch_factor "$loader_prefetch" \
  --use_bert_tokenizer \
  --pretrained \
  --goal "$goal" \
  --equal_sampled \
  --eval_batch_size "$eval_batch_size" \
  --no-detect_anomaly \
  2>&1 | tee -a "$base_log"

fedtoa_print_run_config \
  "fedtoa_vs_baseline_short_mid.sh" "$dataset_name" "fedtoa" "$beta_topo" "$gamma_spec" "$eta_lip" \
  "$warmup_rounds" "$warmup_start_beta" "$warmup_mode" "true" "true" "$topk_edges" "$var_threshold" \
  "$fedtoa_out" "$fedtoa_log"

fedtoa_cmd=(
  python main.py
  --exp_name FedToA
  --result_path "$fedtoa_out"
  --log_path "$log_root"
  --shared_param attn
  --share_scope modality
  --colearn_param none
  --compensation
  --with_aux
  --aux_trained
  --algorithm fedtoa
  --seed "$seed"
  --multi-task
  --datasets "${datasets[@]}"
  --modalities img txt img+txt img+txt
  --data_paths "${data_paths[@]}"
  --Ks "$ic" "$tc" "$mc"
  --Cs "$c"
  --test_size -1
  --split_type diri
  --cncntrtn "$cncntrtn"
  --model_name mome_small_patch16
  --resize 224
  --imnorm
  --eval_type global
  --eval_every 1
  --eval_metrics acc1
  --R "$rounds"
  --E "$local_epochs"
  --B "$batch_size"
  --beta1 0
  --optimizer AdamW
  --lr 1e-4
  --lr_decay 0.99
  --lr_decay_step 1
  --criterion CrossEntropyLoss
  --num_thread "$nt"
  --loader_num_workers "$loader_workers"
  --loader_pin_memory
  --loader_persistent_workers
  --loader_prefetch_factor "$loader_prefetch"
  --use_bert_tokenizer
  --pretrained
  --goal "$goal"
  --equal_sampled
  --eval_batch_size "$eval_batch_size"
  --no-detect_anomaly
  --fedtoa_prompt_only
  --freeze_backbone
  --use_topo
  --use_spec
  --use_lip
  --tau 0.2
  --eig_k 4
  --topk_edges "$topk_edges"
  --beta_topo "$beta_topo"
  --fedtoa_topo_warmup_rounds "$warmup_rounds"
  --fedtoa_topo_warmup_start_beta "$warmup_start_beta"
  --fedtoa_topo_warmup_mode "$warmup_mode"
  --gamma_spec "$gamma_spec"
  --eta_lip "$eta_lip"
  --prompt_len 10
  --diagonal_eps 1e-4
)

# 只有在 var_threshold 不是 none / empty 时才传
if [[ -n "${var_threshold}" && "${var_threshold}" != "none" && "${var_threshold}" != "None" ]]; then
  fedtoa_cmd+=(--fedtoa_var_threshold "$var_threshold")
fi

"${fedtoa_cmd[@]}" 2>&1 | tee "$fedtoa_log"

echo "[COMPARE] done."
echo "[COMPARE] baseline log: ${base_log}"
echo "[COMPARE] fedtoa  log: ${fedtoa_log}"
echo "[COMPARE] baseline out: ${base_out}"
echo "[COMPARE] fedtoa  out: ${fedtoa_out}"
