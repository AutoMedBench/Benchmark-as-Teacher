#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

required=(
  MODEL_PATH TRAIN_FILE TRAIN_MANIFEST OUTPUT_DIR EXPERIMENT_NAME
  BAT_RUNTIME_TOOL_FORMAT BAT_SANDBOX_IMAGE
  BAT_CORE_AGENTIC_WORKSPACE_ROOT BAT_CORE_AGENTIC_RECEIPTS_ROOT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-python}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-${GPUS_PER_NODE}}"
TARGET_STEPS="${TARGET_STEPS:-50}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-7}"
KL_COEF="${KL_COEF:-0.006}"
SAVE_FREQ="${SAVE_FREQ:-5}"
CHECKPOINTS_TO_KEEP="${CHECKPOINTS_TO_KEEP:-3}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-auto}"
RESUME_MODE="${RESUME_MODE:-disable}"
LOGGER="${LOGGER:-console}"

export BAT_EXPECTED_ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS}"
export BAT_LOG_POOL_REWARD_AVERAGES=true

"${PYTHON_BIN}" - "${MODEL_PATH}" "${TRAIN_FILE}" "${TRAIN_MANIFEST}" "${OUTPUT_DIR}" \
  "${BAT_CORE_AGENTIC_WORKSPACE_ROOT}" "${BAT_CORE_AGENTIC_RECEIPTS_ROOT}" \
  "${GPUS_PER_NODE}" "${ROLLOUT_REPLICAS}" <<'PY'
import json
import os
import sys
from pathlib import Path

model, train, manifest, output, workspace, receipts = map(Path, sys.argv[1:7])
for path, kind in ((model, "dir"), (train, "file"), (manifest, "file"), (workspace, "dir"), (receipts, "dir")):
    resolved = path.resolve(strict=True)
    if path.is_symlink() or (kind == "dir" and not resolved.is_dir()) or (kind == "file" and not resolved.is_file()):
        raise SystemExit(f"invalid launch path: {path}")
if workspace.resolve() == receipts.resolve() or workspace.resolve().parent != receipts.resolve().parent:
    raise SystemExit("sandbox workspace and receipt roots must be distinct siblings")
output.resolve().mkdir(parents=True, exist_ok=True)
gpus, replicas = map(int, sys.argv[7:9])
if gpus < 1 or replicas != gpus:
    raise SystemExit("reference topology requires one rollout replica per GPU")
payload = json.loads(manifest.read_text(encoding="utf-8"))
if payload.get("artifact_kind") != "bat_three_pool_round" or payload.get("rows_per_batch") != 8:
    raise SystemExit("training manifest is not an admitted BaT round")
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/verify_bat_core_dataset.py" \
  --train "${TRAIN_FILE}" --manifest "${TRAIN_MANIFEST}" >/dev/null
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_rl_train_leakage.py" "${TRAIN_FILE}" >/dev/null
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_rl_tool_protocol.py" "${TRAIN_FILE}" \
  --runtime-tool-format "${BAT_RUNTIME_TOOL_FORMAT}" >/dev/null

hydra_args=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=false"
  "algorithm.norm_adv_by_std_in_grpo=false"
  "critic.enable=false"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=true"
  "actor_rollout_ref.actor.use_kl_loss=true"
  "actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
  "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
  "actor_rollout_ref.actor.ppo_mini_batch_size=8"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.shuffle=false"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.n=4"
  "actor_rollout_ref.rollout.temperature=0.7"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.multi_turn.enable=true"
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=${REPO_ROOT}/config/bat_core_execute_code_tool.yaml"
  "actor_rollout_ref.rollout.multi_turn.format=${BAT_RUNTIME_TOOL_FORMAT}"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${REPO_ROOT}/config/bat_core_agent_loop.yaml"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.prompt_key=prompt"
  "data.reward_fn_key=data_source"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.train_batch_size=8"
  "data.val_batch_size=1"
  "data.shuffle=false"
  "data.truncation=left"
  "data.return_raw_chat=true"
  "reward.custom_reward_function.path=${REPO_ROOT}/rl/grpo/core_reward.py"
  "reward.custom_reward_function.name=compute_score_with_breakdown"
  "reward.reward_manager.source=importlib"
  "reward.reward_manager.name=CoreRewardManager"
  "reward.reward_manager.module.path=${REPO_ROOT}/rl/grpo/core_reward_manager.py"
  "reward.reward_model.enable=false"
  "trainer.project_name=benchmark-as-teacher"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.logger=${LOGGER}"
  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
  "trainer.total_epochs=1"
  "trainer.total_training_steps=${TARGET_STEPS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=${CHECKPOINTS_TO_KEEP}"
  "trainer.max_critic_ckpt_to_keep=${CHECKPOINTS_TO_KEEP}"
  "trainer.test_freq=-1"
  "trainer.val_before_train=false"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.default_local_dir=${OUTPUT_DIR}"
)

set +e
"${PYTHON_BIN}" -m rl.grpo.run_verl_core "${hydra_args[@]}" "$@"
training_rc=$?
set -e

if [[ -n "${TRAINING_HEALTH_PATH:-}" ]]; then
  "${PYTHON_BIN}" - "${TRAINING_HEALTH_PATH}" "${training_rc}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1]).resolve()
returncode = int(sys.argv[2])
payload = {"schema_version": 1, "status": "passed" if returncode == 0 else "failed", "returncode": returncode}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False, mode="w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, path)
PY
fi
exit "${training_rc}"
