#!/usr/bin/env bash
# tools/docker_run.sh
# Run a command (default: a shell) in the training image on a GPU node, as the calling user, with the code,
# checkpoints, logs, token streams and JAX cache mounted (docs/launch.md, section 5).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v docker &>/dev/null; then
  echo "Error: docker command not found in PATH" >&2
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-ttt-h100:latest}"
CODE_DIR="${CODE_DIR:-${ROOT_DIR}}"
DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/tokens}"
CKPT_DIR="${CKPT_DIR:-${CODE_DIR}/ckpt}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/log}"
# Not JAX_*: those variables are passed into the container
CACHE_DIR="${CACHE_DIR:-${HOME}/.jax_cache}"

# Ensure mount sources exist on host so the container runtime won't create them as root
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${CACHE_DIR}"
# ... and the mount points in the repository, when an override put a source elsewhere (a link is kept as it is)
for d in "${CODE_DIR}/ckpt" "${CODE_DIR}/log"; do
  [[ -e "${d}" || -L "${d}" ]] || mkdir -p "${d}"
done

# Docker runs containers as root unless told otherwise: as the calling user, the files written to the mounts are
# the user's, and pkill / kill -USR1 on the host reach the training process (section 7).
RUN_ARGS=(--rm --init --ipc=host --net=host --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER
          --ulimit memlock=-1:-1 --ulimit stack=67108864)
# --user sets only the primary group: the others, which shared directories and group ACLs may need, one by one
for gid in $(id -G); do
  RUN_ARGS+=(--group-add "${gid}")
done

# Forward NCCL, TTT, XLA, JAX and WANDB variables by name: the values are read from this environment and never
# appear on a command line (ps would show WANDB_API_KEY). The image sets the cache directory itself.
for var in $(compgen -e | grep -E '^(NCCL_|TTT_|XLA_|JAX_|WANDB_)|^CUDA_VISIBLE_DEVICES$' \
             | grep -vx 'JAX_COMPILATION_CACHE_DIR'); do
  RUN_ARGS+=(-e "${var}")
done

if [[ -d "/dev/infiniband" ]]; then
  RUN_ARGS+=(--device /dev/infiniband)
  echo "[docker_run] Network:       InfiniBand / RoCE device (/dev/infiniband) detected and mounted" >&2
else
  echo "[docker_run] Network:       no /dev/infiniband, NCCL falls back to TCP sockets" >&2
fi

# ckpt and data/tokens may be links to other file systems: their real paths are mounted at the linked names
RUN_ARGS+=(
  -v "${CODE_DIR}:/workspace"
  -v "${CKPT_DIR}:/workspace/ckpt"
  -v "${LOG_DIR}:/workspace/log"
  -v "${CACHE_DIR}:/jax_cache"
  -w /workspace
)

if [[ -d "${DATA_DIR}" ]]; then
  [[ -e "${CODE_DIR}/data/tokens" || -L "${CODE_DIR}/data/tokens" ]] || mkdir -p "${CODE_DIR}/data/tokens"
  RUN_ARGS+=(-v "${DATA_DIR}:/workspace/data/tokens:ro")
  echo "[docker_run] Data:          ${DATA_DIR} -> /workspace/data/tokens (read-only)" >&2
else
  echo "[docker_run] Data:          ${DATA_DIR} does not exist, not mounted" >&2
fi

# An interactive session only from a terminal: a launch with its output in a log file (section 5) has none
if [[ -t 0 && -t 1 ]]; then
  RUN_ARGS+=(-it)
fi

echo "[docker_run] Image:         ${IMAGE_NAME}" >&2
echo "[docker_run] Code mount:    ${CODE_DIR} -> /workspace" >&2
echo "[docker_run] Checkpoint:    ${CKPT_DIR} -> /workspace/ckpt" >&2
echo "[docker_run] Logs:          ${LOG_DIR} -> /workspace/log" >&2
echo "[docker_run] JAX Cache:     ${CACHE_DIR} -> /jax_cache" >&2

# Without arguments the image's CMD runs: an interactive bash
exec docker run "${RUN_ARGS[@]}" "${IMAGE_NAME}" "$@"
