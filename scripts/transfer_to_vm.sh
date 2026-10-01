#!/usr/bin/env bash
# Transfer the NeuroPager repo + existing checkpoints to a Linux VM, preserving
# the exact nested checkpoint path (experiments/generalization-experiment-2/
# checkpoints/) so scripts/generalization_experiment_2.py's relative-path
# resolution (Path("experiments/generalization-experiment-2"), resolved
# against CWD -- see check further down for how this was verified) finds all
# existing checkpoints on the VM without any code change.
#
# Never touches checkpoint *contents* -- this only copies files.
#
# Usage:
#   scripts/transfer_to_vm.sh <VM_USER> <VM_IP>
#
# Run from Git Bash, MSYS2, or WSL on the Windows source machine.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <VM_USER> <VM_IP>" >&2
    exit 1
fi

VM_USER="$1"
VM_IP="$2"
VM="${VM_USER}@${VM_IP}"
REMOTE_ROOT="neuropager"

if ! command -v rsync >/dev/null 2>&1; then
    echo "FAIL: rsync not found on PATH." >&2
    echo "  Git Bash/MSYS2: install via 'pacman -S rsync' in the MSYS2 shell, or use WSL instead." >&2
    echo "  WSL: 'sudo apt install rsync' inside the WSL distro." >&2
    exit 1
fi

# REPO_ROOT is derived from this script's own location via cd+pwd, which
# already reports paths in whatever convention the CURRENT shell uses
# (/c/... under Git Bash/MSYS2, /cygdrive/c/... under Cygwin, /mnt/c/...
# under WSL) -- rsync invoked from that same shell interprets those paths
# consistently by construction, so no manual prefix table is needed. This
# is only an informational check to flag the shell flavor for the log.
case "$(uname -s 2>/dev/null || echo unknown)" in
    MSYS*|MINGW*) echo "shell: Git Bash / MSYS2" ;;
    CYGWIN*)      echo "shell: Cygwin" ;;
    Linux*)       echo "shell: WSL / Linux" ;;
    *)            echo "shell: unknown ($(uname -s 2>/dev/null || echo '?'))" ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "repo root: ${REPO_ROOT}"
echo "target:    ${VM}:~/${REMOTE_ROOT}"
echo

echo "--- creating remote directory structure ---"
ssh "${VM}" "mkdir -p ~/${REMOTE_ROOT}/experiments/generalization-experiment-2/checkpoints ~/${REMOTE_ROOT}/logs"

echo
echo "--- transferring code + reproducibility files ---"
rsync -avz --progress \
    "${REPO_ROOT}/src" \
    "${REPO_ROOT}/scripts" \
    "${REPO_ROOT}/pyproject.toml" \
    "${REPO_ROOT}/README.md" \
    "${REPO_ROOT}/LICENSE" \
    "${REPO_ROOT}/requirements-lock.txt" \
    "${VM}:~/${REMOTE_ROOT}/"

echo
echo "--- transferring checkpoints (preserving nested path) ---"
rsync -avz --progress \
    "${REPO_ROOT}/experiments/generalization-experiment-2/checkpoints/" \
    "${VM}:~/${REMOTE_ROOT}/experiments/generalization-experiment-2/checkpoints/"

echo
echo "--- transferring historical logs (best-effort) ---"
shopt -s nullglob
LOG_FILES=("${REPO_ROOT}"/../tmp/generalization_experiment_2_full_v*.log /tmp/generalization_experiment_2_full_v*.log)
shopt -u nullglob
if [ ${#LOG_FILES[@]} -eq 0 ]; then
    echo "WARN: no historical log files matched /tmp/generalization_experiment_2_full_v*.log -- skipping, continuing."
else
    rsync -avz --progress "${LOG_FILES[@]}" "${VM}:~/${REMOTE_ROOT}/logs/"
fi

echo
echo "--- verifying checkpoint transfer integrity ---"
LOCAL_COUNT=$(find "${REPO_ROOT}/experiments/generalization-experiment-2/checkpoints" -name '*.json' -type f | wc -l | tr -d ' ')
REMOTE_COUNT=$(ssh "${VM}" "find ~/${REMOTE_ROOT}/experiments/generalization-experiment-2/checkpoints -name '*.json' -type f | wc -l" | tr -d ' \r')

echo "local  *.json count:  ${LOCAL_COUNT}"
echo "remote *.json count:  ${REMOTE_COUNT}"

if [ "${LOCAL_COUNT}" != "${REMOTE_COUNT}" ]; then
    echo "FAIL: local/remote checkpoint counts do not match -- do not proceed until resolved." >&2
    exit 1
fi

echo
echo "TRANSFER COMPLETE: ${LOCAL_COUNT}/${LOCAL_COUNT} checkpoint files verified on ${VM}."
echo "Next: ssh ${VM}, then run scripts/vm_preflight.sh from ~/${REMOTE_ROOT}."
