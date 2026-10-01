#!/usr/bin/env bash
# Preflight check for generalization-experiment-2 on a new VM. Backs up the
# transferred checkpoints (never modifies them), validates every checkpoint's
# schema/config_hash, confirms this environment computes the same config_hash
# as the source machine, and prints GO/NO-GO.
#
# Does NOT launch the experiment.
#
# Usage (run on the VM, from the repo root, e.g. ~/neuropager):
#   scripts/vm_preflight.sh

set -uo pipefail

CHECKPOINT_DIR="experiments/generalization-experiment-2/checkpoints"
BACKUP_DIR="${HOME}/checkpoints_backup_$(date +%Y%m%d_%H%M%S)"

echo "=== NeuroPager VM preflight ==="
echo

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "NO-GO: ${CHECKPOINT_DIR} does not exist -- are you running this from the repo root?" >&2
    exit 1
fi

echo "--- environment snapshot ---"
python --version 2>&1 || echo "python not found on PATH"
echo "vCPUs: $(nproc 2>/dev/null || echo '?')"
free -h 2>/dev/null || echo "free not available"
df -h "${HOME}" 2>/dev/null
echo

echo "--- backing up checkpoints (read-only w.r.t. the originals) ---"
mkdir -p "${BACKUP_DIR}"
cp -r "${CHECKPOINT_DIR}" "${BACKUP_DIR}/"
BACKUP_COUNT=$(find "${BACKUP_DIR}" -name '*.json' -type f | wc -l | tr -d ' ')
echo "backed up ${BACKUP_COUNT} checkpoint file(s) to ${BACKUP_DIR}"
echo

GO=1

echo "--- running validate_checkpoints.py ---"
if python scripts/validate_checkpoints.py; then
    echo "validate_checkpoints.py: PASS"
else
    echo "validate_checkpoints.py: FAIL"
    GO=0
fi
echo

echo "--- running check_config_hash.py ---"
if PYTHONPATH=src python scripts/check_config_hash.py; then
    echo "check_config_hash.py: PASS"
else
    echo "check_config_hash.py: FAIL"
    GO=0
fi
echo

echo "=== RESULT ==="
if [ "${GO}" -eq 1 ]; then
    echo "GO -- checkpoints validated, config_hash confirmed portable, backup at ${BACKUP_DIR}."
    echo "The experiment has NOT been launched. Review and run the resume command explicitly."
else
    echo "NO-GO -- see failures above. Do not resume until resolved."
    exit 1
fi
