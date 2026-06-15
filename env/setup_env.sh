#!/usr/bin/env bash
#
# setup_env.sh — create (or update) and activate the SBP Studio Conda environment.
#
# Linux / macOS. Run from anywhere; paths resolve relative to this script.
#
#   Create/update only:     bash env/setup_env.sh
#   Create/update + keep it activated in YOUR shell:
#                           source env/setup_env.sh      # (or: . env/setup_env.sh)
#
# `conda activate` only changes the shell it runs IN, so to stay in the env after
# the script finishes you must `source` it (a plain `bash …` runs in a subshell).
# When not sourced, the script prints the exact `conda activate` command to run.
# ---------------------------------------------------------------------------------
set -uo pipefail

# Resolve this script's directory (so environment.yml is found from any CWD).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
ENV_NAME="$(awk -F': *' '/^name:/{print $2; exit}' "${ENV_FILE}" 2>/dev/null)"
ENV_NAME="${ENV_NAME:-sbp_studio}"

# Detect whether we were sourced (so we can actually activate the caller's shell).
_SOURCED=0
[ "${BASH_SOURCE[0]:-$0}" != "${0}" ] && _SOURCED=1

echo ">> SBP Studio environment setup"
echo "   env file : ${ENV_FILE}"
echo "   env name : ${ENV_NAME}"

# 1) conda must be available.
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' was not found on PATH. Install Miniconda/Anaconda first:" >&2
    echo "       https://docs.conda.io/en/latest/miniconda.html" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    return 1 2>/dev/null || exit 1
fi

# 2) Make `conda activate` usable inside this non-interactive shell.
CONDA_BASE="$(conda info --base 2>/dev/null)"
# shellcheck disable=SC1091
[ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ] && \
    source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 3) Create the env, or update it in place if it already exists (--prune removes
#    dependencies that are no longer listed, keeping the env in sync with the file).
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo ">> Environment '${ENV_NAME}' exists — updating from ${ENV_FILE} …"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    echo ">> Creating environment '${ENV_NAME}' from ${ENV_FILE} …"
    conda env create -f "${ENV_FILE}"
fi
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
    echo "ERROR: conda env create/update failed (exit ${STATUS})." >&2
    return "${STATUS}" 2>/dev/null || exit "${STATUS}"
fi

# 4) Activate.
echo ">> Activating '${ENV_NAME}' …"
conda activate "${ENV_NAME}" 2>/dev/null

echo
echo ">> Done. Verify hardware acceleration with:"
echo "     python -m sbp_studio.cli.main accel"
echo ">> Launch the GUI with:"
echo "     python -m sbp_studio.gui"
if [ "${_SOURCED}" -eq 0 ]; then
    echo
    echo "NOTE: this ran in a subshell, so your current shell is NOT in the env."
    echo "      Activate it with:   conda activate ${ENV_NAME}"
    echo "      (or re-run sourced:  source env/setup_env.sh)"
fi
