#!/usr/bin/env bash
# Stage all CSLC-S1 inputs for a Sentinel-1 granule, then run the CSLC-S1 SAS
# (s1_cslc.py) inside the COMPASS Docker image on the generated runconfig.
#
# Two steps, either of which can be skipped:
#   1. stage_cslc_inputs.py all <granule> -i <workdir>/input_data   (host, needs
#      Earthdata ~/.netrc; see scripts/README.md)
#   2. docker run ... s1_cslc.py <workdir>/runconfig_cslc_s1.yaml
#
# The workdir is bind-mounted into the container at the SAME path and used as the
# working directory, so the relative paths the stager writes into the runconfig
# (input_data/..., output_s1_cslc) resolve unchanged inside the container.
#
# Usage:
#   run_cslc.sh <GRANULE> [options] [-- <extra stage_cslc_inputs.py args>]
#
# Options:
#   --workdir DIR         Run directory (default: ./<GRANULE>).
#   --image IMAGE         Docker image (default: opera/cslc_s1:final_0.5.7).
#   --python PY           Python for staging (default: python).
#   --az-time-offset SEC  Constant azimuth-time re-registration offset (seconds),
#                         written into the runconfig's correction_luts block.
#                         Zero/omitted leaves geocoding unchanged.
#   --skip-staging        Reuse inputs already in <workdir>/input_data.
#   --skip-sas            Only stage; do not run the SAS.
#   --no-user             Run the container as its default user (do not map host uid).
#   --docker-arg ARG      Extra `docker run` arg (repeatable).
#   -h, --help            This help.
#
# Everything after `--` is forwarded to stage_cslc_inputs.py, e.g. to match a
# golden delivery:
#   run_cslc.sh <GRANULE> -- --product-type FINAL --sol-code jpl --margin 1.7
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGER="$SCRIPT_DIR/stage_cslc_inputs.py"

IMAGE="opera/cslc_s1:final_0.5.7"
PYTHON="${PYTHON:-python}"
WORKDIR=""
RUN_STAGING=1
RUN_SAS=1
MAP_USER=1
AZ_OFFSET=""
declare -a STAGE_ARGS=()
declare -a DOCKER_ARGS=()
GRANULE=""

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workdir)    WORKDIR="$2"; shift 2;;
        --image)      IMAGE="$2"; shift 2;;
        --python)     PYTHON="$2"; shift 2;;
        --az-time-offset) AZ_OFFSET="$2"; shift 2;;
        --skip-staging) RUN_STAGING=0; shift;;
        --skip-sas)   RUN_SAS=0; shift;;
        --no-user)    MAP_USER=0; shift;;
        --docker-arg) DOCKER_ARGS+=("$2"); shift 2;;
        --)           shift; STAGE_ARGS+=("$@"); break;;
        -h|--help)    usage; exit 0;;
        -*)           echo "Unknown option: $1" >&2; usage; exit 1;;
        *)            GRANULE="$1"; shift;;
    esac
done

[[ -n "$GRANULE" ]] || { echo "ERROR: no granule given" >&2; usage; exit 1; }
GRANULE="${GRANULE%.zip}"; GRANULE="${GRANULE%.SAFE}"
: "${WORKDIR:=$PWD/$GRANULE}"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
INPUT_DIR="$WORKDIR/input_data"
RUNCONFIG="$WORKDIR/runconfig_cslc_s1.yaml"

if [[ "$RUN_STAGING" -eq 1 ]]; then
    echo ">>> [1/2] Staging inputs for $GRANULE -> $INPUT_DIR"
    "$PYTHON" "$STAGER" all "$GRANULE" -i "$INPUT_DIR" "${STAGE_ARGS[@]}"
else
    echo ">>> [1/2] Skipping staging (reusing $INPUT_DIR)"
fi

if [[ "$RUN_SAS" -eq 0 ]]; then
    echo ">>> [2/2] Skipping SAS (--skip-sas). Runconfig: $RUNCONFIG"
    exit 0
fi

[[ -f "$RUNCONFIG" ]] || { echo "ERROR: runconfig not found: $RUNCONFIG" >&2; exit 1; }

if [[ -n "$AZ_OFFSET" ]]; then
    echo ">>> Setting correction_luts.azimuth_time_offset = $AZ_OFFSET in $RUNCONFIG"
    "$PYTHON" - "$RUNCONFIG" "$AZ_OFFSET" <<'EOF'
# Replace any existing correction_luts block (idempotent under --skip-staging
# reruns) and insert a fresh one before the worker: group.
import re
import sys

path, val = sys.argv[1], sys.argv[2]
text = open(path).read()
text = re.sub(r"\n {10}correction_luts:\n(?: {12,}.*\n)*", "\n", text)
block = (
    "          correction_luts:\n"
    "              enabled: True\n"
    f"              azimuth_time_offset: {val}\n"
)
text = text.replace("      worker:", block + "      worker:", 1)
open(path, "w").write(text)
EOF
fi

run_args=(--rm -v "$WORKDIR":"$WORKDIR" -w "$WORKDIR")
if [[ "$MAP_USER" -eq 1 ]]; then
    # Map the host user so products are host-owned; HOME must be writable for the
    # numba/isce3 caches when not running as the image's built-in user.
    run_args+=(-u "$(id -u):$(id -g)" -e HOME=/tmp)
fi
run_args+=("${DOCKER_ARGS[@]}" "$IMAGE" s1_cslc.py "$RUNCONFIG")

echo ">>> [2/2] Running CSLC-S1 SAS in Docker: $IMAGE"
echo "    docker run ${run_args[*]}"
docker run "${run_args[@]}"

echo ">>> Done. Products under $WORKDIR/output_s1_cslc"
