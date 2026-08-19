#!/usr/bin/env bash
# Stage a Sentinel-1 SAFE once, then run the CSLC-S1 SAS once per IW subswath,
# each restricted to that subswath's bursts and given its own azimuth-time
# offset -- e.g. to reproduce az_offset_demo's per-subswath calibration
# (iw1=-74.53e-6 iw2=-73.76e-6 iw3=-74.09e-6) without depending on
# compass_batch.
#
# run_cslc.sh's --az-time-offset applies one scalar to the whole SAFE, because
# stage_cslc_inputs.py writes a single runconfig with no burst_id filter (all
# subswaths together). This script wraps run_cslc.sh: it stages inputs and the
# base runconfig exactly as run_cslc.sh does (--skip-sas), then for each
# ivN=VAL given, enumerates that subswath's burst ids inside the Docker image
# (s1reader), derives a runconfig restricted to just those burst ids with
# azimuth_time_offset patched to VAL, and runs the SAS on it. All subswath
# runs share the same product_path/scratch_path -- COMPASS lays products out
# per burst_id/date, so they cannot collide -- and run sequentially so Docker
# resource limits don't have to be divided between them.
#
# Usage:
#   run_cslc_per_subswath.sh <GRANULE> --az-time-offset ivN=VAL [ivN=VAL ...] \
#       [options] [-- <extra stage_cslc_inputs.py args>]
#
# Options (same meaning as run_cslc.sh):
#   --workdir DIR     Run directory (default: ./<GRANULE>).
#   --image IMAGE     Docker image (default: opera/cslc_s1:final_0.5.7).
#   --python PY       Python for staging (default: python).
#   --pol POL         Polarization for burst enumeration (default: vv).
#   --skip-staging    Reuse inputs already in <workdir>/input_data.
#   --no-user         Run the container as its default user (do not map host uid).
#   --docker-arg ARG  Extra `docker run` arg (repeatable).
#   -h, --help        This help.
#
# Example:
#   run_cslc_per_subswath.sh <GRANULE> --workdir DIR --skip-staging \
#       --az-time-offset iw1=-74.53e-6 iw2=-73.76e-6 iw3=-74.09e-6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_CSLC="$SCRIPT_DIR/run_cslc.sh"

IMAGE="opera/cslc_s1:final_0.5.7"
PYTHON="${PYTHON:-python}"
WORKDIR=""
POL="vv"
RUN_STAGING=1
MAP_USER=1
declare -a AZ_OFFSETS=()
declare -a STAGE_ARGS=()
declare -a DOCKER_ARGS=()
GRANULE=""

usage() { sed -n '2,32p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workdir)        WORKDIR="$2"; shift 2;;
        --image)          IMAGE="$2"; shift 2;;
        --python)         PYTHON="$2"; shift 2;;
        --pol)            POL="$2"; shift 2;;
        --az-time-offset)
            shift
            while [[ $# -gt 0 && "$1" != --* && "$1" != -- ]]; do
                AZ_OFFSETS+=("$1"); shift
            done
            ;;
        --skip-staging)   RUN_STAGING=0; shift;;
        --no-user)        MAP_USER=0; shift;;
        --docker-arg)     DOCKER_ARGS+=("$2"); shift 2;;
        --)               shift; STAGE_ARGS+=("$@"); break;;
        -h|--help)        usage; exit 0;;
        -*)               echo "Unknown option: $1" >&2; usage; exit 1;;
        *)                GRANULE="$1"; shift;;
    esac
done

[[ -n "$GRANULE" ]] || { echo "ERROR: no granule given" >&2; usage; exit 1; }
[[ ${#AZ_OFFSETS[@]} -gt 0 ]] || { echo "ERROR: no --az-time-offset ivN=VAL given" >&2; usage; exit 1; }

GRANULE="${GRANULE%.zip}"; GRANULE="${GRANULE%.SAFE}"
: "${WORKDIR:=$PWD/$GRANULE}"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
RUNCONFIG="$WORKDIR/runconfig_cslc_s1.yaml"

# --- Step 1: stage inputs + base runconfig via run_cslc.sh (no SAS run) ---
run_cslc_args=(--workdir "$WORKDIR" --image "$IMAGE" --python "$PYTHON" --skip-sas)
[[ "$RUN_STAGING" -eq 0 ]] && run_cslc_args+=(--skip-staging)
[[ "$MAP_USER" -eq 0 ]] && run_cslc_args+=(--no-user)
for a in "${DOCKER_ARGS[@]}"; do run_cslc_args+=(--docker-arg "$a"); done
"$RUN_CSLC" "$GRANULE" "${run_cslc_args[@]}" -- "${STAGE_ARGS[@]}"

[[ -f "$RUNCONFIG" ]] || { echo "ERROR: runconfig not found: $RUNCONFIG" >&2; exit 1; }

SAFE_PATH=$(sed -n '/safe_file_path:/{n;s/^ *- *//p}' "$RUNCONFIG")
ORBIT_PATH=$(sed -n '/orbit_file_path:/{n;s/^ *- *//p}' "$RUNCONFIG")
[[ -n "$SAFE_PATH" && -n "$ORBIT_PATH" ]] || {
    echo "ERROR: could not read safe/orbit path from $RUNCONFIG" >&2; exit 1;
}

docker_run() {
    local run_args=(--rm -v "$WORKDIR":"$WORKDIR" -w "$WORKDIR")
    if [[ "$MAP_USER" -eq 1 ]]; then
        run_args+=(-u "$(id -u):$(id -g)" -e HOME=/tmp)
    fi
    run_args+=("${DOCKER_ARGS[@]}" "$IMAGE" "$@")
    docker run "${run_args[@]}"
}

# --- Step 2: one restricted runconfig + SAS run per requested subswath ---
for tok in "${AZ_OFFSETS[@]}"; do
    iw_name="${tok%%=*}"; val="${tok#*=}"
    iw_num="${iw_name#iw}"
    [[ "$iw_num" =~ ^[1-3]$ ]] || { echo "ERROR: bad subswath '$iw_name' (expected iw1/iw2/iw3)" >&2; exit 1; }

    echo ">>> [$iw_name] Enumerating bursts (pol=$POL)"
    mapfile -t burst_ids < <(docker_run python3 -c "
import sys
from s1reader.s1_reader import load_bursts
safe, orbit, iw, pol = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ids = sorted({str(b.burst_id) for b in load_bursts(safe, orbit, iw, pol=pol)})
print('\n'.join(ids))
" "$SAFE_PATH" "$ORBIT_PATH" "$iw_num" "$POL")

    if [[ ${#burst_ids[@]} -eq 0 ]]; then
        echo "    no $iw_name bursts in this SAFE, skipping"
        continue
    fi
    echo "    ${#burst_ids[@]} burst(s): ${burst_ids[*]}"

    sub_runconfig="$WORKDIR/runconfig_cslc_s1_${iw_name}.yaml"
    "$PYTHON" - "$RUNCONFIG" "$sub_runconfig" "$val" "${burst_ids[@]}" <<'EOF'
import re
import sys

src, dst, val, burst_ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
text = open(src).read()

# Restrict to this subswath's bursts (the base runconfig's burst_id: key is
# empty, meaning "all bursts").
items = "\n".join(f"          - {b}" for b in burst_ids)
text = text.replace("          burst_id:\n", f"          burst_id:\n{items}\n", 1)

# Same idempotent correction_luts patch as run_cslc.sh's --az-time-offset.
text = re.sub(r"\n {10}correction_luts:\n(?: {12,}.*\n)*", "\n", text)
block = (
    "          correction_luts:\n"
    "              enabled: True\n"
    f"              azimuth_time_offset: {val}\n"
)
text = text.replace("      worker:", block + "      worker:", 1)

open(dst, "w").write(text)
EOF

    echo ">>> [$iw_name] Running CSLC-S1 SAS: azimuth_time_offset=$val"
    docker_run s1_cslc.py "$sub_runconfig"
done

echo ">>> Done. Products under $WORKDIR/output_s1_cslc"
