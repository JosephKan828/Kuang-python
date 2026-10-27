#!/usr/bin/env bash

# ====================================================
# Pipeline settings
# ====================================================

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
launch_command=("$BASH" "$script_dir/RunScript.sh" "$@")
started_at="$(TZ=Asia/Taipei date --iso-8601=seconds)"
python_bin="${PYTHON:-python}"
notification_email="${KUANG_NOTIFY_EMAIL:-}"

# These defaults can also be overridden with command-line options below.
model_args=(--model two-way --rad-scaling 1.0)
output_dir=""
run_dir=""
x_block_size=64
reconstruction_workers=4
run_simulation=true
run_analysis=true
run_reconstruction=true
run_plot=false
overwrite=false

# ====================================================
# Command-line options
# ====================================================

usage() {
    cat <<'HELP'
Usage: bash scripts/RunScript.sh [options]

Default workflow: simulation -> eigen-analysis -> reconstruction -> run record -> optional email

Workflow options:
  --skip-simulation          Skip the simulation stage
  --skip-analysis            Skip the eigen-analysis stage
  --reconstruct              Enable Fourier/Galerkin reconstruction
  --skip-reconstruction      Skip Fourier/Galerkin reconstruction
  --plot                     Generate eigen-analysis figures (default: off)
  --no-plot                  Skip plotting
  --notify-email ADDRESS     Send completion email to ADDRESS (default: $KUANG_NOTIFY_EMAIL)
  --no-email                 Disable email for this invocation
  --x-block-size N           Reconstruction spatial block size (default: 64)
  --reconstruction-workers N Reconstruction processes (default: 4; 1 for serial)
  --output-dir PATH          Parent directory; create PATH/TIMESTAMP (default: output/pipeline/)
  --run-dir PATH             Use an existing run directory directly, without a new timestamp
  --overwrite               Replace outputs of the selected numerical stages
  --python PATH             Python executable (default: $PYTHON or python)
  -h, --help                Show this help

Model options forwarded to both simulation and eigen-analysis:
  --model one-way|two-way    Default: two-way
  --rad-scaling FLOAT        Default: 1.0
  --members N               Ensemble size
  --tmax DAYS                Simulation duration
  --dt DAYS                 Time interval
  --dx METERS               Horizontal spacing
  --dz METERS               Vertical spacing
  --zonal-wavenumbers K ...  Zonal wavenumber samples
  --seed N                  Random seed

Unspecified model settings use the current Python classes. Use Bash, not sh.
Selected stages run sequentially and the pipeline stops on the first failure.
--output-dir and --run-dir are mutually exclusive.
All stages share the resolved run directory; logs are saved in its logs/TIMESTAMP/.
Successful runs save logs/TIMESTAMP/RunRecord.json and append to the parent's RunHistory.jsonl.
Email uses exported KUANG_SMTP_* settings; email failures do not invalidate results.
HELP
}

fail() {
    printf 'Pipeline error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --skip-simulation) run_simulation=false; shift ;;
        --skip-analysis) run_analysis=false; shift ;;
        --reconstruct) run_reconstruction=true; shift ;;
        --skip-reconstruction) run_reconstruction=false; shift ;;
        --plot) run_plot=true; shift ;;
        --no-plot) run_plot=false; shift ;;
        --no-email) notification_email=""; shift ;;
        --notify-email)
            require_value "$@"
            notification_email="$2"
            shift 2
            ;;
        --overwrite) overwrite=true; shift ;;
        --python)
            require_value "$@"
            python_bin="$2"
            shift 2
            ;;
        --output-dir)
            require_value "$@"
            output_dir="$2"
            shift 2
            ;;
        --run-dir)
            require_value "$@"
            run_dir="$2"
            shift 2
            ;;
        --x-block-size)
            require_value "$@"
            x_block_size="$2"
            shift 2
            ;;
        --reconstruction-workers)
            require_value "$@"
            reconstruction_workers="$2"
            shift 2
            ;;
        --model|--rad-scaling|--members|--tmax|--dt|--dx|--dz|--seed)
            require_value "$@"
            model_args+=("$1" "$2")
            shift 2
            ;;
        --zonal-wavenumbers)
            model_args+=("$1")
            shift
            [[ $# -gt 0 && "$1" != --* ]] || fail '--zonal-wavenumbers requires values.'
            while [[ $# -gt 0 && "$1" != --* ]]; do
                model_args+=("$1")
                shift
            done
            ;;
        *) fail "Unknown option: $1 (use --help)." ;;
    esac
done
export KUANG_NOTIFY_EMAIL="$notification_email"

# ====================================================
# Check the selected workflow
# ====================================================

if ! "$run_simulation" && ! "$run_analysis" && ! "$run_reconstruction" && ! "$run_plot"; then
    fail 'No stages selected.'
fi
command -v "$python_bin" >/dev/null 2>&1 || fail "Python executable not found: $python_bin"
if "$run_reconstruction" && [[ ! "$x_block_size" =~ ^[1-9][0-9]*$ ]]; then
    fail '--x-block-size must be a positive integer.'
fi
if "$run_reconstruction" && [[ ! "$reconstruction_workers" =~ ^[1-9][0-9]*$ ]]; then
    fail '--reconstruction-workers must be a positive integer.'
fi

[[ -z "$output_dir" || -z "$run_dir" ]] || fail 'Use either --output-dir or --run-dir, not both.'
timestamp="$(TZ=Asia/Taipei date +%Y-%m-%d-%H-%M-%S-%N)-$$"
if [[ -n "$run_dir" ]]; then
    [[ -d "$run_dir" ]] || fail "Run directory does not exist: $run_dir"
    output_dir="$run_dir"
else
    output_dir="${output_dir:-$project_root/output/pipeline}/$timestamp"
fi

# Check prerequisites when their producing stage is skipped.
if "$run_reconstruction" && ! "$run_simulation"; then
    for name in Metadata.h5 State.h5; do
        [[ -f "$output_dir/$name" ]] || fail "Reconstruction needs $output_dir/$name."
    done
fi
if "$run_plot" && ! "$run_analysis"; then
    [[ -f "$output_dir/EigenAnalysis.h5" ]] || fail "Plotting needs $output_dir/EigenAnalysis.h5."
fi

# Protect all selected numerical outputs before starting any stage.
outputs=()
if "$run_simulation"; then outputs+=(Metadata.h5 State.h5 Config.json); fi
if "$run_analysis"; then outputs+=(EigenAnalysis.h5); fi
if "$run_reconstruction"; then outputs+=(FourierState.h5 GalerkinState.h5); fi
if ! "$overwrite"; then
    for name in "${outputs[@]}"; do
        [[ ! -e "$output_dir/$name" ]] || fail "$output_dir/$name exists; use --overwrite to replace it."
    done
fi

mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
log_dir="$output_dir/logs/$timestamp"
mkdir -p -- "$log_dir"
common_args=("${model_args[@]}" --output-dir "$output_dir")
reconstruction_args=("$output_dir" --x-block-size "$x_block_size" --workers "$reconstruction_workers")
if "$overwrite"; then
    common_args+=(--overwrite)
    reconstruction_args+=(--overwrite)
fi

# ====================================================
# Stage execution
# ====================================================

run_stage() {
    local label="$1" script="$2"
    shift 2
    local log_file="$log_dir/$label.log"
    printf '\nStarting %s\n' "$label"
    # Log the exact invocation; arrays preserve spaces and literal arguments.
    {
        printf 'Command:'
        printf ' %q' "$python_bin" -u "$script_dir/$script" "$@"
        printf '\n'
    } | tee "$log_file"
    if "$python_bin" -u "$script_dir/$script" "$@" 2>&1 | tee -a "$log_file"; then
        printf 'Completed %s\n' "$label"
    else
        local status=$?
        printf 'Stage %s failed (exit %s). Log: %s\n' "$label" "$status" "$log_file" >&2
        return "$status"
    fi
}

printf 'Output directory: %s\nPython: %s\nLogs: %s\n' "$output_dir" "$python_bin" "$log_dir"
if "$run_simulation"; then run_stage simulation Simulation.py "${common_args[@]}"; fi
if "$run_analysis"; then run_stage eigen_analysis EigenAnalysis.py "${common_args[@]}"; fi
if "$run_reconstruction"; then run_stage reconstruction Reconstruct.py "${reconstruction_args[@]}"; fi
if "$run_plot"; then run_stage plot Plot_Eigen.py "$output_dir"; fi
run_stage run_record RunRecord.py --output-dir "$output_dir" \
    --started-at "$started_at" --log-directory "$log_dir" -- "${launch_command[@]}"
if [[ -n "$notification_email" ]]; then
    if run_stage email EmailNotification.py "$log_dir/RunRecord.json"; then
        :
    else
        printf 'Email notification failed; completed results and run record are preserved.\n' >&2
    fi
else
    printf 'Email notification disabled; set KUANG_NOTIFY_EMAIL or use --notify-email.\n'
fi
printf '\nPipeline completed. Results: %s\n' "$output_dir"
