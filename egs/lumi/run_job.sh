#!/bin/bash
# Usage: ./run_job.sh --debug|--train|--test --devices <num> [other args]

mode=""
nodes=""
devices=2
other_args=()
slurm_account=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --debug)
            mode="debug"
            shift
            ;;
        --train)
            mode="train"
            shift
            ;;
        --test)
            mode="test"
            shift
            ;;
        --devices)
            devices="$2"
            shift; shift
            ;;
        --nodes)
            nodes="$2"
            shift; shift
            ;;
        -A)
            slurm_account="$2"
            shift; shift
            ;;
        *)
            other_args+=("$1")
            shift
            ;;
    esac
done

if [ -z "$mode" ]; then
    echo "Error: Must specify one of --debug, --train, or --test"
    exit 1
fi

if [ -z "$slurm_account" ]; then
    echo "Error: Must specify SLURM account with -A <account>"
    exit 1
fi

# Select the correct sbatch script template
case $mode in
    debug)
        template="egs/lumi/sbatch_debug.sh"
        ;;
    train)
        template="egs/lumi/sbatch_train.sh"
        ;;
    test)
        template="egs/lumi/sbatch_test.sh"
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

# Generate a temporary sbatch script with the correct GPU and node counts
mkdir -p ./egs/lumi/tmp
TMP_SCRIPT=$(mktemp ./egs/lumi/tmp/sbatch_job_XXXXXX.sh)
nodes=${nodes:-1}
sed "s/__GPUS__/$devices/g; s/__NODES__/$nodes/g" "$template" > "$TMP_SCRIPT"

# Submit the job
sbatch -A "$slurm_account" "$TMP_SCRIPT" --devices "$devices" --nodes "$nodes" "${other_args[@]}"

# Optionally, remove the temporary script after submission
# rm "$TMP_SCRIPT"
