#!/bin/bash -l
#SBATCH --job-name=saga-debug
#SBATCH --output=sbatch_logs/%J.log
#SBATCH --error=sbatch_logs/%J.log
#SBATCH --partition=dev-g
#SBATCH --nodes=__NODES__
#SBATCH --ntasks-per-node=__GPUS__
#SBATCH --gpus-per-node=__GPUS__
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60000M
#SBATCH --time=0-00:30:00

set -e

# Parse arguments
yaml_file=""
ckpt_path=""
compile_disabled=""
devices=2
while [[ $# -gt 0 ]]; do
	key="$1"
	case $key in
		-c)
			yaml_file="$2"
			shift; shift
			;;
		--devices)
			devices="$2"
			shift; shift
			;;
        --nodes)
            nodes="$2"
            shift; shift
            ;;
		--compile_disabled)
			compile_disabled="--compile_disabled"
			shift
			;;
		*)
			echo "Unknown argument: $key"
			exit 1
			;;
	esac
done

if [ -z "$yaml_file" ]; then
	echo "Usage: $0 -c <yaml_file> [--ckpt_path <ckpt_path> --compile_disabled]"
	exit 1
fi

# Path to the environment. Same as INSTALL_DIR in create_environment.sh
ENV_DIR=./lumi_env/image

# Name of the image to to use
IMAGE_NAME=saganet.sif

# Load the required modules
module purge
module load LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings
module load git

source ~/.bashrc
export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

# Tell RCCL to use only Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

# Run the training script:
# We use the Singularity container from 'create_environment.sh'
# with the --bind option to mount the virtual environment in $ENV_DIR/myenv.sqsh
# into the container at /user-software.
#

echo "Using $devices GPUs"
echo "Using ${nodes:-1} nodes"

# The number of GPUs and nodes are auto-detected from the SLURM environment variables.
cmd="srun singularity exec \
	-B $ENV_DIR/myenv.sqsh:/user-software:image-src=/ $ENV_DIR/$IMAGE_NAME \
	torchrun --standalone --nproc_per_node=$devices train.py \
	--config-name $yaml_file \
	debug=True \
	batch_size=2 \
	num_iterations=2 \
	eval_interval=20 \
	val_interval=20 \
	save_checkpoint_interval=1 \
	save_weights_interval=1 \
	save_eval_interval=20 \
	log_extra_interval=20 \
"

if [ -n "$compile_disabled" ]; then
	cmd+=" compile=False "
fi

# Read and export secret WANDB API key as environment variable
if [ -f .wandb_api_key ]; then
	export WANDB_API_KEY=$(cat .wandb_api_key)
fi

echo "$cmd"
eval $cmd

# Bonus:
# With full-node allocations, i.e. running jobs on standard-g or small-g with slurm argument `--exclusive,
# it is beneficial to set the CPU bindings (see https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/#gpu-binding)

# # Define CPU binding for optimal performance in full-node allocations
# CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
# CPU_BIND="${CPU_BIND},fe0000,fe000000"
# CPU_BIND="${CPU_BIND},fe,fe00"
# CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

# srun --cpu-bind=$CPU_BIND singularity exec \
#    -B $ENV_DIR/myenv.sqsh:/user-software:image-src=/ $ENV_DIR/$IMAGE_NAME \
#     python -m pytorch_example.run \
#         --num_gpus=$SLURM_GPUS_ON_NODE \
#         --num_nodes=$SLURM_JOB_NUM_NODES \
