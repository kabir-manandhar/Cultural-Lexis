#!/bin/bash

source env.sh

# Check for minimum number of required arguments
if [ $# -lt 4 ]; then
	    echo "Usage: $0 docker_image head_node_address --head|--worker path_to_hf_home [additional_args...]"
	        exit 1
fi

# Assign the first three arguments and shift them away
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker
PATH_TO_HF_HOME="$4"
shift 4

# Additional arguments are passed directly to the Docker command
ADDITIONAL_ARGS=("$@")

# Validate node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
	    echo "Error: Node type must be --head or --worker"
	        exit 1
fi

# Define a function to cleanup on EXIT signal
# cleanup() {
# 	    docker stop node
# 	    docker rm node
# 	}
# trap cleanup EXIT

# Command setup for head or worker node
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
	    RAY_START_CMD+=" --head --port=9174"
    else
	        RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:9174"
fi

# additional args is -e VLLM_HOST_IP=VALUE
# convert it to export VLLM_HOST_IP=VALUE

ADDITIONAL_CONTENT=""
for i in "${ADDITIONAL_ARGS[@]}"
do
	# remove -e from the string
	ARG=$(echo $i | sed 's/-e //')
	# check if empty string
	if [ -z "$ARG" ]; then
		continue
	fi
	# add export to the string
	ARG=$(echo "export $ARG")
	ADDITIONAL_CONTENT+="${ARG};"
done

echo "Running additional content: ${ADDITIONAL_CONTENT}"

eval $ADDITIONAL_CONTENT

echo "Checking if the env is set by running: echo \$VLLM_HOST_IP"
echo $VLLM_HOST_IP


# Run the command with the user specified parameters and additional arguments
echo "Running command: ${RAY_START_CMD}"

sleep 5
# not in docker, just run 
eval ${RAY_START_CMD}

