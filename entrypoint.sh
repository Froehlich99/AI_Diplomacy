#!/bin/bash
set -euo pipefail

: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG env var required}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME env var required}"
: "${B2_BUCKET:?B2_BUCKET env var required}"

echo "=== AI Diplomacy K8s Runner ==="
echo "Config:     $EXPERIMENT_CONFIG"
echo "Experiment: $EXPERIMENT_NAME"
echo "B2 Bucket:  $B2_BUCKET"

# Authorize B2 (reads B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY from env)
b2 account authorize

# Download prior game memories if specified
MEMORY_ARG=""
if [ -n "${PRIOR_GAME_B2_PATH:-}" ]; then
    echo "Downloading prior memories from b2://$B2_BUCKET/$PRIOR_GAME_B2_PATH ..."
    mkdir -p /app/prior_memories
    b2 sync "b2://$B2_BUCKET/$PRIOR_GAME_B2_PATH" /app/prior_memories/
    MEMORY_ARG="--initial_memory_dir /app/prior_memories"
    echo "Prior memories downloaded:"
    ls -la /app/prior_memories/
fi

OUTPUT_DIR="/app/results/$EXPERIMENT_NAME"

# Run experiment
echo "Starting experiment..."
python experiment_runner.py "$EXPERIMENT_CONFIG" --output_dir "$OUTPUT_DIR" $MEMORY_ARG

# Upload results to B2
echo "Uploading results to b2://$B2_BUCKET/$EXPERIMENT_NAME ..."
b2 sync "$OUTPUT_DIR" "b2://$B2_BUCKET/$EXPERIMENT_NAME"

echo "=== Done ==="
