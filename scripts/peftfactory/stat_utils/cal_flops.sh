#!/bin/bash

CONFIG_DIR="scripts/peftfactory/stat_utils/configs"
SCRIPT="scripts/peftfactory/stat_utils/cal_flops.py"
RESULTS=()

for config in "$CONFIG_DIR"/*.yaml; do
    method=$(basename "$config" .yaml)
    echo "Running $SCRIPT with method: $method"
    python "$SCRIPT" "$config"
    result_file="scripts/peftfactory/stat_utils/flops_${method}.json"
    if [[ -f "$result_file" ]]; then
        number=$(jq '.number' "$result_file")
        echo "FLOPs for $method: $number"
        RESULTS+=("$method: $number")
    fi
done

echo "Summary of all FLOPs numbers:"
for entry in "${RESULTS[@]}"; do
    echo "$entry"
done