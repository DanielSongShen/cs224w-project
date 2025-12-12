#!/bin/bash
# Usage: ./json_array_to_jsonl.sh input.json output.jsonl

input="$1"
output="$2"

if [ -z "$input" ] || [ -z "$output" ]; then
  echo "Usage: $0 input.json output.jsonl"
  exit 1
fi

jq -c '.[]' "$input" > "$output"
