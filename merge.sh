#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-.}"          # folder to scan (default: current dir)
OUT="${2:-merged.mp4}" # output file

cd "$DIR"

# Build the concat list (sorted by name)
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT

for f in *.mp4; do
    [[ -e "$f" ]] || { echo "No .mp4 files found."; exit 1; }
    printf "file '%s'\n" "$PWD/${f//\'/\'\\\'\'}" >> "$LIST"
done

ffmpeg -f concat -safe 0 -i "$LIST" -c copy "$OUT"
echo "Done: $OUT"

