#!/usr/bin/env bash
# Extracts all AevaScenes tarballs in data/aevascenes_v0.2/ in parallel.
set -uo pipefail

OUT_DIR="${OUT_DIR:-/c/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data/aevascenes_v0.2}"
LOG="$OUT_DIR/.extract.log"
PARALLEL="${PARALLEL:-3}"

cd "$OUT_DIR"

extract_one() {
  local f="$1"
  local started done_ts rc
  started=$(date '+%Y-%m-%d %H:%M:%S')
  printf '%s START %s\n' "$started" "$f" >> "$LOG"
  tar -xf "$f"
  rc=$?
  done_ts=$(date '+%Y-%m-%d %H:%M:%S')
  if [[ $rc -eq 0 ]]; then
    printf '%s OK    %s\n' "$done_ts" "$f" >> "$LOG"
  else
    printf '%s FAIL  %s rc=%s\n' "$done_ts" "$f" "$rc" >> "$LOG"
  fi
}
export -f extract_one
export LOG

total=$(ls *.tar.gz 2>/dev/null | wc -l)
printf '=== Extraction started %s (%s files, parallel=%s) ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "$total" "$PARALLEL" >> "$LOG"

ls *.tar.gz | xargs -P "$PARALLEL" -I {} bash -c 'extract_one "$@"' _ {}

printf '=== Extraction finished %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
