#!/usr/bin/env bash
# Downloads the AevaScenes dataset using curl with resume + parallelism.
# The upstream scripts/download_dataset.sh from the SDK requires wget which
# is not installed on this machine; this is a curl-based replacement.
#
# Usage:
#   bash scripts/download_aevascenes.sh
#   PARALLEL=4 bash scripts/download_aevascenes.sh   # override concurrency
set -uo pipefail

URL_FILE="${URL_FILE:-/c/Users/djoko/Downloads/signed_urls.txt}"
OUT_DIR="${OUT_DIR:-/c/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data/aevascenes_v0.2}"
LOG="$OUT_DIR/.download.log"
PARALLEL="${PARALLEL:-3}"

if [[ ! -f "$URL_FILE" ]]; then
  echo "ERROR: URL file not found: $URL_FILE" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

download_one() {
  local url="$1"
  local path="${url%%\?*}"
  local fname="${path##*/}"
  local started done_ts sz rc
  started=$(date '+%Y-%m-%d %H:%M:%S')
  printf '%s START %s\n' "$started" "$fname" >> "$LOG"
  curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 --max-time 1800 \
       --speed-time 60 --speed-limit 100000 \
       -sS -o "$fname" "$url"
  rc=$?
  done_ts=$(date '+%Y-%m-%d %H:%M:%S')
  if [[ $rc -eq 0 ]]; then
    sz=$(stat -c%s "$fname" 2>/dev/null || stat -f%z "$fname" 2>/dev/null || echo "?")
    printf '%s OK    %s (%s bytes)\n' "$done_ts" "$fname" "$sz" >> "$LOG"
  else
    printf '%s FAIL  %s rc=%s\n' "$done_ts" "$fname" "$rc" >> "$LOG"
  fi
}
export -f download_one
export LOG

total=$(grep -c '^https' "$URL_FILE" || true)
printf '=== AevaScenes download started %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
printf 'URL file: %s\nOutput:   %s\nTotal:    %s files, parallel=%s\n' \
  "$URL_FILE" "$OUT_DIR" "$total" "$PARALLEL" >> "$LOG"

grep '^https' "$URL_FILE" | xargs -P "$PARALLEL" -I {} bash -c 'download_one "$@"' _ {}

printf '=== AevaScenes download finished %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
