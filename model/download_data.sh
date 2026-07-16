#!/usr/bin/env bash
#
# Download the KDD Cup 1998 learning set into data/.
#
# Dataset: KDD Cup 1998 direct-mail dataset, UCI ML Repository, CC BY 4.0.
# Citation: Parsa, I. (1998). KDD Cup 1998 [Data set]. https://doi.org/10.24432/C5401H
#
# The file is fetched from a GitHub mirror rather than UCI. The mirror was verified
# byte-identical to the UCI copy on 2026-07-16.
#
# Idempotent: if data/cup98lrn.txt already exists and is the right size, this exits 0
# without re-downloading. Any size mismatch is a hard failure, since a truncated or
# substituted file would silently invalidate every downstream number.
#
# Usage: bash model/download_data.sh

set -euo pipefail

readonly MIRROR_URL="https://raw.githubusercontent.com/facebookresearch/metamulti/main/codes/cup98lrn.txt.gz"
readonly EXPECTED_BYTES=117167952

# Resolve paths relative to the repo root, so the script works from any directory.
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DATA_DIR="${REPO_ROOT}/data"
readonly TARGET="${DATA_DIR}/cup98lrn.txt"
readonly ARCHIVE="${TARGET}.gz"

# Portable stat: BSD (macOS) uses -f%z, GNU uses -c%s.
file_size() {
  stat -f%z "$1" 2>/dev/null || stat -c%s "$1"
}

fail_on_size_mismatch() {
  local path="$1" actual
  actual="$(file_size "${path}")"
  if [[ "${actual}" -ne "${EXPECTED_BYTES}" ]]; then
    echo "ERROR: ${path} is ${actual} bytes, expected ${EXPECTED_BYTES}." >&2
    echo "The download is truncated or the mirror has changed. Not usable." >&2
    echo "Delete the file and re-run to retry." >&2
    exit 1
  fi
}

main() {
  if [[ -f "${TARGET}" ]]; then
    echo "${TARGET} already exists. Verifying size."
    fail_on_size_mismatch "${TARGET}"
    echo "OK: ${EXPECTED_BYTES} bytes. Nothing to do."
    exit 0
  fi

  mkdir -p "${DATA_DIR}"

  echo "Downloading learning set from the mirror."
  # Fail on HTTP errors instead of writing an error page to disk.
  curl -fsSL -o "${ARCHIVE}" "${MIRROR_URL}"

  echo "Decompressing."
  gunzip "${ARCHIVE}"

  echo "Verifying size."
  fail_on_size_mismatch "${TARGET}"

  echo "OK: ${TARGET}, ${EXPECTED_BYTES} bytes."
}

main "$@"
