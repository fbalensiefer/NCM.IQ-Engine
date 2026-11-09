#!/usr/bin/env bash
# Extract and verify the official native LibTorch SDK.

set -euo pipefail
version=2.13.0
output=${1:-.libtorch}
mkdir -p "$output"
output=$(cd "$output" && pwd)
if [[ -f "$output/include/torch/csrc/api/include/torch/torch.h" && -d "$output/lib" ]]; then
  printf '%s\n' "$output"
  exit 0
fi
temp=$(mktemp -d)
trap 'rm -rf "$temp"' EXIT
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)
    metadata=$(curl -fsSL "https://pypi.org/pypi/torch/$version/json" |
      jq -er '.urls[] | select(.filename | test("cp311-[^-]+-macosx_.*_arm64\\.whl$")) | [.url, .digests.sha256] | @tsv')
    read -r url hash <<< "$metadata"
    ;;
  Linux/x86_64)
    url="https://download-r2.pytorch.org/whl/cpu/torch-$version%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"
    hash=6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b
    ;;
  Linux/aarch64)
    url="https://download-r2.pytorch.org/whl/cpu/torch-$version%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl"
    hash=84453b69508ec79902f899c5ed9495acb9e2bbe9fda5f1d5d6f19e3c3842e1a7
    ;;
  *) echo "Supply LibTorch $version for your platform via LIBTORCH" >&2; exit 1 ;;
esac
curl -fLsS --retry 2 "$url" -o "$temp/torch.zip"
printf '%s  %s\n' "$hash" "$temp/torch.zip" | shasum -a 256 -c - >&2
unzip -q "$temp/torch.zip" 'torch/include/*' 'torch/lib/*' -d "$temp"
mv "$temp/torch/include" "$temp/torch/lib" "$output/"
printf '%s\n' "$output"
