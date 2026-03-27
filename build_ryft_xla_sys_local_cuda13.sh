#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly CRATE_DIR="$REPO_ROOT/crates/ryft-xla-sys"
readonly LOCAL_BIN_DIR="$REPO_ROOT/.tools/bin"
readonly LOCAL_CACHE_DIR="$REPO_ROOT/.cache"
readonly BAZELISK_VERSION="1.26.0"
readonly CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb"
readonly PLUGIN_OUTPUT_NAME="pjrt-plugin-linux-arm64-cuda-13.tar.gz"
readonly PLUGIN_LIBRARY_NAME="libpjrt-plugin-cuda-13.so"
readonly SYS_OUTPUT_NAME="ryft-xla-sys-linux-arm64-cpu.tar.gz"

BUILD_CPU_ARCHIVE=0
SKIP_BASE_PACKAGES=0
SKIP_CUDA_INSTALL=0
BAZEL_BIN=""
KEEP_PLUGIN_LIBRARY=0

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

die() {
  printf '[%s] error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: ./$SCRIPT_NAME [options]

Build the local CUDA 13 ARM64 ryft-xla-sys artifact from this repo root.

By default this mirrors the linux-arm64-cuda-13 workflow job and produces:
  $PLUGIN_OUTPUT_NAME

Options:
  --also-build-cpu-archive  Also build $SYS_OUTPUT_NAME for this platform.
  --bazel PATH              Use an existing bazel/bazelisk binary at PATH.
  --skip-base-packages      Do not apt-install common build prerequisites.
  --skip-cuda-install       Do not apt-install CUDA 13 packages.
  --keep-plugin-library     Keep $CRATE_DIR/$PLUGIN_LIBRARY_NAME after packaging.
  -h, --help                Show this help text.
EOF
}

require_repo_root() {
  [[ -f "$REPO_ROOT/Cargo.toml" ]] || die "run this from the ryft repo root"
  [[ -f "$CRATE_DIR/WORKSPACE" ]] || die "missing $CRATE_DIR/WORKSPACE"
}

require_supported_host() {
  [[ "$(uname -s)" == "Linux" ]] || die "this script only supports Linux"

  case "$(uname -m)" in
    aarch64|arm64)
      ;;
    *)
      die "this script only supports ARM64 hosts"
      ;;
  esac

  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
      log "warning: expected Ubuntu 24.04, found ${PRETTY_NAME:-unknown}; continuing anyway"
    fi
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

run_with_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    require_command sudo
    sudo "$@"
  fi
}

download_to() {
  local url="$1"
  local destination="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$destination"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -qO "$destination" "$url"
    return
  fi

  die "need either curl or wget to download dependencies"
}

apt_install_if_needed() {
  local packages=()
  local package
  for package in "$@"; do
    if ! dpkg -s "$package" >/dev/null 2>&1; then
      packages+=("$package")
    fi
  done

  if [[ "${#packages[@]}" -eq 0 ]]; then
    return
  fi

  log "installing apt packages: ${packages[*]}"
  run_with_sudo apt-get update -qq
  run_with_sudo apt-get install -y --no-install-recommends "${packages[@]}"
}

ensure_base_packages() {
  if [[ "$SKIP_BASE_PACKAGES" -eq 1 ]]; then
    return
  fi

  require_command dpkg
  require_command apt-get
  apt_install_if_needed \
    build-essential \
    ca-certificates \
    curl \
    git \
    pkg-config \
    python3 \
    unzip \
    wget \
    zip
}

ensure_cuda_13() {
  if [[ "$SKIP_CUDA_INSTALL" -eq 1 ]]; then
    return
  fi

  require_command dpkg
  require_command apt-get
  require_command mktemp
  require_command rm

  local needs_cuda=0
  local package
  for package in cuda-13-0 cuda-toolkit-13-0 cuda-nvcc-13-0 cuda-nvrtc-dev-13-0; do
    if ! dpkg -s "$package" >/dev/null 2>&1; then
      needs_cuda=1
      break
    fi
  done

  if [[ "$needs_cuda" -eq 0 ]]; then
    log "CUDA 13 packages already installed"
  else
    local temp_dir
    temp_dir="$(mktemp -d)"
    trap 'rm -rf "$temp_dir"' RETURN

    log "installing CUDA 13 packages for Ubuntu 24.04 ARM64"
    download_to "$CUDA_KEYRING_URL" "$temp_dir/cuda-keyring_1.1-1_all.deb"
    run_with_sudo dpkg -i "$temp_dir/cuda-keyring_1.1-1_all.deb"
    run_with_sudo apt-get update -qq
    run_with_sudo apt-get install -y --no-install-recommends \
      cuda-13-0 \
      cuda-toolkit-13-0 \
      cuda-nvcc-13-0 \
      cuda-nvrtc-dev-13-0
    trap - RETURN
    rm -rf "$temp_dir"
  fi

  if [[ -d /usr/local/cuda-13.0/bin ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
    export PATH="$CUDA_HOME/bin:$PATH"
    if [[ -d "$CUDA_HOME/lib64" ]]; then
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi
}

resolve_bazel() {
  if [[ -n "$BAZEL_BIN" ]]; then
    [[ -x "$BAZEL_BIN" ]] || die "--bazel path is not executable: $BAZEL_BIN"
    printf '%s\n' "$BAZEL_BIN"
    return
  fi

  if command -v bazel >/dev/null 2>&1; then
    command -v bazel
    return
  fi

  if command -v bazelisk >/dev/null 2>&1; then
    command -v bazelisk
    return
  fi

  mkdir -p "$LOCAL_BIN_DIR"
  local local_bazel="$LOCAL_BIN_DIR/bazel"
  if [[ ! -x "$local_bazel" ]]; then
    local url="https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-arm64"
    log "installing repo-local bazelisk v${BAZELISK_VERSION} at $local_bazel"
    download_to "$url" "$local_bazel"
    chmod +x "$local_bazel"
  fi
  printf '%s\n' "$local_bazel"
}

sha256_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
    return
  fi

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return
  fi

  die "need sha256sum or shasum to print checksums"
}

extract_xla_commit() {
  grep 'XLA_COMMIT = "' "$CRATE_DIR/WORKSPACE" | sed 's/.*XLA_COMMIT = "\([^"]*\)".*/\1/'
}

build_plugin_archive() {
  local bazel="$1"
  local plugin_archive_path="$REPO_ROOT/$PLUGIN_OUTPUT_NAME"
  local plugin_library_path="$CRATE_DIR/$PLUGIN_LIBRARY_NAME"

  log "building //:pjrt-gpu-plugin with --config=linux_arm64 --config=cuda-13"
  (
    cd "$CRATE_DIR"
    "$bazel" build --config=linux_arm64 --config=cuda-13 //:pjrt-gpu-plugin
    cp bazel-bin/libpjrt-gpu-plugin.so "$plugin_library_path"
  )

  rm -f "$plugin_archive_path"
  tar -C "$CRATE_DIR" -czf "$plugin_archive_path" "$PLUGIN_LIBRARY_NAME"

  if [[ "$KEEP_PLUGIN_LIBRARY" -eq 0 ]]; then
    rm -f "$plugin_library_path"
  fi

  log "wrote $plugin_archive_path"
  log "sha256 $(sha256_file "$plugin_archive_path")"
}

build_cpu_archive() {
  local bazel="$1"
  local sys_archive_path="$REPO_ROOT/$SYS_OUTPUT_NAME"

  log "building //:ryft-xla-sys-archive with --config=linux_arm64"
  (
    cd "$CRATE_DIR"
    "$bazel" build --config=linux_arm64 //:ryft-xla-sys-archive
    cp bazel-bin/ryft-xla-sys-archive.tar.gz "$sys_archive_path"
  )

  log "wrote $sys_archive_path"
  log "sha256 $(sha256_file "$sys_archive_path")"
}

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --also-build-cpu-archive)
        BUILD_CPU_ARCHIVE=1
        ;;
      --bazel)
        shift
        [[ "$#" -gt 0 ]] || die "--bazel requires a path"
        BAZEL_BIN="$1"
        ;;
      --skip-base-packages)
        SKIP_BASE_PACKAGES=1
        ;;
      --skip-cuda-install)
        SKIP_CUDA_INSTALL=1
        ;;
      --keep-plugin-library)
        KEEP_PLUGIN_LIBRARY=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
    shift
  done
}

main() {
  parse_args "$@"
  require_repo_root
  require_supported_host
  ensure_base_packages
  ensure_cuda_13

  mkdir -p "$LOCAL_CACHE_DIR"
  export BAZELISK_HOME="${BAZELISK_HOME:-$LOCAL_CACHE_DIR/bazelisk}"
  export USE_BAZEL_VERSION="${USE_BAZEL_VERSION:-$(< "$CRATE_DIR/.bazelversion")}"

  local bazel
  bazel="$(resolve_bazel)"
  log "using bazel command: $bazel"
  log "using Bazel version: $USE_BAZEL_VERSION"
  log "using XLA commit: $(extract_xla_commit)"

  build_plugin_archive "$bazel"
  if [[ "$BUILD_CPU_ARCHIVE" -eq 1 ]]; then
    build_cpu_archive "$bazel"
  fi

  log "done"
}

main "$@"
