#!/usr/bin/env bash
set -euo pipefail

DEFAULT_INSTALL_ROOT="/project/bioinformatics/Danuser_lab/Dean/dean/matlab_runtime"
PETAKIT5D_URL_DEFAULT="https://github.com/abcucberkeley/PetaKit5D/archive/refs/heads/main.zip"
MATLAB_RUNTIME_URL_DEFAULT="https://ssd.mathworks.com/supportfiles/downloads/R2023a/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2023a_Update_6_glnxa64.zip"
MATLAB_RUNTIME_VERSION_DEFAULT="R2023a"

usage() {
    cat <<'EOF'
Install PetaKit5D MCC assets and MATLAB Runtime for ClearEx deconvolution.

Usage:
  scripts/install_petakit_runtime.sh [--force] [INSTALL_ROOT]

Defaults:
  INSTALL_ROOT=/project/bioinformatics/Danuser_lab/Dean/dean/matlab_runtime

Environment overrides:
  CLEAREX_PETAKIT_RUNTIME_ROOT  Default INSTALL_ROOT when no argument is given.
  PETAKIT5D_URL                 PetaKit5D zip URL.
  MATLAB_RUNTIME_URL            MATLAB Runtime installer zip URL.
  MATLAB_RUNTIME_VERSION        MATLAB Runtime version directory, default R2023a.

After installation, source:
  INSTALL_ROOT/clearex_petakit_env.sh
EOF
}

force=0
install_root="${CLEAREX_PETAKIT_RUNTIME_ROOT:-$DEFAULT_INSTALL_ROOT}"

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --force)
            force=1
            shift
            ;;
        *)
            install_root="$1"
            shift
            ;;
    esac
done

petakit5d_url="${PETAKIT5D_URL:-$PETAKIT5D_URL_DEFAULT}"
matlab_runtime_url="${MATLAB_RUNTIME_URL:-$MATLAB_RUNTIME_URL_DEFAULT}"
matlab_runtime_version="${MATLAB_RUNTIME_VERSION:-$MATLAB_RUNTIME_VERSION_DEFAULT}"

install_root="$(realpath -m "$install_root")"
petakit_root="$install_root/PetaKit5D"
matlab_runtime_parent="$install_root/MATLAB_Runtime"
matlab_runtime_root="$matlab_runtime_parent/$matlab_runtime_version"
env_file="$install_root/clearex_petakit_env.sh"

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    elif command -v python >/dev/null 2>&1; then
        python_bin="python"
    else
        echo "ERROR: python3 or python is required for downloading runtime archives." >&2
        exit 1
    fi
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: unzip is required to extract runtime archives." >&2
    exit 1
fi

mkdir -p "$install_root"

download_file() {
    local url="$1"
    local output="$2"
    if [[ -s "$output" ]]; then
        echo "Using existing download: $output"
        return
    fi
    echo "Downloading $url"
    "$python_bin" - "$url" "$output" <<'PY'
import sys
import urllib.request

url, output = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, output)
PY
}

install_petakit5d() {
    local zip_path="$install_root/PetaKit5D.zip"
    local extracted="$install_root/PetaKit5D-main"

    if [[ "$force" -eq 0 && -x "$petakit_root/mcc/linux/run_mccMaster.sh" ]]; then
        echo "PetaKit5D MCC assets already installed at $petakit_root"
        return
    fi

    rm -rf "$petakit_root" "$extracted"
    download_file "$petakit5d_url" "$zip_path"
    unzip -o -q "$zip_path" -d "$install_root"
    mv "$extracted" "$petakit_root"
    rm -f "$zip_path"

    chmod +x "$petakit_root/mcc/linux/run_mccMaster.sh" || true
    chmod +x "$petakit_root/mcc/linux/mccMaster" || true
    chmod +x "$petakit_root/mcc/linux_with_jvm/run_mccMaster.sh" || true
    chmod +x "$petakit_root/mcc/linux_with_jvm/mccMaster" || true
}

install_matlab_runtime() {
    local zip_path="$install_root/matlabRuntime.zip"
    local tmp_dir="$install_root/matlabRuntimeTmp"
    local install_script="$tmp_dir/install"

    if [[ "$force" -eq 0 && -d "$matlab_runtime_root" && -n "$(find "$matlab_runtime_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "MATLAB Runtime already installed at $matlab_runtime_root"
        return
    fi

    rm -rf "$matlab_runtime_root" "$tmp_dir"
    mkdir -p "$matlab_runtime_parent"
    download_file "$matlab_runtime_url" "$zip_path"
    unzip -o -q "$zip_path" -d "$tmp_dir"
    if [[ ! -x "$install_script" ]]; then
        echo "ERROR: MATLAB Runtime installer not found at $install_script" >&2
        exit 1
    fi
    "$install_script" -agreeToLicense yes -destinationFolder "$matlab_runtime_parent"
    rm -f "$zip_path"
    rm -rf "$tmp_dir"
}

install_petakit5d
install_matlab_runtime

cat > "$env_file" <<EOF
export CLEAREX_PETAKIT5D_ROOT="$petakit_root"
export CLEAREX_MATLAB_RUNTIME_ROOT="$matlab_runtime_root"
EOF

if [[ ! -x "$petakit_root/mcc/linux/run_mccMaster.sh" ]]; then
    echo "ERROR: missing executable $petakit_root/mcc/linux/run_mccMaster.sh" >&2
    exit 1
fi
if [[ ! -x "$petakit_root/mcc/linux/mccMaster" ]]; then
    echo "ERROR: missing executable $petakit_root/mcc/linux/mccMaster" >&2
    exit 1
fi
if [[ ! -d "$matlab_runtime_root" || -z "$(find "$matlab_runtime_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: missing or empty MATLAB Runtime directory $matlab_runtime_root" >&2
    exit 1
fi

echo "PetaKit5D runtime installation is ready."
echo "Source this file before running ClearEx deconvolution:"
echo "  source \"$env_file\""
