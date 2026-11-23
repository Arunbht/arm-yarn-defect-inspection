#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${1:-$ROOT/rpi_env}"

echo "=========================================================="
echo "  Raspberry Pi 5: Smart Sequential Setup & Verification   "
echo "=========================================================="

# 0. Filesystem & Symlink Check (Protects against FAT32/exFAT USB runs)
if ! ln -s /dev/null "$ROOT/.symlink_test" 2>/dev/null; then
    echo "  [!] ERROR: The current directory does not support Linux symlinks (e.g., FAT32/exFAT USB drive)."
    echo "      Python virtual environments require a Linux filesystem (ext4)."
    echo "      Please copy this folder to your Pi home directory and run from there:"
    echo "        cp -r \"$ROOT\" ~/rpi5_deployment && cd ~/rpi5_deployment && ./setup_rpi5.sh"
    exit 1
fi
rm -f "$ROOT/.symlink_test" 2>/dev/null || true

# 1. Architecture Check
ARCH=$(uname -m)
echo "[1/6] Hardware architecture: $ARCH"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo "  [!] WARNING: Detected '$ARCH'. This setup is targeted for aarch64 (Raspberry Pi 5 64-bit OS)."
else
    echo "  [✓] 64-bit ARM architecture confirmed."
fi

# 2. Swap Check (Warn for 4GB Pi)
TOTAL_RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0")
SWAP_MB=$(free -m 2>/dev/null | awk '/^Swap:/{print $2}' || echo "0")
echo "[2/6] Memory check: Total RAM = ${TOTAL_RAM_MB} MB | Active Swap = ${SWAP_MB} MB"
if [ "$TOTAL_RAM_MB" -gt 0 ] && [ "$TOTAL_RAM_MB" -lt 7000 ] && [ "$SWAP_MB" -lt 1500 ]; then
    echo "  [!] Notice: On a 4GB Pi 5, at least 2GB swap is recommended for Llama weight loading."
    echo "      To configure 2GB swap:"
    echo "        sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile"
    echo "        sudo dphys-swapfile setup && sudo dphys-swapfile swapon"
fi

# 3. System Packages Check (APT)
echo "[3/6] Checking system-level tools (venv, pip, git)..."
MISSING_APT=()

if ! python3 -c "import venv" >/dev/null 2>&1; then
    MISSING_APT+=("python3-venv")
fi

if ! command -v git >/dev/null 2>&1; then
    MISSING_APT+=("git")
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
    MISSING_APT+=("python3-pip")
fi

if [ "${#MISSING_APT[@]}" -gt 0 ]; then
    echo "  [+] Missing system packages: ${MISSING_APT[*]}"
    echo "  [+] Installing missing packages via apt..."
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        else
            echo "  [!] Error: 'sudo' is required to install system packages (${MISSING_APT[*]})."
            echo "      Please run as root or ask your administrator to install: ${MISSING_APT[*]}"
            exit 1
        fi
    fi
    $SUDO apt update || echo "  [!] apt update returned non-zero, continuing with current package lists..."
    $SUDO apt install -y "${MISSING_APT[@]}"
else
    echo "  [✓] All required system tools (python3-venv, pip, git) are already present."
fi

# 4. Virtual Environment Check
echo "[4/6] Checking Python virtual environment..."
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "  [+] Creating clean virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
else
    echo "  [✓] Virtual environment already exists at $VENV_DIR. Skipping creation."
fi

# 5. Python Libraries Check (PIP)
echo "[5/6] Checking Python dependencies inside virtual environment..."

# PyTorch & Torchvision
if ! "$VENV_DIR/bin/python" -c "import torch, torchvision" >/dev/null 2>&1; then
    echo "  [+] Installing PyTorch & Torchvision for ARM64 CPU..."
    "$VENV_DIR/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
    "$VENV_DIR/bin/python" -m pip install torch torchvision
else
    echo "  [✓] PyTorch & Torchvision already installed."
fi

# Core dependencies (Pillow, numpy, transformers, optimum, etc.)
MISSING_PIP=()
for mod in numpy PIL huggingface_hub accelerate optimum transformers tokenizers; do
    if ! "$VENV_DIR/bin/python" -c "import $mod" >/dev/null 2>&1; then
        MISSING_PIP+=("$mod")
    fi
done

if [ "${#MISSING_PIP[@]}" -gt 0 ]; then
    echo "  [+] Installing missing Python modules: ${MISSING_PIP[*]}..."
    "$VENV_DIR/bin/python" -m pip install numpy Pillow huggingface_hub "accelerate>=0.26.0" "optimum~=2.0.0" "transformers==5.0.0rc1" "tokenizers==0.22.2"
else
    echo "  [✓] Supporting Python packages (numpy, Pillow, transformers, optimum, tokenizers) already installed."
fi

# ExecuTorch Core
if ! "$VENV_DIR/bin/python" -c "import executorch" >/dev/null 2>&1; then
    echo "  [+] Installing ExecuTorch..."
    "$VENV_DIR/bin/python" -m pip install executorch
else
    echo "  [✓] ExecuTorch already installed."
fi

# Optimum-ExecuTorch (Installed locally inside venv/src to avoid /tmp permission clashes)
if ! "$VENV_DIR/bin/python" -c "import optimum.executorch" >/dev/null 2>&1; then
    echo "  [+] Installing optimum-executorch..."
    SRC_DIR="$VENV_DIR/src/optimum-executorch"
    mkdir -p "$(dirname "$SRC_DIR")"
    if [ ! -d "$SRC_DIR/.git" ]; then
        rm -rf "$SRC_DIR" 2>/dev/null || true
        git clone --depth 1 https://github.com/huggingface/optimum-executorch.git "$SRC_DIR" || {
            echo "  [!] Warning: Could not clone optimum-executorch from GitHub."
            echo "      Please verify internet connection on the Pi."
        }
    fi
    if [ -d "$SRC_DIR" ]; then
        "$VENV_DIR/bin/python" -m pip install --no-deps "$SRC_DIR"
    fi
else
    echo "  [✓] optimum-executorch already installed."
fi

# 6. Verification
echo "[6/6] Verifying end-to-end environment..."
"$VENV_DIR/bin/python" -c "
import sys
try:
    import torch, torchvision, numpy, PIL
    print('  [✓] Vision Stack Ready:')
    print('      - PyTorch:', torch.__version__)
    print('      - Torchvision:', torchvision.__version__)
except Exception as e:
    print('  [!] Vision Stack Error:', e, file=sys.stderr)

try:
    import executorch, optimum.executorch, transformers
    print('  [✓] Edge LLM Stack Ready:')
    print('      - ExecuTorch:', executorch.__version__)
    print('      - Optimum-ExecuTorch: Ready')
    print('      - Transformers:', transformers.__version__)
except Exception as e:
    print('  [!] Notice regarding LLM Stack:', e)
"

echo "=========================================================="
echo "  Setup is complete and verified!                         "
echo "  To activate in your terminal:                           "
echo "    source $VENV_DIR/bin/activate                         "
echo "  To run vision test:                                     "
echo "    python3 test_vision.py                                "
echo "  To start live inspection:                               "
echo "    python3 inspect_live.py --watch-dir sequence_30 -x 1.0"
echo "=========================================================="
