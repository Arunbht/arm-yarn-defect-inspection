# Raspberry Pi 5 Deployment: On-Device Single-Yarn Defect Inspection & Edge GenAI Advisory

This folder is a self-contained, standalone deployment package designed specifically to be copied and run on a **Raspberry Pi 5 (ARM64)**. All training, quantization, and evaluation prerequisites have already been completed on the host workstation.

---

## 1. Hardware & System Prerequisites

- **Target Board:** Raspberry Pi 5 (4GB or 8GB model recommended)
- **Processor:** Broadcom BCM2712 quad-core Arm Cortex-A76 @ 2.4 GHz
- **Operating System:** Raspberry Pi OS (64-bit) / Debian 12 (Bookworm) `aarch64`
- **Free Disk Space:** Minimum 4 GB free space
- **Swap Space (Important for 4GB Pi 5):**
  - The quantized Llama 3.2 1B ExecuTorch model consumes ~1.4 GB on load and peaks at ~2.1 GB during generation.
  - If using a **4GB Raspberry Pi 5**, configure at least **2 GB of swap** to prevent Linux kernel out-of-memory (OOM) kills:
    ```bash
    sudo dphys-swapfile swapoff
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
    ```

---

## 2. Directory Contents

```text
rpi5_deployment/
├── yarn.pt               # Fine-tuned MobileNetV3-Small weights (6.0 MB, 98.00% chronological acc)
├── llama_model/          # Quantized ExecuTorch XNNPACK Llama 3.2 1B assets (~935 MB)
│   ├── model.pte         # 8da4w/8w quantized ExecuTorch binary (914 MB)
│   ├── tokenizer.json    # Fast tokenizer vocabulary
│   ├── tokenizer_config.json
│   └── chat_template.jinja
├── llm_advice.py         # 1-shot on-device LLM operator advisory daemon (CLI & JSON IPC)
├── inspect_live.py       # Live provider watcher (processes 1 image every X seconds)
├── benchmark_pi_complete.py # Comprehensive all-in-one telemetry & benchmark suite
├── sequence_30/          # 30 ordered industrial test frames (normal, irregularity, knotting)
├── test_vision.py        # Sanity check & latency benchmark script for the vision model
├── setup_rpi5.sh         # Smart sequential environment setup & check script
├── requirements_rpi5.txt # Pinned pip dependencies for ARM64 / aarch64
├── latest_output.json    # Live JSON output for external UI / dashboards
├── latest_advice.txt     # Real-time 2-line plain text operator guidance
├── inspection_log.csv    # Continuous audit trail spreadsheet
└── README.md             # This guide
```

---

## 3. Quickstart: Setting up on Raspberry Pi 5

### Step 1: Copy to your Raspberry Pi
Transfer the updated scripts to your Raspberry Pi 5:
```bash
# From laptop terminal to Pi:
scp setup_rpi5.sh inspect_live.py llm_advice.py test_vision.py requirements_rpi5.txt pi@<PI_IP>:~/rpi5_deployment/
```

### Step 2: Run the automated environment setup
On your Raspberry Pi 5 terminal, enter the directory and run:
```bash
cd ~/rpi5_deployment
chmod +x setup_rpi5.sh test_vision.py inspect_live.py llm_advice.py
./setup_rpi5.sh
```
The smart setup script will:
1. Verify `aarch64` architecture and ext4 symlink compatibility.
2. Check memory and notify if 2GB swap is recommended.
3. Check and sequentially install missing system packages (`python3-venv`, `pip`, `git`).
4. Build a local virtual environment (`rpi_env`).
5. Check and install ARM64 CPU-optimized PyTorch, Torchvision, and Pillow.
6. Check and install `optimum-executorch` into a self-contained local build directory.
7. Verify all modules can be imported with zero errors.

### Step 3: Activate the virtual environment
```bash
source rpi_env/bin/activate
```

---

## 4. Verification & Testing

### A. Verify Vision Model (`test_vision.py`)
Run the standalone vision benchmark:
```bash
# Auto-discover sequence_30 and benchmark:
python3 test_vision.py

# Benchmark all 30 frames in sequence_30:
python3 test_vision.py sequence_30

# Test a single frame:
python3 test_vision.py sequence_30/01_normal_20230301135515286.bmp

# Test a non-yarn image to verify Out-of-Distribution (OOD) rejection:
python3 test_vision.py some_photo.jpg
```
**Expected Output:**
- Loads `yarn.pt` onto the CPU with PyTorch 2.4+ compatibility.
- Runs Out-of-Distribution validation (rejects cluttered/non-yarn images).
- Runs cold-start warm-up pass.
- Reports prediction, confidence, and steady-state CPU latency per frame (**20–35 ms / ~30–50 FPS** on Cortex-A76).

### B. Verify Llama 3.2 1B Advisory (`llm_advice.py`)
Test the on-device quantized LLM generating operator guidance:
```bash
# Immediate self-test (no arguments needed):
python3 llm_advice.py

# Or query a specific defect:
python3 llm_advice.py knotting HIGH 0.99 --history "5 defects in last 20 yarns (knotting)"
```
**Expected Output:**
```text
=======================================================
Defect:     KNOTTING (99.0%) | Severity: HIGH
-------------------------------------------------------
ACTION: Pause winding spindle and check knotter alignment.
CHECK: Inspect knotter knife and clean yarn path guides.
-------------------------------------------------------
Cortex-A76 Performance: 42 tokens generated in 7.82s (~5.4 tok/s)
=======================================================
```

### C. Run the Live Inspection Watcher (`inspect_live.py`)
Run the real-time folder watcher. It monitors `sequence_30`, processes existing frames spaced by $X$ seconds, and immediately reacts when you drop new images into `sequence_30`:

```bash
# ==========================================================
# Strategy 3: Split-Terminal Mode (Side-by-Side Viewing)
# ==========================================================

# TERMINAL 1 (Left Window - Live Camera Vision Feed, 25ms):
python3 inspect_live.py -x 1.5

# TERMINAL 2 (Right Window - GenAI Operator Guidance Monitor):
python3 inspect_live.py --view-advisory

# OR Launch both side-by-side automatically with 1 command (via tmux):
chmod +x launch_split.sh
./launch_split.sh
```

---

## 5. Performance Targets & Resource Utilization

| Component | Model / Format | Memory (RAM) | Expected Speed on RPi 5 |
| :--- | :--- | :--- | :--- |
| **Vision Classifier** | MobileNetV3-Small (`yarn.pt`) | ~180 MB | **~25–35 ms / frame (30–45 FPS)** |
| **Edge LLM Advisor** | Llama 3.2 1B (`model.pte`) | ~1.4 GB (load) / ~2.1 GB (peak) | **~3.5–5.5 tokens/sec** |

---

## 6. Built-in Error Handling & Safety Features

1. **Copy Race-Condition Protection:** When new images are copied over network/USB, a 3-attempt progressive backoff (100ms, 250ms, 500ms) prevents reading incomplete image writes.
2. **Out-of-Distribution (OOD) Guard:** Automatically checks optical backlight, aspect ratio ($\ge 1.3:1$), and clutter variance to reject non-yarn photos with `[REJECTED - WRONG / NON-YARN IMAGE]`. Can be bypassed with `--no-ood`.
3. **Thread-Safe Atomic I/O:** `latest_output.json`, `latest_advice.txt`, and `inspection_log.csv` use thread locks and atomic replacement (`.tmp` -> replace) to prevent UI read collisions.
4. **Graceful Daemon IPC:** Inter-process communication between `inspect_live.py` and `llm_advice.py` separates `stderr` logs from JSON-RPC `stdout`, preventing JSON parsing errors.
5. **Clean Signal Termination:** Traps both `SIGINT` (Ctrl+C) and `SIGTERM` (systemd stop) to cleanly terminate the background Llama process without leaving zombie memory allocations.
