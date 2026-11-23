# Results Summary for Technical Report
**Bharat AI-SoC Challenge 2026–27 | Problem Statement 5**

## 1. Target Hardware & Operating Environment
- **Platform:** Raspberry Pi 5 Model B Rev 1.1
- **CPU:** Broadcom BCM2712 Quad-Core ARM Cortex-A76 @ 2.4 GHz (ARMv8.2-A, aarch64)
- **RAM:** 8,251,728 kB (~7.86 GiB physical LPDDR4X-4267; >6.2 GiB available headroom)
- **Operating System:** Debian GNU/Linux 13 (trixie) | Kernel 6.18.50+rpt-rpi-2712-arm64
- **Software Stack:** Python 3.13.5, PyTorch 2.15.0+cpu, TorchVision 0.29.2+cpu, ExecuTorch 1.6.0 (XNNPACK)
- **Hardware Telemetry:** Initial Temp: 58.7°C, Volts: 0.9317V, Throttling Register: 0x0 (no throttling)

---

## 2. Dataset Citation & Splitting Integrity
- **Dataset Source:** Single-yarn defect dataset from Mendeley Data (DOI: 10.17632/t7hcfmn482.1, CC BY 4.0).
- **Split Distribution:** 3,000 images across 6 classes (500 images per class).
  - Train: 2,100 images (70%)
  - Val: 450 images (15%)
  - Test: 450 images (15%)
- **Data Integrity Audit:**
  - Analysis revealed that standard random splitting interleaved consecutive video frames across train/val/test (>220 transitions per class), synthetically inflating baseline scores to 100%.
  - To ensure scientific validity, a contiguous chronological split was reconstructed (earliest 70% to train, middle 15% to val, final 15% future frames to test).

---

## 3. Vision Model (MobileNetV3-Small) Performance on ARM Cortex-A76
- **Architecture:** MobileNetV3-Small (`yarn.pt`, 6.0 MB, 1,524,006 parameters, FP32).
- **Preprocessing Pipeline:** Central vertical band crop (`crop_band`, 30%–70%) + `Resize((96, 768))` + ImageNet Normalization.
- **Evaluation Accuracy:**
  - **Held-Out Chronological Test Split (N=450 Future Frames):** **98.00% Overall Accuracy** (Macro-F1: 0.9799).
  - **Static Test Split (N=436 Frames):** **100.00% Overall Accuracy**.
- **Per-Class Breakdown (Chronological Split):**
  - `irregularity`: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 (75/75)
  - `knotting`: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 (75/75)
  - `long segment lack`: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 (75/75)
  - `normal`: Precision: 0.8929 | Recall: 1.0000 | F1: 0.9434 (75/75)
  - `short segment lack`: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 (75/75)
  - `thinness`: Precision: 1.0000 | Recall: 0.8800 | F1: 0.9362 (66/75)
- **On-Device Cortex-A76 Latency (Measured on Pi 5):**
  - Image Decode (PIL): 5.30 ms
  - Preprocessing: 6.77 ms
  - Forward Inference: 24.72 ms (Min: 24.54 ms, Max: 24.86 ms)
  - Total Per-Frame Latency: 36.79 ms
  - Throughput: **40.5 FPS** (comfortably exceeds 30 FPS physical line speed by 35%).

---

## 4. Edge LLM Advisory (ExecuTorch Llama 3.2 1B Instruct on ARM64)
- **Model Checkpoint:** `llama_model/model.pte` (914.12 MiB, 8da4w linear, 8w embedding, bfloat16, XNNPACK backend).
- **Prompt Strategy:** Low-latency 1-shot template (~199 tokens), reducing prompt prefill time by ~2.03s compared to 2-shot.
- **On-Device Performance (Measured on Pi 5):**
  - Cold Load Time: 7.40 s (from storage to RAM).
  - Resident RAM Footprint: 668.9 MB after load; 1,496.3 MB peak generation RSS.
  - Time-To-First-Token (TTFT): 6.55 s (measured via single-token prefill test).
  - Total Advisory Response Time: Mean = 6.69 s (Min: 4.99 s, Max: 13.39 s).
  - Decode Generation Speed: Mean = 3.8 tok/s (Min: 1.42 tok/s, Max: 4.61 tok/s).
  - Validator Pass Rate: 10 / 10 (100.0% strict two-line `ACTION:` and `CHECK:` syntax compliance).

---

## 5. Streaming Architecture & Fallback Safety
- **Dual-Panel Live Execution:**
  - Terminal 1: Live operator advisory card (`inspect_live.py --view-advisory`).
  - Terminal 2: Real-time inspection folder watcher (`inspect_live.py --watch-dir sequence_30 -x 1.0`).
- **Decoupled Asynchronous Processing:**
  - Vision classifier processes frames synchronously at 1.0s intervals.
  - Normal frames bypass the LLM via the fast path (<0.1 ms logging).
  - Defect frames trigger an asynchronous background Llama advisory worker.
  - If a defect occurs while Llama is actively generating advice, an instantaneous deterministic rule-based fallback advisory is logged (`advice_source: fallback`), guaranteeing 0 pipeline stalls and 0 dropped inspection records.
