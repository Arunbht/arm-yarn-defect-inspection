# PS5: On-Device Single-Yarn Defect Detection & Edge GenAI Advisory

Entry for the **Bharat AI-SoC Challenge 2026–27**, Problem Statement 5.

This repository implements an end-to-end, fully air-gapped on-device inspection and machine operator advisory system:
1. **Vision Classifier:** Fine-tuned MobileNetV3-Small classifying single-yarn defects (`yarn.pt`, 6.0 MB). Preprocessed with vertical central crop (`crop_band`, 30%–70%) and `Resize((96, 768))`. Evaluated at **98.00% accuracy on held-out chronological future frames** (and 100% on legacy random split).
2. **Edge LLM Advisor:** Quantized Llama 3.2 1B Instruct exported via ExecuTorch/XNNPACK (`8da4w`/`8w`), generating two-line machine operator instructions (`ACTION:` and `CHECK:`) using a low-latency 1-shot prompt template (Pi 5: 6.69 s average response, 3.8 tok/s end-to-end).
3. **Traceability & Alerting:** Automated CSV audit logging (`inspection_log.csv`) with live dual-panel terminal operator alerts and deterministic rule-based fallback when the LLM daemon is generating.

---

## 1. Quickstart & Raspberry Pi 5 Deployment

All scripts and configuration required for the Raspberry Pi 5 are located in `rpi5_deployment/`.

### Setup Environment
```bash
cd rpi5_deployment
chmod +x setup_rpi5.sh inspect_live.py llm_advice.py test_vision.py benchmark_pi_complete.py
./setup_rpi5.sh
source rpi_env/bin/activate
```

### Live Inspection Demo
Run the split-terminal live inspection system:

- **Terminal 1 (Dedicated Operator Advisory Display):**
  ```bash
  python3 inspect_live.py --view-advisory
  ```

- **Terminal 2 (Live Vision Inspection Engine at 1.0s Cadence):**
  ```bash
  python3 inspect_live.py --watch-dir sequence_30 -x 1.0
  ```

---

## 2. All-In-One Benchmark Suite

To measure and record hardware telemetry, per-stage vision timings, TTFT, 10 advisory calls, and a 5-minute sustained stream:
```bash
cd rpi5_deployment
python3 benchmark_pi_complete.py
```
This produces a terminal report and saves a detailed machine-readable record to `pi5_benchmark_results.json`.

---

## 3. Performance Summary

Measured values only. "Pending" means not yet measured at the deployed 96x768 input on the Raspberry Pi 5.

| Component | Measured on | Result |
|---|---|---|
| Vision classifier (MobileNetV3-Small, 96x768) | Laptop, Intel i3-1315U | 33.6 ms inference, 91.5 ms camera-to-alert, 23.4 FPS (classifier only) |
| Vision classifier | Raspberry Pi 5 | Image read 5.30 ms; preprocess + inference at 96x768 pending |
| Llama 3.2 1B (1-shot) | Raspberry Pi 5 | 6.69 s average response (10 calls), 3.8 tok/s end-to-end, 7.4 s load, 1,496 MB peak RSS; TTFT not measured on the Pi |
| Llama 3.2 1B (1-shot) | Laptop | 7.06 s mean TTFT, 9.12 s mean total (30 prompts) |
| Offline pipeline | Laptop (network disabled) | Full pipeline ran with advice_source = llm |

See the technical report for methodology, limitations and pending items.

---

## 4. Models & Evaluation

- **Vision Accuracy:**
  - **Chronological Split (Unseen Future Frames, N=450):** **98.00% Accuracy**, Macro-F1: 0.9799 (5 of 6 classes 100% accurate; minor confusion only between sub-millimeter thinness and normal).
  - **Static Test Split (N=436):** **100.00% Accuracy**.
- **LLM Prompt Strategy:** 1-shot template (199 prompt tokens), reducing prefill latency by ~2.03s compared to 2-shot with 100% ACTION/CHECK formatting compliance.

---

## 5. Dataset Attribution & License

- **Dataset:** Single-yarn defect dataset from Mendeley Data (DOI: 10.17632/t7hcfmn482.1, CC-BY 4.0).
- **License:** Open-source software released under the permissive MIT License for academic and contest evaluation.
