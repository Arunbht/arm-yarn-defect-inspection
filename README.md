# PS5: On-Device Single-Yarn Defect Detection & Edge Llama Advisory

Entry for the **Bharat AI-SoC Challenge 2026–27**, Problem Statement 5. 

This repository implements an on-device inspection and operator advisory system:
1. **Vision Classifier:** MobileNetV3-Small classifying single-yarn defects in ~15.9 ms on CPU (`yarn.pt`, 6.22 MB).
2. **Edge LLM Advisor:** Quantized Llama 3.2 1B Instruct exported via ExecuTorch/XNNPACK (`8da4w`/`8w`), generating two-line operator advisories (`ACTION:` and `CHECK:`) at ~10–19 tokens/s.
3. **Traceability & Alerting:** Automated CSV audit logging with operator terminal alerts for `MEDIUM` and `HIGH` severity defects, plus frame replay streaming.

---

## 1. Setup

A unified environment setup script `setup_env.sh` is provided. It installs PyTorch (CPU), Torchvision, Hugging Face Optimum, and ExecuTorch:

```bash
git clone <REPO_URL>
cd <REPO_FOLDER>

# Build the unified Python virtual environment
./setup_env.sh
source llmenv/bin/activate
```

*(Note: Torch and Torchvision must be installed from matching wheel builds to avoid C++ ABI mismatches such as missing `torchvision::nms` operators).*

---

## 2. Dataset Citation & Source

"Li, lizile (2025), 'Yarn Defect Dataset', Mendeley Data, V1, doi: 10.17632/t7hcfmn482.1. CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Download: https://data.mendeley.com/datasets/t7hcfmn482/1. Original dataset: 3,000 images, 500 per class. We split each class 350/75/75 (train/val/test)."

*(Note: The capture protocol and number of physical yarn samples are not described by the dataset page).*

### Directory Structure:
Extract the dataset to `Yarn Defect Dataset/` in the project root, or set `export YARN_DATASET_DIR="/path/to/dataset"`:
```text
Yarn Defect Dataset/
├── train/ (2,100 images; 350 per class)
├── val/   (450 images; 75 per class)
└── test/  (450 images; 75 per class)
```

Verify dataset integrity:
```bash
python check.py
# or: python check.py "/path/to/Yarn Defect Dataset"
```

---

## 3. Llama 3.2 1B ExecuTorch Export

The Llama model is quantized using Hugging Face Optimum and ExecuTorch XNNPACK:

1. Request access to [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face.
2. Authenticate:
   ```bash
   huggingface-cli login
   # or export HF_TOKEN="hf_..."
   ```
3. Run the memory-optimized export:
   ```bash
   ./export_llama.sh
   ```
   *Flags used:* `--recipe xnnpack --use_custom_sdpa --use_custom_kv_cache --max_seq_len 512 --dtype bfloat16 --qlinear 8da4w --qembedding 8w`.
   *(Limiting `--max_seq_len 512` and using `bfloat16` reduces peak export memory from >13 GB down to <3 GB, preventing out-of-memory kernel kills on 8 GB systems).*

4. Test standalone advice generation:
   ```bash
   python llm_advice.py knotting HIGH 0.99
   ```

---

## 4. Running the Inspection Pipeline

Run batch inspection on any image folder:
```bash
python inspect_yarn.py "Yarn Defect Dataset/test"
```

### Stream Replay Mode:
To simulate a real-time frame stream at a target FPS with non-blocking background LLM advisories and operator alert banners:
```bash
python inspect_yarn.py "Yarn Defect Dataset/test" --stream 2.0
```

### Expected Output:
```text
  >>> [OPERATOR ALERT - HIGH SEVERITY] <<< 
 DEFECT: KNOTTING (Confidence: 100.0%) | Trend: 8 defects in last 20 yarns (knotting)

  [ASYNC LLM ADVICE COMPLETED for Image_20230227100642991.bmp]
  Latency from Alert: 6.21s | Speed: 5.3 tok/s | Source: llm
  ACTION: Immediately stop spindle and inspect yarn for visible signs of knotting.
  CHECK: Verify yarn quality and inspect for any visible signs of damage or breakage.

[1/100] Image_20230227100642991.bmp -> knotting 1.00 [HIGH] (vision 29.6ms)
```

Results are saved to `inspection_log.csv` with columns:
`timestamp, file, defect, confidence, severity, advice, latency_ms, advice_source, llm_tokens_s`.

---

## 5. Model Evaluation & Benchmark Results

Run test split evaluation:
```bash
python test.py
# or: python test.py "/path/to/Yarn Defect Dataset"
```

### Measured Performance & Benchmarks:
- **Vision Model:** MobileNetV3-Small (`yarn.pt`, 6.22 MB).
- **Test Accuracy on Held-Out Split:** 100.0% (450/450 correct; 75 per class) *(measured)*.
- **Vision Latency:** ~15.9 ms per image on laptop CPU (~63 FPS idle; 19–109 ms under multi-threaded LLM load) *(measured)*.
- **LLM Model:** ExecuTorch XNNPACK (`model.pte`, 914.12 MiB) *(measured)*.
- **LLM Cold Load Time:** 6.45 s *(measured)*.
- **LLM RAM Footprint:** 1,384 MB load, 2,130 MB peak generation *(measured)*.
- **LLM Generation Speed:** 3.8–6.5 tokens/s (35–48 tokens generated per advisory) *(measured)*.
- **Advice Latency (Alert to Completion):** 5.20–8.62 s per advisory *(measured)*.
- **Stream Benchmarks (100-Frame Stress Test):**
  - **2.0 FPS Stream:** 1.96 sustained FPS, 4 dropped frames, 6 background LLM calls dispatched, 5.71 s avg advice latency *(measured)*.
  - **10.0 FPS Stream:** 7.18 sustained FPS, 66 dropped frames, 2 background LLM calls dispatched, 8.10 s avg advice latency *(measured)*.
- **Operator Advisory Notice:** Generated advisories are suggestions for human machine operators, not automatic machine actions. All critical mechanical shutdowns require operator confirmation.

---

## 6. Dataset & Methodology Disclosure

- **Data Splits:** 2,100 train / 450 val / 450 test (70/15/15 stratified split across 6 classes).
- **Duplicate Verification:** SHA-256 hash comparison across all 3,000 images confirmed **0 exact duplicate files** and **0 duplicate filenames** across splits. Near-duplicates or temporal video correlations were not evaluated.
- **Training Iterations:** The initial `train.py` run evaluated the held-out test split at the end of training. The architectural refinements in `train2.py` (central vertical band crop, 96×768 aspect ratio, horizontal flip, and fine-tuning layer 9+) were informed by those initial observations before `RUN_TEST` was frozen.

---

## 7. Limitations & Honest Engineering Notes

1. **Hardware Scope:** All benchmarks were measured on a development laptop (Intel Core i3-1315U, 7.5 GB RAM, Linux Mint 22.3). Deployment on an actual Arm board (Raspberry Pi, Cortex-A SoC) is pending physical hardware testing.
2. **Dataset Nature:** The dataset was collected under controlled studio/lab lighting. Performance under varying ambient factory lighting or dust accumulation requires industrial domain adaptation.
3. **Severity Triage:** The `SEVERITY` mapping (`NONE`, `MEDIUM`, `HIGH`) is a heuristic rule based on manufacturing defect impact; it should be calibrated to plant-specific tolerance thresholds.
4. **Estimates Clarification:** The visual cropping was estimated to cut convolutional pixel area by >60%; actual inference latency dropped from ~22 ms to ~16 ms. The vision process baseline RAM is estimated at ~180 MB.

---

## 8. Technical Challenges & Workarounds Encountered

1. **Host OOM on ExecuTorch Export:** Tracing Llama 3.2 1B with default 131k context in FP32 required >13.3 GB of virtual memory, triggering the Linux kernel OOM killer. Fixed by setting `--max_seq_len 512` and `--dtype bfloat16`.
2. **Missing Tokenizer Files in Output:** The exporter wrote `model.pte` without tokenizer files, causing input shape mismatches (`torch.Size([1, 0])`) in XNNPACK. Fixed by saving tokenizers into `llama_model/`.
3. **Torchvision ABI Mismatch:** Pre-existing `yarnenv` failed with `operator torchvision::nms does not exist` due to CUDA/CPU build divergence. Resolved by unifying dependencies into a single ABI-matched environment.
4. **Duplicate Optimum CLI Commands:** Installing `optimum-executorch` in editable mode generated conflicting subparser errors. Resolved by cleaning up dist-info metadata.
