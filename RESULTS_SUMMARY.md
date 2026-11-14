# Results Summary for Technical Report
**Bharat AI-SoC Challenge 2026–27 | Problem Statement 5**

## 1. Dataset Split & Hash Integrity
- **Distribution:** 3,000 total images balanced across 6 classes:
  - `train`: 2,100 (350 per class)
  - `val`: 450 (75 per class)
  - `test`: 450 (75 per class)
- **Hash Integrity:** SHA-256 analysis confirmed **0 duplicate filenames** and **0 duplicate files** across splits. Near-duplicates were not evaluated.

## 2. Vision Model
- **Architecture:** MobileNetV3-Small (`yarn.pt`, 6.22 MB).
- **Input Size:** `(96, 768)` after central 40% vertical crop (`crop_band`).
- **Validation Accuracy:** 100.0%.
- **Held-Out Test Accuracy:** 100.0% (450/450 correct).
- **Latency:** ~15.9 ms per image on Intel Core i3-1315U CPU.
- **Confusion Matrix (Held-out Test Split):**

| Actual \ Predicted | Irregularity | Knotting | Long lack | Normal | Short lack | Thinness |
|---|---:|---:|---:|---:|---:|---:|
| Irregularity | 75 | 0 | 0 | 0 | 0 | 0 |
| Knotting | 0 | 75 | 0 | 0 | 0 | 0 |
| Long lack | 0 | 0 | 75 | 0 | 0 | 0 |
| Normal | 0 | 0 | 0 | 75 | 0 | 0 |
| Short lack | 0 | 0 | 0 | 0 | 75 | 0 |
| Thinness | 0 | 0 | 0 | 0 | 0 | 75 |

*(Note: The test split was initially evaluated in the exploratory train.py before train2.py was finalized).*

## 3. Llama 3.2 1B Advisory (ExecuTorch / XNNPACK)
- **Checkpoint:** `llama_model/model.pte` (914.12 MiB).
- **Quantization:** `8da4w` linear, `8w` embedding, `bfloat16`, max sequence length 512.
- **Decoding:** Greedy decoding, capped at 48 new tokens with strict two-line validation (`ACTION:` and `CHECK:`).
- **Cold Standalone Inference:** 35 tokens in 6.02 s (5.82 tokens/s; load time: 6.45 s; peak RAM: 2,130 MB).
- **Warm Continuous Batch Inference:** ~3.69–5.5 s per image (approx 10–19 tokens/s).
- **Output Sample (Chat Template with Few-Shot):**
  ```text
  ACTION: Perform manual inspection of the knotting section to verify the knot is removed.
  CHECK: Verify the knot is removed and the section is clean and free of debris.
  ```

## 4. Engineering Limitations
- Host-only measurements on Intel Core i3-1315U with 7.5 GB RAM. Target Arm hardware testing is pending.
- Visual cropping compute reduction (>60%) and vision RAM (~180 MB) are engineering estimates.
- Severity mapping is a heuristic risk triage.
