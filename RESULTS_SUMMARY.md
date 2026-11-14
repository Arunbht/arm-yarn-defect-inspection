# Results Summary for Technical Report
**Bharat AI-SoC Challenge 2026–27 | Problem Statement 5**

## 1. Dataset Citation, Split & Hash Integrity
- **Citation:** "Li, lizile (2025), 'Yarn Defect Dataset', Mendeley Data, V1, doi: 10.17632/t7hcfmn482.1. CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Download: https://data.mendeley.com/datasets/t7hcfmn482/1. Original dataset: 3,000 images, 500 per class. We split each class 350/75/75 (train/val/test)."
- **Disclosures:** The capture protocol and number of physical yarn samples are not described by the dataset page.
- **Split Distribution:**
  - `train`: 2,100 (350 per class)
  - `val`: 450 (75 per class)
  - `test`: 450 (75 per class)
- **Hash Integrity:** SHA-256 analysis confirmed **0 duplicate filenames** and **0 duplicate files** across splits. Near-duplicates were not evaluated.

## 2. Vision Model (MobileNetV3-Small)
- **Architecture:** MobileNetV3-Small (`yarn.pt`, 6.22 MB) *(measured)*.
- **Input Size:** `(96, 768)` after central 40% vertical band crop (`crop_band`).
- **Validation Accuracy:** 100.0% *(measured)*.
- **Held-Out Test Accuracy:** 100.0% (450/450 correct) *(measured)*.
- **Latency:** ~15.9 ms per image idle (~63 FPS idle; 19–109 ms under multi-threaded LLM load) on Intel Core i3-1315U CPU *(measured)*.
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
- **Checkpoint:** `llama_model/model.pte` (914.12 MiB) *(measured)*.
- **Quantization:** `8da4w` linear, `8w` embedding, `bfloat16`, max sequence length 512 *(measured)*.
- **Model Load Time:** 6.45 s *(measured)*.
- **RAM Footprint:** 1,384 MB load, 2,130 MB peak during generation *(measured)*.
- **LLM Generation Speed:** 3.8–6.5 tokens/s across defect types *(measured)*.
- **Advice Latency (Alert to Completion):** 5.20–8.62 s per advisory *(measured)*.
- **Output Validation:** Strict validator enforcing both `ACTION:` and `CHECK:`, $\le 250$ characters, and 0 repeated lines.
- **Human Operator Notice:** Generated advice is an advisory suggestion for a human machine operator, not an automatic machine action. All critical machine interventions require operator confirmation.

### Context-Aware Advisory Adaptation (Task C):
The LLM prompt incorporates rolling history of the last $N=20$ inspections (defect count and dominant defect type):
- **Isolated Knotting (0 defects in window):**
  ```text
  ACTION: Resume winding spindle and adjust tension on bobbin.
  CHECK: Inspect bobbin for proper tension and yarn alignment.
  ```
- **Recurring Knotting Cluster (8 defects in window):**
  ```text
  ACTION: Perform yarn re-wind and re-tension on winding spindle.
  CHECK: Inspect yarn for excessive twist and check for any signs of breakage.
  ```

## 4. Streaming Performance Benchmarks (Task B)
Measured across a 100-frame test sequence using asynchronous queue size 1 and a 10s per-class cooldown:
- **2.0 FPS Stream Target:**
  - Sustained FPS: **1.96 FPS** *(measured)*
  - Dropped Frames: **4** *(measured)*
  - Background LLM Calls: **6** *(measured)*
  - Average Advice Latency: **5.71 s** (range: 5.20 s – 6.42 s) *(measured)*
- **10.0 FPS Stream Target:**
  - Sustained FPS: **7.18 FPS** *(measured)*
  - Dropped Frames: **66** *(measured; caused by CPU saturation from concurrent XNNPACK inference)*
  - Background LLM Calls: **2** *(measured)*
  - Average Advice Latency: **8.10 s** (range: 7.58 s – 8.62 s) *(measured)*

## 5. Engineering Limitations & Estimates
- All measurements performed on an Intel Core i3-1315U laptop (6 cores, 7.5 GB RAM, Linux Mint 22.3). Physical Arm board port is pending.
- Convolutional area reduction from cropping (>60%) and vision standalone RAM (~180 MB) are engineering estimates.
- Severity mapping (`NONE`, `MEDIUM`, `HIGH`) is a heuristic manufacturing risk triage.
