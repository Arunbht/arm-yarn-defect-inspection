# Results summary for technical report

## Dataset split

| Split | Images | Images per class |
|---|---:|---:|
| Train | 2,100 | 350 |
| Validation | 450 | 75 |
| Test | 450 | 75 |
| Total | 3,000 | 500 |

Classes: irregularity, knotting, long segment lack, normal, short segment lack, thinness.

## Vision model

- Architecture: MobileNetV3-Small transfer learning.
- Input: centre 30–70% vertical yarn band, resized to 96×768.
- Checkpoint: `yarn.pt`, 6.22 MB.
- Reported validation accuracy: 100.0%.
- Reported held-out test accuracy: 100.0% (450/450 images).
- Reported laptop CPU latency: approximately 15.9 ms/image.

If the reported 450/450 test result is reproduced, the confusion matrix in the class order above is:

| Actual \\ Predicted | Irregularity | Knotting | Long lack | Normal | Short lack | Thinness |
|---|---:|---:|---:|---:|---:|---:|
| Irregularity | 75 | 0 | 0 | 0 | 0 | 0 |
| Knotting | 0 | 75 | 0 | 0 | 0 | 0 |
| Long lack | 0 | 0 | 75 | 0 | 0 | 0 |
| Normal | 0 | 0 | 0 | 75 | 0 | 0 |
| Short lack | 0 | 0 | 0 | 0 | 75 | 0 |
| Thinness | 0 | 0 | 0 | 0 | 0 | 75 |

## Llama advisory measurement (Measured on Hardware)

Standalone execution benchmark:
```bash
python llm_advice.py knotting HIGH 0.98
```

- **Model Checkpoint:** `llama_model/model.pte` (Quantized via ExecuTorch/XNNPACK, `8da4w` linear, `8w` embedding, `bfloat16`, max sequence length 512)
- **Model Checkpoint Size:** 915 MB
- **Measured Generated Tokens:** 64 tokens
- **Cold Standalone Latency:** 4.78 seconds (13.38 tokens/s)
- **Warm Continuous Batch Advisory Latency:** 3,691.4 ms (~3.69s) average per image across full 75-image batch run
- **End-to-End Pipeline Verification:** Verified via `inspect_yarn.py` on held-out test split, logging defect class, confidence, severity tier, live Llama advisory, and latency to `inspection_log.csv`.
- **Output Sample:**
  ```text
  Assessment: The machine is prone to knotting, which can lead to catastrophic failure. Immediate action is to inspect the machine for any signs of knotting. One follow-up check is to verify that the machine is functioning properly and that the knotting issue has been resolved.
  ```
