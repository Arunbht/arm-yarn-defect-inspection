# PS5: On-device single-yarn defect detection

Entry for Bharat AI-SoC Challenge 2026-27, Problem Statement 5. The system classifies a yarn image, assigns a severity, produces a short Llama advisory, and logs the result.

## Dataset and rights

The dataset is intentionally excluded from Git. Before publishing, add its original source, license/permission, and citation here. It must be used only when you have the required rights.

## Approach

- Vision: MobileNetV3-Small transfer learning, central 30–70% vertical band, resized to 96×768.
- Classes: irregularity, knotting, long segment lack, normal, short segment lack, thinness.
- Advisory: Llama 3.2 1B Instruct exported to ExecuTorch/XNNPACK with `8da4w` linear and `8w` embedding quantization.
- Runtime: `inspect_yarn.py` keeps one Llama process alive for the complete image batch. If it cannot start or reply, it keeps the existing deterministic safety fallback.

## Results

| Split | Images | Per class |
|---|---:|---:|
| Train | 2,100 | 350 |
| Validation | 450 | 75 |
| Test | 450 | 75 |

Reported vision results: validation accuracy 100%, test accuracy 100% (450/450), 6.22 MB checkpoint, and approximately 15.9 ms/image on laptop CPU. Run `python test.py` once only for the report’s final confusion matrix; do not tune on the test split.

Llama latency and tokens/s are printed by `llm_advice.py` after export. Record those measured values rather than estimates.

## Setup

```bash
cd ~/Downloads/ARM_CONTEST
source yarnenv/bin/activate
python check.py
python test.py
```

### Llama advisory

```bash
cd ~/Downloads/ARM_CONTEST
./setup_llmenv.sh
source llmenv/bin/activate
huggingface-cli login                 # after accepting Meta's model license
./export_llama.sh
python llm_advice.py knotting HIGH 0.98
```

`export_llama.sh` uses the current Optimum ExecuTorch XNNPACK exporter. It creates `llama_model/`, which is deliberately ignored by Git.

## Inspection

```bash
cd ~/Downloads/ARM_CONTEST
source yarnenv/bin/activate
python inspect_yarn.py "path/to/images"
```

The process writes `inspection_log.csv`. For the full advisory integration, launch the same command after the Llama model has been exported; `inspect_yarn.py` uses `llmenv/bin/python` directly for its advisory subprocess.

## Deliverables still needed

- Add dataset source, license/permission, and citation.
- Record final confusion matrix and classification report from the untouched test set.
- Record Llama measured latency and tokens/s after export.
- Add working-application screenshots/demo video and technical report.
- Create the empty GitHub repository under `Arunbht`, then add its remote and push.
