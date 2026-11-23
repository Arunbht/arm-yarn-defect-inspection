#!/usr/bin/env python3
"""
ExecuTorch Llama 3.2 1B Operator Advisory Component for Raspberry Pi 5.
------------------------------------------------------------------------
Generates context-aware, 2-line machine-operator advisories (ACTION & CHECK).
Operates in two modes:
  1. CLI Mode:        python3 llm_advice.py knotting HIGH 0.99
  2. JSON Daemon IPC: python3 llm_advice.py --serve (for background live inspection)
"""
import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

# Silence harmless PyTorch internal compilation and Enum deprecation warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(os.environ.get("LLAMA_MODEL_DIR", ROOT / "llama_model"))

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Textile Defect: irregularity\nSeverity: MEDIUM\nConfidence: 0.95\n"
            "Recent Inspections: 1 defect in last 20 yarns (irregularity)"
        )
    },
    {
        "role": "assistant",
        "content": (
            "ACTION: Mark yarn section and adjust tension on winding spindle.\n"
            "CHECK: Verify traveler speed and package alignment."
        )
    },
    {
        "role": "user",
        "content": (
            "Textile Defect: knotting\nSeverity: HIGH\nConfidence: 0.99\n"
            "Recent Inspections: 5 defects in last 20 yarns (knotting)"
        )
    },
    {
        "role": "assistant",
        "content": (
            "ACTION: Pause winding spindle and check knotter alignment.\n"
            "CHECK: Inspect knotter knife and clean yarn path guides."
        )
    }
]


def build_prompt(tokenizer, label, severity, confidence, history_str=""):
    context_line = f"\nRecent Inspections: {history_str}" if history_str else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an automated quality control assistant in a textile spinning mill inspecting textile yarn bobbins. "
                "Provide machine operator instructions for textile yarn defects. Write exactly two lines:\n"
                "ACTION: <one short action>\n"
                "CHECK: <one short check>\n"
                "Be conservative. Output nothing else."
            )
        },
        *FEW_SHOT_EXAMPLES[:2],  # 1-shot (reduces prompt prefill latency by ~2s)
        {
            "role": "user",
            "content": f"Textile Defect: {label}\nSeverity: {severity}\nConfidence: {confidence:.2f}{context_line}"
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def validate_output(text, label="defect", severity="MEDIUM"):
    """
    Validates and cleans Llama generated output:
    - Enforces strict 2-line structure (ACTION: and CHECK:)
    - Falls back to deterministic operator guidance if syntax is broken
    """
    if not text:
        return False, f"ACTION: Inspect yarn for {label}.\nCHECK: Verify spindle tension and yarn path."

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    action_line = next((l for l in lines if l.upper().startswith("ACTION:")), None)
    check_line = next((l for l in lines if l.upper().startswith("CHECK:")), None)

    # Valid structured response
    if action_line and check_line and action_line != check_line:
        cleaned = f"{action_line}\n{check_line}"
        if len(cleaned) <= 250:
            return True, cleaned

    # Loose response normalization
    if len(lines) >= 2 and lines[0] != lines[1]:
        clean_a = lines[0] if lines[0].upper().startswith("ACTION:") else f"ACTION: {lines[0]}"
        clean_c = lines[1] if lines[1].upper().startswith("CHECK:") else f"CHECK: {lines[1]}"
        cleaned = f"{clean_a}\n{clean_c}"
        if len(cleaned) <= 250:
            return True, cleaned

    # Deterministic fallback
    if severity == "HIGH":
        fallback = f"ACTION: Stop spindle immediately for {label}.\nCHECK: Clean knotter and check thread path."
    else:
        fallback = f"ACTION: Mark yarn section for {label}.\nCHECK: Inspect traveler speed and bobbin tension."
    return False, fallback


class Advisor:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise SystemExit(
                f"[!] Error: Model directory not found: {model_dir}\n"
                "    Please ensure 'llama_model/' is copied to the deployment directory."
            )
        pte_file = model_dir / "model.pte"
        if not pte_file.exists():
            raise SystemExit(
                f"[!] Error: Quantized weights 'model.pte' not found in: {model_dir}\n"
                "    Please ensure 'model.pte' (~914 MB) is placed inside 'llama_model/'."
            )

        try:
            from optimum.executorch import ExecuTorchModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as err:
            raise SystemExit(
                f"[!] Error: Missing ExecuTorch / Transformers libraries ({err}).\n"
                "    Please activate your virtual environment: source rpi_env/bin/activate\n"
                "    Or run: ./setup_rpi5.sh"
            )

        # Informational messages sent to stderr to keep stdout strictly for JSON-RPC IPC
        print(f"Loading ExecuTorch Llama 3.2 1B from {model_dir.name}... (~6-12s on CPU)", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = ExecuTorchModelForCausalLM.from_pretrained(str(model_dir), export=False)
        print("[✓] Llama 3.2 1B resident in RAM and ready.", file=sys.stderr)

    def advise(self, label, severity, confidence, history_str=""):
        try:
            prompt = build_prompt(self.tokenizer, label, severity, confidence, history_str)
            input_tokens = self.tokenizer.encode(prompt)
            started = time.perf_counter()
            tokens = self.model.generate(input_tokens, echo=False, max_seq_len=len(input_tokens) + 48)
            elapsed = time.perf_counter() - started
            raw_text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            is_valid, valid_advice = validate_output(raw_text, label, severity)
            return valid_advice, len(tokens), elapsed
        except Exception as err:
            print(f"[!] Warning: Exception during Llama token generation: {err}", file=sys.stderr)
            _, fallback = validate_output("", label, severity)
            return fallback, 0, 0.0


def main():
    parser = argparse.ArgumentParser(description="On-Device ExecuTorch Llama 3.2 Advisory Engine")
    parser.add_argument("label", nargs="?", default=None,
                        help="Defect label (e.g. knotting, irregularity, thinness)")
    parser.add_argument("severity", nargs="?", default=None,
                        help="Defect severity (NONE, MEDIUM, HIGH)")
    parser.add_argument("confidence", nargs="?", type=float, default=None,
                        help="Confidence score 0.0-1.0 (e.g. 0.99)")
    parser.add_argument("--history", type=str, default="",
                        help="Recent defect history string")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Path to quantized llama_model directory")
    parser.add_argument("--serve", action="store_true",
                        help="Run in persistent JSON-RPC IPC daemon mode over stdin/stdout")
    args = parser.parse_args()

    # 1. Initialize Advisor
    advisor = Advisor(args.model_dir)

    # 2. Daemon Mode for background live inspection
    if args.serve:
        print(json.dumps({"status": "ready"}), flush=True)
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    hist = item.get("history", "")
                    advice, tokens, seconds = advisor.advise(
                        item["label"], item["severity"], float(item["confidence"]), hist
                    )
                    tok_s = round(tokens / seconds, 2) if seconds > 0 else 0.0
                    print(json.dumps({
                        "ok": True,
                        "advice": advice,
                        "tokens": tokens,
                        "seconds": seconds,
                        "tok_s": tok_s
                    }), flush=True)
                except Exception as error:
                    print(json.dumps({"ok": False, "error": str(error)}), flush=True)
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        return

    # 3. Interactive / Self-Test Mode
    if args.label is None or args.severity is None or args.confidence is None:
        print("\n[Self-Test] No defect arguments provided. Running self-test query for 'knotting' (HIGH, 0.98)...")
        label = "knotting"
        severity = "HIGH"
        confidence = 0.98
        history = "3 defects in last 20 yarns (knotting)"
    else:
        label = args.label
        severity = args.severity
        confidence = args.confidence
        history = args.history

    try:
        advice, tokens, seconds = advisor.advise(label, severity, confidence, history)
        tok_s = tokens / seconds if seconds > 0 else 0.0
        print("\n" + "=" * 55)
        print(f"Defect:     {label.upper()} ({confidence * 100:.1f}%) | Severity: {severity}")
        print("-" * 55)
        print(advice)
        print("-" * 55)
        print(f"Cortex-A76 Performance: {tokens} tokens generated in {seconds:.2f}s (~{tok_s:.1f} tok/s)")
        print("=" * 55 + "\n")
    except Exception as e:
        print(f"[!] Error during generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
