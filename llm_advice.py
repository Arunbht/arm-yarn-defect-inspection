#!/usr/bin/env python3
"""ExecuTorch Llama advisory component for yarn-defect inspection with trend context."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

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


def validate_output(text):
    """
    Validates output:
    - Must contain both 'ACTION:' and 'CHECK:'
    - Must be under ~250 characters
    - Must have no repeated lines
    Returns (is_valid, cleaned_advice)
    """
    if not text or len(text.strip()) > 250:
        return False, None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False, None
    # No repeated lines
    if len(lines) != len(set(lines)):
        return False, None
    action_line = next((l for l in lines if l.startswith("ACTION:")), None)
    check_line = next((l for l in lines if l.startswith("CHECK:")), None)
    if not action_line or not check_line:
        return False, None
    cleaned = f"{action_line}\n{check_line}"
    if len(cleaned) > 250:
        return False, None
    return True, cleaned


class Advisor:
    def __init__(self, model_dir):
        from optimum.executorch import ExecuTorchModelForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = ExecuTorchModelForCausalLM.from_pretrained(model_dir, export=False)

    def advise(self, label, severity, confidence, history_str=""):
        prompt = build_prompt(self.tokenizer, label, severity, confidence, history_str)
        input_tokens = self.tokenizer.encode(prompt)
        started = time.perf_counter()
        tokens = self.model.generate(input_tokens, echo=False, max_seq_len=len(input_tokens) + 48)
        elapsed = time.perf_counter() - started
        raw_text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        is_valid, valid_advice = validate_output(raw_text)
        if not is_valid:
            raise ValueError(f"Output failed validation (len={len(raw_text)}): {raw_text!r}")
        return valid_advice, len(tokens), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?")
    parser.add_argument("severity", nargs="?")
    parser.add_argument("confidence", nargs="?", type=float)
    parser.add_argument("--history", type=str, default="")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if not args.model_dir.exists():
        raise SystemExit(f"Llama model directory not found: {args.model_dir}")
    advisor = Advisor(args.model_dir)
    if args.serve:
        print(json.dumps({"status": "ready"}), flush=True)
        for line in sys.stdin:
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
        return
    if None in (args.label, args.severity, args.confidence):
        parser.error("label, severity, and confidence are required unless --serve is used")
    try:
        advice, tokens, seconds = advisor.advise(args.label, args.severity, args.confidence, args.history)
        print(advice)
        print(f"[Llama: {tokens} generated tokens, {seconds:.2f}s, {tokens / seconds:.2f} tokens/s]")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
