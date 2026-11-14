#!/usr/bin/env python3
"""ExecuTorch Llama advisory component for yarn-defect inspection."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(os.environ.get("LLAMA_MODEL_DIR", ROOT / "llama_model"))

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Defect: irregularity\nSeverity: MEDIUM\nConfidence: 0.95"},
    {"role": "assistant", "content": "ACTION: Mark section and adjust yarn tension on winding spindle.\nCHECK: Verify spinning traveler speed and package alignment."},
    {"role": "user", "content": "Defect: knotting\nSeverity: HIGH\nConfidence: 0.99"},
    {"role": "assistant", "content": "ACTION: Stop winding spindle immediately and cut out the knotted section.\nCHECK: Inspect knotter mechanism and yarn guide clearance."}
]


def build_prompt(tokenizer, label, severity, confidence):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an industrial yarn-inspection operator assistant. "
                "When given a machine classification, write exactly two short lines:\n"
                "ACTION: <one short sentence>\n"
                "CHECK: <one short sentence>\n"
                "Be conservative. Do not claim certainty. Do not output anything else."
            )
        },
        *FEW_SHOT_EXAMPLES,
        {
            "role": "user",
            "content": f"Defect: {label}\nSeverity: {severity}\nConfidence: {confidence:.2f}"
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

    def advise(self, label, severity, confidence):
        prompt = build_prompt(self.tokenizer, label, severity, confidence)
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
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if not args.model_dir.exists():
        raise SystemExit(f"Llama model directory not found: {args.model_dir}")
    advisor = Advisor(args.model_dir)
    if args.serve:
        for line in sys.stdin:
            try:
                item = json.loads(line)
                advice, tokens, seconds = advisor.advise(item["label"], item["severity"], float(item["confidence"]))
                tok_s = round(tokens / seconds, 2) if seconds > 0 else 0.0
                print(json.dumps({"ok": True, "advice": advice, "tokens": tokens, "seconds": seconds, "tok_s": tok_s}), flush=True)
            except Exception as error:
                print(json.dumps({"ok": False, "error": str(error)}), flush=True)
        return
    if None in (args.label, args.severity, args.confidence):
        parser.error("label, severity, and confidence are required unless --serve is used")
    try:
        advice, tokens, seconds = advisor.advise(args.label, args.severity, args.confidence)
        print(advice)
        print(f"[Llama: {tokens} generated tokens, {seconds:.2f}s, {tokens / seconds:.2f} tokens/s]")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
