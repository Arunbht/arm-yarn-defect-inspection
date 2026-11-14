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


def build_prompt(label, severity, confidence):
    return (
        "You are an industrial yarn-inspection operator assistant. Given a machine "
        "classification, write exactly two short lines: ACTION: immediate action; "
        "CHECK: one follow-up check. Be conservative. Do not claim certainty.\n"
        f"Defect: {label}\nSeverity: {severity}\nConfidence: {confidence:.2f}"
    )


class Advisor:
    def __init__(self, model_dir):
        from optimum.executorch import ExecuTorchModelForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = ExecuTorchModelForCausalLM.from_pretrained(model_dir, export=False)

    def advise(self, label, severity, confidence):
        prompt = build_prompt(label, severity, confidence)
        input_tokens = self.tokenizer.encode(prompt)
        started = time.perf_counter()
        tokens = self.model.generate(input_tokens, echo=False, max_seq_len=len(input_tokens) + 64)
        elapsed = time.perf_counter() - started
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()][:2]
        if len(lines) < 2:
            raise RuntimeError("Llama did not return two advisory lines")
        return "\n".join(lines), len(tokens), elapsed


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
                print(json.dumps({"ok": True, "advice": advice, "tokens": tokens, "seconds": seconds}), flush=True)
            except Exception as error:
                print(json.dumps({"ok": False, "error": str(error)}), flush=True)
        return
    if None in (args.label, args.severity, args.confidence):
        parser.error("label, severity, and confidence are required unless --serve is used")
    advice, tokens, seconds = advisor.advise(args.label, args.severity, args.confidence)
    print(advice)
    print(f"[Llama: {tokens} generated tokens, {seconds:.2f}s, {tokens / seconds:.2f} tokens/s]")


if __name__ == "__main__":
    main()
