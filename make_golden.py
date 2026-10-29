"""Record reference outputs from the baseline server for the regression tests.

    ./serve.sh baseline && python make_golden.py
"""
import json
from pathlib import Path

from datasets import load_dataset

from tests.common import (NUM_GSM8K, OPEN_PROMPTS, complete_many, current_config,
                          gsm8k_accuracy, gsm8k_prompt)


def main():
    config = current_config()
    if config != "baseline":
        raise SystemExit(f"Golden outputs should come from the baseline config, server is running '{config}'.")

    gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(NUM_GSM8K))
    gsm_prompts = [gsm8k_prompt(q) for q in gsm["question"]]

    open_out = complete_many(OPEN_PROMPTS, max_tokens=256)
    gsm_out = complete_many(gsm_prompts, max_tokens=384)

    golden = {
        "config": config,
        "open_prompts": OPEN_PROMPTS,
        "open_outputs": open_out,
        "gsm8k_prompts": gsm_prompts,
        "gsm8k_answers": list(gsm["answer"]),
        "gsm8k_outputs": gsm_out,
        "gsm8k_accuracy": gsm8k_accuracy(gsm_out, gsm["answer"]),
    }
    Path("golden").mkdir(exist_ok=True)
    Path("golden/baseline.json").write_text(json.dumps(golden, indent=2))
    print(f"Saved golden/baseline.json (GSM8K accuracy {golden['gsm8k_accuracy']:.3f})")


if __name__ == "__main__":
    main()
