"""Compare the running server's outputs against golden/baseline.json.

    ./serve.sh baseline   && python make_golden.py
    ./serve.sh spec-draft && pytest -s tests
"""
import json
import os
from pathlib import Path

import pytest

from tests.common import complete_many, current_config, first_divergence, gsm8k_accuracy

GOLDEN = Path("golden/baseline.json")
MIN_EXACT = float(os.environ.get("MIN_EXACT_MATCH", "0.95"))
MAX_GSM8K_DROP = float(os.environ.get("MAX_GSM8K_DROP", "0.015"))


@pytest.fixture(scope="session")
def golden():
    if not GOLDEN.exists():
        pytest.skip("golden/baseline.json not found, run make_golden.py against the baseline server first")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="session")
def config():
    return current_config()


@pytest.fixture(scope="session")
def open_outputs(golden):
    return complete_many(golden["open_prompts"], max_tokens=256)


@pytest.fixture(scope="session")
def gsm_outputs(golden):
    return complete_many(golden["gsm8k_prompts"], max_tokens=384)


def compare(suite, config, expected, actual):
    """Exact-match rate plus where each mismatch starts. Writes results/regression_<suite>_<config>.json."""
    mismatches = []
    for i, (e, a) in enumerate(zip(expected, actual)):
        d = first_divergence(e, a)
        if d is not None:
            mismatches.append({"idx": i, "first_diff_char": d, "expected_len": len(e),
                               "relative_pos": round(d / max(len(e), 1), 3)})
    rate = 1 - len(mismatches) / len(actual)

    Path("results").mkdir(exist_ok=True)
    Path(f"results/regression_{suite}_{config}.json").write_text(json.dumps(
        {"config": config, "suite": suite, "exact_match": rate, "mismatches": mismatches}, indent=2))

    print(f"\n[{config}] {suite}: {rate:.1%} exact match, {len(mismatches)} mismatches")
    for m in mismatches[:10]:
        print(f"  #{m['idx']}: first difference at char {m['first_diff_char']} of "
              f"{m['expected_len']} ({m['relative_pos']:.0%} through)")
    return rate


def test_open_prompts_match_baseline(golden, config, open_outputs):
    rate = compare("open", config, golden["open_outputs"], open_outputs)
    assert rate >= MIN_EXACT, f"exact match {rate:.1%} is below {MIN_EXACT:.0%}"


def test_gsm8k_outputs_match_baseline(golden, config, gsm_outputs):
    rate = compare("gsm8k", config, golden["gsm8k_outputs"], gsm_outputs)
    assert rate >= MIN_EXACT, f"exact match {rate:.1%} is below {MIN_EXACT:.0%}"


def test_gsm8k_accuracy_not_regressed(golden, config, gsm_outputs):
    acc = gsm8k_accuracy(gsm_outputs, golden["gsm8k_answers"])
    base = golden["gsm8k_accuracy"]
    print(f"\n[{config}] GSM8K accuracy {acc:.3f}, baseline {base:.3f}")
    assert acc >= base - MAX_GSM8K_DROP, f"GSM8K accuracy dropped by {base - acc:.3f}"


def test_identical_prompts_identical_outputs(open_outputs):
    k = len(open_outputs) // 3
    a, b, c = open_outputs[:k], open_outputs[k:2 * k], open_outputs[2 * k:3 * k]
    rate = sum(x == y == z for x, y, z in zip(a, b, c)) / k
    print(f"\nidentical prompts with identical outputs: {rate:.1%}")
    assert rate >= MIN_EXACT, f"only {rate:.1%} of repeated prompts produced identical outputs"
