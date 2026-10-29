"""Helpers shared by make_golden.py and the regression tests."""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

URL = os.environ.get("VLLM_URL", "http://localhost:8000")
NUM_GSM8K = int(os.environ.get("NUM_GSM8K", "200"))
CONCURRENCY = int(os.environ.get("REGRESSION_CONCURRENCY", "16"))

_BASE_PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Summarize the causes of World War I in three sentences.",
    "What is the difference between TCP and UDP?",
    "Give a recipe for a simple tomato sauce.",
    "Explain gradient descent to a high school student.",
    "Describe a city street on a rainy autumn evening.",
    "List five tips for writing clear technical documentation.",
    "What does the CAP theorem state?",
    "Translate 'Where is the train station?' into French, Spanish and German.",
    "Write a SQL query that returns the top 3 customers by total order value.",
    "Explain what a KV cache is in transformer inference.",
    "Describe how vaccines train the immune system.",
    "Write a bash one-liner that counts lines in all .py files in a directory.",
    "What are the trade-offs between microservices and a monolith?",
    "Explain the Monty Hall problem and why switching is better.",
    "Write a short professional email declining a meeting invitation.",
    "How does public-key cryptography work?",
    "Explain big-O notation with two examples.",
    "Write a JSON object describing a book with title, author, year and tags.",
]
# Each prompt appears three times so the tests can also check that identical
# requests in the same batch get identical greedy outputs.
OPEN_PROMPTS = _BASE_PROMPTS * 3

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=f"{URL}/v1", api_key="EMPTY", timeout=600)
    return _client


def model_id():
    return client().models.list().data[0].id


def current_config():
    p = Path("logs/current_config")
    return p.read_text().strip() if p.exists() else "unknown"


def complete(prompt, max_tokens, model):
    r = client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        seed=0,
    )
    return r.choices[0].message.content or ""


def complete_many(prompts, max_tokens):
    """Greedy completions, sent concurrently so the server batches them like real traffic."""
    model = model_id()
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        return list(ex.map(lambda p: complete(p, max_tokens, model), prompts))


def gsm8k_prompt(question):
    return (f"{question}\n\nSolve the problem step by step. "
            "Finish with a final line of the form '#### <number>'.")


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_number(text):
    """Final numeric answer: the number after the last '####', else the last number in the text."""
    if not text:
        return None
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    nums = _NUM.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def gsm8k_accuracy(outputs, answers):
    correct = 0
    for out, ans in zip(outputs, answers):
        pred, gold = extract_number(out), extract_number(ans)
        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            correct += 1
    return correct / len(answers)


def first_divergence(a, b):
    """Index of the first character where a and b differ, or None if they are equal."""
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))
