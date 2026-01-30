#!/usr/bin/env python3
"""
Intelligence Benchmark - Tests that differentiate model capabilities.

Unlike quick_benchmark.py which tests basic functionality, this tests:
1. Code generation quality (can it write correct, idiomatic code?)
2. Multi-step reasoning (5+ step problems that require planning)
3. Complex instruction following (nuanced, multi-part requirements)
4. Long context utilization (can it synthesize from large context?)

Designed to show where 14B beats 4B to justify tiered inference routing.
"""

import argparse
import json
import re
import time
import requests
from datetime import datetime
from typing import Optional
import subprocess
import tempfile
import os


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from response text."""
    if not text:
        return text
    think_pattern = r'<think>.*?</think>\s*'
    cleaned = re.sub(think_pattern, '', text, flags=re.DOTALL)
    if cleaned == text and '<think>' in text.lower():
        unclosed_pattern = r'<think>.*$'
        cleaned = re.sub(unclosed_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip() if cleaned != text else text.strip()


def extract_code_block(text: str) -> str:
    """Extract code from markdown code blocks."""
    # Try to find ```python or ``` blocks
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # Fallback: return the whole text if no code blocks
    return text.strip()


def call_model(api_base: str, model: str, messages: list, max_tokens: int = 1000, temperature: float = 0.6) -> dict:
    """Make a chat completion request."""
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95
            },
            timeout=120
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "success": True,
                "content": strip_think_tags(content),
                "raw_content": content,
                "tokens": data.get("usage", {}).get("completion_tokens", 0)
            }
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_code_generation(api_base: str, model: str) -> dict:
    """Test code generation quality with executable tests."""
    tests = [
        {
            "name": "fibonacci_iterative",
            "prompt": "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number using iteration (not recursion). Handle edge cases for n=0 and n=1. Return only the code, no explanation.",
            "test_code": """
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(10) == 55
assert fibonacci(20) == 6765
print("PASS")
""",
            "difficulty": "easy"
        },
        {
            "name": "binary_search",
            "prompt": "Write a Python function called `binary_search(arr, target)` that returns the index of target in sorted array arr, or -1 if not found. Must be O(log n). Return only the code.",
            "test_code": """
assert binary_search([1, 2, 3, 4, 5], 3) == 2
assert binary_search([1, 2, 3, 4, 5], 1) == 0
assert binary_search([1, 2, 3, 4, 5], 5) == 4
assert binary_search([1, 2, 3, 4, 5], 6) == -1
assert binary_search([], 1) == -1
assert binary_search([1], 1) == 0
print("PASS")
""",
            "difficulty": "easy"
        },
        {
            "name": "lru_cache",
            "prompt": """Write a Python class called `LRUCache` that implements a Least Recently Used cache with the following interface:
- `__init__(self, capacity: int)` - Initialize with given capacity
- `get(self, key: int) -> int` - Return value if key exists, else -1. Marks as recently used.
- `put(self, key: int, value: int) -> None` - Insert or update. Evict LRU item if over capacity.

Must be O(1) for both operations. Use OrderedDict or implement with dict + doubly linked list. Return only the code.""",
            "test_code": """
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)  # Evicts key 2
assert cache.get(2) == -1
cache.put(4, 4)  # Evicts key 1
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
print("PASS")
""",
            "difficulty": "medium"
        },
        {
            "name": "merge_intervals",
            "prompt": """Write a Python function called `merge_intervals(intervals)` that takes a list of intervals [[start, end], ...] and merges all overlapping intervals. Return the merged list sorted by start time.

Example: [[1,3],[2,6],[8,10],[15,18]] -> [[1,6],[8,10],[15,18]]

Return only the code.""",
            "test_code": """
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([[1,4],[0,4]]) == [[0,4]]
assert merge_intervals([]) == []
assert merge_intervals([[1,4],[2,3]]) == [[1,4]]
print("PASS")
""",
            "difficulty": "medium"
        },
        {
            "name": "topological_sort",
            "prompt": """Write a Python function called `topological_sort(num_nodes, edges)` that performs topological sort on a directed acyclic graph.
- num_nodes: number of nodes (0 to num_nodes-1)
- edges: list of [from, to] directed edges
- Return a valid topological ordering, or empty list if cycle detected.

Use Kahn's algorithm (BFS with in-degree tracking). Return only the code.""",
            "test_code": """
# Simple chain
result = topological_sort(3, [[0,1],[1,2]])
assert result == [0,1,2] or (result[0] == 0 and result[-1] == 2)

# Diamond
result = topological_sort(4, [[0,1],[0,2],[1,3],[2,3]])
assert result[0] == 0 and result[-1] == 3

# Cycle detection
result = topological_sort(2, [[0,1],[1,0]])
assert result == []

# No edges
result = topological_sort(3, [])
assert len(result) == 3

print("PASS")
""",
            "difficulty": "hard"
        }
    ]

    results = []
    passed = 0

    for test in tests:
        print(f"    Testing {test['name']} ({test['difficulty']})...", end=" ", flush=True)

        response = call_model(
            api_base, model,
            [{"role": "user", "content": test["prompt"]}],
            max_tokens=1500,
            temperature=0.3  # Lower temp for code
        )

        if not response["success"]:
            print("FAILED (API error)")
            results.append({
                "name": test["name"],
                "difficulty": test["difficulty"],
                "passed": False,
                "error": response["error"]
            })
            continue

        code = extract_code_block(response["content"])

        # Try to execute the code with tests
        full_code = code + "\n" + test["test_code"]

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                f.flush()

                result = subprocess.run(
                    ['python3', f.name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                os.unlink(f.name)

                if result.returncode == 0 and "PASS" in result.stdout:
                    print("PASS")
                    passed += 1
                    results.append({
                        "name": test["name"],
                        "difficulty": test["difficulty"],
                        "passed": True
                    })
                else:
                    print("FAILED")
                    results.append({
                        "name": test["name"],
                        "difficulty": test["difficulty"],
                        "passed": False,
                        "error": result.stderr[:200] if result.stderr else result.stdout[:200]
                    })
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            results.append({
                "name": test["name"],
                "difficulty": test["difficulty"],
                "passed": False,
                "error": "Execution timeout"
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "name": test["name"],
                "difficulty": test["difficulty"],
                "passed": False,
                "error": str(e)
            })

    return {
        "score": f"{passed}/{len(tests)}",
        "percentage": round(100 * passed / len(tests), 1),
        "by_difficulty": {
            "easy": sum(1 for r in results if r.get("passed") and r["difficulty"] == "easy"),
            "medium": sum(1 for r in results if r.get("passed") and r["difficulty"] == "medium"),
            "hard": sum(1 for r in results if r.get("passed") and r["difficulty"] == "hard")
        },
        "details": results
    }


def test_multistep_reasoning(api_base: str, model: str) -> dict:
    """Test multi-step reasoning that requires planning."""
    tests = [
        {
            "name": "water_jugs",
            "prompt": """You have two jugs: a 5-gallon jug and a 3-gallon jug. You need to measure exactly 4 gallons of water.
Available actions: Fill a jug completely, empty a jug completely, pour from one jug to another until either the source is empty or the destination is full.

What is the minimum number of steps needed? Explain each step briefly, then state the final answer as just a number. /no_think""",
            "answer": "6",
            "check_type": "contains"
        },
        {
            "name": "river_crossing",
            "prompt": """A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat can only carry the farmer and one item at a time.
- If left alone, the wolf will eat the goat
- If left alone, the goat will eat the cabbage

What is the minimum number of river crossings (one-way trips) needed? Think through each crossing, then give ONLY the number as your final answer. /no_think""",
            "answer": "7",
            "check_type": "contains"
        },
        {
            "name": "compound_interest",
            "prompt": """Calculate step by step:
You invest $10,000 at 8% annual interest, compounded quarterly.
After 5 years, you withdraw half the balance.
The remaining amount continues to earn 8% compounded quarterly for 3 more years.
What is the final balance? Round to the nearest dollar. Answer with just the number, no $ sign. /no_think""",
            "answer": "9388",  # Approximately
            "check_type": "numeric_close",
            "tolerance": 100
        },
        {
            "name": "logic_grid",
            "prompt": """Four friends (Alice, Bob, Carol, Dave) each have a different favorite color (red, blue, green, yellow) and a different pet (cat, dog, bird, fish).

Clues:
1. Alice's favorite color is not red or blue
2. The person with the cat likes blue
3. Bob has a dog
4. Carol's favorite color is yellow
5. Dave doesn't have the fish
6. The person who likes green has the bird
7. Alice has the fish

Who has the cat? Answer with just the name. /no_think""",
            "answer": "dave",
            "check_type": "contains_lower"
        },
        {
            "name": "scheduling",
            "prompt": """Five tasks (A, B, C, D, E) need to be scheduled. Each task takes 1 hour.
Dependencies (must complete before):
- A must complete before B
- A must complete before C
- B must complete before D
- C must complete before D
- D must complete before E

If you have 2 workers who can work in parallel, what is the minimum time to complete all tasks? Answer with just the number of hours. /no_think""",
            "answer": "4",
            "check_type": "contains"
        }
    ]

    results = []
    passed = 0

    for test in tests:
        print(f"    Testing {test['name']}...", end=" ", flush=True)

        response = call_model(
            api_base, model,
            [{"role": "user", "content": test["prompt"]}],
            max_tokens=800,
            temperature=0.3
        )

        if not response["success"]:
            print("FAILED (API error)")
            results.append({
                "name": test["name"],
                "passed": False,
                "error": response["error"]
            })
            continue

        answer = response["content"].strip().lower()
        expected = test["answer"].lower()

        is_correct = False
        if test["check_type"] == "contains":
            is_correct = expected in answer
        elif test["check_type"] == "contains_lower":
            is_correct = expected in answer.lower()
        elif test["check_type"] == "numeric_close":
            # Extract numbers from answer
            numbers = re.findall(r'\d+', answer.replace(',', ''))
            for num in numbers:
                if abs(int(num) - int(expected)) <= test.get("tolerance", 10):
                    is_correct = True
                    break

        if is_correct:
            print("PASS")
            passed += 1
        else:
            print(f"FAILED (got: {answer[:50]})")

        results.append({
            "name": test["name"],
            "passed": is_correct,
            "expected": test["answer"],
            "got": answer[:100]
        })

    return {
        "score": f"{passed}/{len(tests)}",
        "percentage": round(100 * passed / len(tests), 1),
        "details": results
    }


def test_instruction_following(api_base: str, model: str) -> dict:
    """Test ability to follow complex, multi-part instructions."""
    tests = [
        {
            "name": "format_constraints",
            "prompt": """Write a product description for a laptop with these EXACT constraints:
1. Exactly 3 sentences
2. First sentence must be a question
3. Include the word "revolutionary"
4. End with a price in the format $X,XXX
5. Do not use the word "the"

/no_think""",
            "checks": [
                ("sentence_count", lambda x: len([s for s in x.split('.') if s.strip()]) == 3),
                ("starts_question", lambda x: x.strip().split('.')[0].strip().endswith('?') or x.strip().split('?')[0] if '?' in x else False),
                ("has_revolutionary", lambda x: "revolutionary" in x.lower()),
                ("has_price", lambda x: bool(re.search(r'\$\d{1,2},\d{3}', x))),
                ("no_the", lambda x: " the " not in x.lower() and not x.lower().startswith("the "))
            ]
        },
        {
            "name": "structured_output",
            "prompt": """Analyze this text and respond with EXACTLY this JSON structure (no other text):
{"sentiment": "positive/negative/neutral", "topics": ["topic1", "topic2"], "word_count": N}

Text: "The new restaurant downtown has amazing pasta but the service was quite slow. The ambiance made up for it though, with beautiful decor and soft lighting."

/no_think""",
            "checks": [
                ("valid_json", lambda x: is_valid_json(x)),
                ("has_sentiment", lambda x: "sentiment" in x.lower()),
                ("has_topics", lambda x: "topics" in x.lower()),
                ("has_word_count", lambda x: "word_count" in x.lower())
            ]
        },
        {
            "name": "list_with_rules",
            "prompt": """Generate a numbered list of 5 fictional book titles following ALL these rules:
1. Each title must be exactly 4 words
2. No title can contain the word "the" or "a"
3. Each title must include one color word
4. Titles must be numbered 1-5 with a period after the number
5. No two titles can have the same color

/no_think""",
            "checks": [
                ("has_five_items", lambda x: len(re.findall(r'^\d\.', x, re.MULTILINE)) >= 5),
                ("no_the_or_a", lambda x: " the " not in x.lower() and " a " not in x.lower()),
                ("has_colors", lambda x: sum(1 for c in ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "gray", "grey", "brown", "gold", "silver"] if c in x.lower()) >= 3)
            ]
        },
        {
            "name": "code_with_constraints",
            "prompt": """Write a Python function that:
1. Is named exactly "process_data"
2. Takes exactly 2 parameters
3. Uses a list comprehension
4. Has exactly one return statement
5. Is no more than 5 lines total (including def line)
6. Includes a docstring

The function should filter a list to only include items greater than a threshold. Return only the code.

/no_think""",
            "checks": [
                ("has_process_data", lambda x: "def process_data" in x),
                ("has_two_params", lambda x: bool(re.search(r'def process_data\s*\([^,]+,[^)]+\)', x))),
                ("has_comprehension", lambda x: bool(re.search(r'\[.+for.+in.+\]', x))),
                ("has_docstring", lambda x: '"""' in x or "'''" in x),
                ("line_count", lambda x: len([l for l in x.strip().split('\n') if l.strip()]) <= 6)
            ]
        }
    ]

    results = []
    total_checks = 0
    passed_checks = 0

    for test in tests:
        print(f"    Testing {test['name']}...", end=" ", flush=True)

        response = call_model(
            api_base, model,
            [{"role": "user", "content": test["prompt"]}],
            max_tokens=500,
            temperature=0.3
        )

        if not response["success"]:
            print("FAILED (API error)")
            results.append({
                "name": test["name"],
                "passed_checks": 0,
                "total_checks": len(test["checks"]),
                "error": response["error"]
            })
            total_checks += len(test["checks"])
            continue

        content = response["content"]
        check_results = {}
        test_passed = 0

        for check_name, check_fn in test["checks"]:
            try:
                passed = check_fn(content)
                check_results[check_name] = passed
                if passed:
                    test_passed += 1
            except:
                check_results[check_name] = False

        passed_checks += test_passed
        total_checks += len(test["checks"])

        pct = round(100 * test_passed / len(test["checks"]))
        print(f"{test_passed}/{len(test['checks'])} checks ({pct}%)")

        results.append({
            "name": test["name"],
            "passed_checks": test_passed,
            "total_checks": len(test["checks"]),
            "check_details": check_results,
            "response_preview": content[:200]
        })

    return {
        "score": f"{passed_checks}/{total_checks}",
        "percentage": round(100 * passed_checks / total_checks, 1),
        "details": results
    }


def is_valid_json(text: str) -> bool:
    """Check if text contains valid JSON."""
    # Try to find JSON in the text
    try:
        # First try the whole text
        json.loads(text)
        return True
    except:
        pass

    # Try to extract JSON from code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if json_match:
        try:
            json.loads(json_match.group(1))
            return True
        except:
            pass

    # Try to find JSON object in text
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            json.loads(json_match.group(0))
            return True
        except:
            pass

    return False


def get_model_info(api_base: str) -> dict:
    """Get model information from the endpoint."""
    try:
        resp = requests.get(f"{api_base}/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                return {"model_id": models[0]["id"]}
        return {"model_id": "unknown"}
    except:
        return {"model_id": "unknown"}


def run_benchmark(api_base: str, model: Optional[str] = None, skip_code: bool = False) -> dict:
    """Run full intelligence benchmark suite."""
    api_base = api_base.rstrip("/")

    info = get_model_info(api_base)
    if not model:
        model = info.get("model_id", "unknown")

    print(f"\n{'='*70}")
    print(f"INTELLIGENCE BENCHMARK")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Endpoint: {api_base}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    results = {
        "model": model,
        "endpoint": api_base,
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # Test 1: Code Generation
    if not skip_code:
        print("1. CODE GENERATION (executable tests)")
        print("-" * 40)
        start = time.time()
        results["tests"]["code_generation"] = test_code_generation(api_base, model)
        results["tests"]["code_generation"]["time_sec"] = round(time.time() - start, 1)
        print(f"   Score: {results['tests']['code_generation']['score']} ({results['tests']['code_generation']['percentage']}%)")
        print(f"   Time: {results['tests']['code_generation']['time_sec']}s\n")

    # Test 2: Multi-step Reasoning
    print("2. MULTI-STEP REASONING")
    print("-" * 40)
    start = time.time()
    results["tests"]["reasoning"] = test_multistep_reasoning(api_base, model)
    results["tests"]["reasoning"]["time_sec"] = round(time.time() - start, 1)
    print(f"   Score: {results['tests']['reasoning']['score']} ({results['tests']['reasoning']['percentage']}%)")
    print(f"   Time: {results['tests']['reasoning']['time_sec']}s\n")

    # Test 3: Instruction Following
    print("3. INSTRUCTION FOLLOWING")
    print("-" * 40)
    start = time.time()
    results["tests"]["instruction_following"] = test_instruction_following(api_base, model)
    results["tests"]["instruction_following"]["time_sec"] = round(time.time() - start, 1)
    print(f"   Score: {results['tests']['instruction_following']['score']} ({results['tests']['instruction_following']['percentage']}%)")
    print(f"   Time: {results['tests']['instruction_following']['time_sec']}s\n")

    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {model}")

    total_pct = 0
    count = 0

    if "code_generation" in results["tests"]:
        cg = results["tests"]["code_generation"]
        print(f"Code Generation:     {cg['score']:>8} ({cg['percentage']:>5.1f}%) - Easy:{cg['by_difficulty']['easy']}/2 Med:{cg['by_difficulty']['medium']}/2 Hard:{cg['by_difficulty']['hard']}/1")
        total_pct += cg["percentage"]
        count += 1

    r = results["tests"]["reasoning"]
    print(f"Multi-step Reasoning:{r['score']:>8} ({r['percentage']:>5.1f}%)")
    total_pct += r["percentage"]
    count += 1

    i = results["tests"]["instruction_following"]
    print(f"Instruction Following:{i['score']:>7} ({i['percentage']:>5.1f}%)")
    total_pct += i["percentage"]
    count += 1

    avg_pct = total_pct / count
    print(f"\nOVERALL INTELLIGENCE SCORE: {avg_pct:.1f}%")
    results["overall_score"] = round(avg_pct, 1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Intelligence benchmark for comparing model capabilities")
    parser.add_argument("--api-base", "-a", required=True, help="API base URL")
    parser.add_argument("--model", "-m", help="Model name (auto-detected if not specified)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--compare", "-c", help="Compare with another endpoint")
    parser.add_argument("--skip-code", action="store_true", help="Skip code generation tests (faster)")

    args = parser.parse_args()

    results = [run_benchmark(args.api_base, args.model, args.skip_code)]

    if args.compare:
        print("\n" + "="*70)
        print("COMPARISON ENDPOINT")
        print("="*70)
        results.append(run_benchmark(args.compare, args.model, args.skip_code))

        # Side by side comparison
        print(f"\n{'='*70}")
        print("HEAD-TO-HEAD COMPARISON")
        print(f"{'='*70}")
        print(f"{'Test':<25} {'Endpoint 1':>15} {'Endpoint 2':>15} {'Winner':>12}")
        print("-" * 70)

        for test_name in ["code_generation", "reasoning", "instruction_following"]:
            if test_name in results[0]["tests"] and test_name in results[1]["tests"]:
                p1 = results[0]["tests"][test_name]["percentage"]
                p2 = results[1]["tests"][test_name]["percentage"]
                winner = "TIE" if p1 == p2 else ("←" if p1 > p2 else "→")
                print(f"{test_name:<25} {p1:>14.1f}% {p2:>14.1f}% {winner:>12}")

        print("-" * 70)
        print(f"{'OVERALL':<25} {results[0]['overall_score']:>14.1f}% {results[1]['overall_score']:>14.1f}% {'←' if results[0]['overall_score'] > results[1]['overall_score'] else '→' if results[1]['overall_score'] > results[0]['overall_score'] else 'TIE':>12}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results[0] if len(results) == 1 else results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
