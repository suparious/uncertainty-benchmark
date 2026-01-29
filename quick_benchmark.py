#!/usr/bin/env python3
"""
Quick Model Benchmark - Fast comparison script for A/B testing vLLM deployments.
Tests: inference speed, basic reasoning, tool calling format compliance.

Updated for 2026 LLMs with support for:
- Reasoning/thinking tokens (<think>...</think> tags)
- Tool calling diagnostics
- Latency percentiles
"""

import argparse
import json
import re
import time
import requests
from datetime import datetime
from typing import Optional


def strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks from response text.

    Modern reasoning models (Qwen3, DeepSeek-R1) output thinking process
    in <think> tags before the final answer.

    Returns the text after the closing </think> tag, or the original text
    if no think tags are present.
    """
    if not text:
        return text

    # First try to match complete think blocks
    think_pattern = r'<think>.*?</think>\s*'
    cleaned = re.sub(think_pattern, '', text, flags=re.DOTALL)

    # If that didn't change anything, check for unclosed <think> tag
    # (happens when response is truncated at max_tokens)
    if cleaned == text and '<think>' in text.lower():
        # Remove everything from <think> to end of string
        unclosed_pattern = r'<think>.*$'
        cleaned = re.sub(unclosed_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # If we removed think tags, return the cleaned text
    # Otherwise return original (model doesn't use think tags)
    return cleaned.strip() if cleaned != text else text.strip()


def extract_final_answer(text: str) -> tuple[str, bool]:
    """
    Extract the final answer from a response, handling think tags.

    Returns:
        tuple: (answer_text, had_think_tags)
    """
    if not text:
        return text, False

    had_think_tags = '<think>' in text.lower() or '</think>' in text.lower()
    answer = strip_think_tags(text)

    return answer, had_think_tags


def percentile(data: list, p: float) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f]) if f != c else sorted_data[f]


def test_inference_speed(api_base: str, model: str, n_samples: int = 10) -> dict:
    """Measure tokens/second for generation with latency percentiles."""
    prompt = "Write a haiku about artificial intelligence."

    times = []
    output_tokens = []
    time_to_first_token = []

    for _ in range(n_samples):
        start = time.time()
        resp = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=120
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            times.append(elapsed)
            output_tokens.append(tokens)

            # Check for time_to_first_token in response (vLLM provides this)
            ttft = data.get("timings", {}).get("time_to_first_token")
            if ttft:
                time_to_first_token.append(ttft)

    if not times:
        return {"error": "All requests failed"}

    avg_time = sum(times) / len(times)
    avg_tokens = sum(output_tokens) / len(output_tokens)

    result = {
        "avg_latency_sec": round(avg_time, 3),
        "avg_tokens": round(avg_tokens, 1),
        "tokens_per_sec": round(avg_tokens / avg_time, 2) if avg_time > 0 else 0,
        "samples": len(times),
        "latency_p50_sec": round(percentile(times, 50), 3),
        "latency_p90_sec": round(percentile(times, 90), 3),
        "latency_p99_sec": round(percentile(times, 99), 3),
    }

    # Add TTFT if available
    if time_to_first_token:
        result["ttft_avg_sec"] = round(sum(time_to_first_token) / len(time_to_first_token), 3)
        result["ttft_p50_sec"] = round(percentile(time_to_first_token, 50), 3)
        result["ttft_p90_sec"] = round(percentile(time_to_first_token, 90), 3)

    return result


def test_reasoning(api_base: str, model: str) -> dict:
    """Test basic reasoning with a few simple questions.

    Handles models that use <think>...</think> tags for reasoning by
    extracting the final answer after the thinking block.
    """
    questions = [
        {
            "q": "If I have 3 apples and buy 5 more, then give away 2, how many do I have?",
            "a": "6"
        },
        {
            "q": "What comes next in this sequence: 2, 4, 8, 16, ?",
            "a": "32"
        },
        {
            "q": "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "a": "0.05"  # Common trick question
        },
        {
            "q": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "a": "no"  # Logic test
        },
        {
            "q": "What is the capital of France?",
            "a": "paris"
        }
    ]

    correct = 0
    results = []
    uses_think_tags = False

    for item in questions:
        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Answer concisely with just the answer value. No explanation needed."},
                        {"role": "user", "content": item["q"] + " /no_think"}  # Disable thinking for concise answers
                    ],
                    "max_tokens": 100,  # Concise answers don't need much
                    "temperature": 0
                },
                timeout=60
            )

            if resp.status_code == 200:
                raw_answer = resp.json()["choices"][0]["message"]["content"]
                answer, had_think = extract_final_answer(raw_answer)
                if had_think:
                    uses_think_tags = True

                answer_lower = answer.strip().lower()
                is_correct = item["a"].lower() in answer_lower
                if is_correct:
                    correct += 1
                results.append({
                    "q": item["q"][:50],
                    "expected": item["a"],
                    "got": answer[:50],
                    "correct": is_correct,
                    "had_think_tags": had_think
                })
        except Exception as e:
            results.append({"q": item["q"][:50], "error": str(e)})

    return {
        "score": f"{correct}/{len(questions)}",
        "percentage": round(100 * correct / len(questions), 1),
        "uses_think_tags": uses_think_tags,
        "details": results
    }


def test_knowledge(api_base: str, model: str) -> dict:
    """Test factual knowledge with MMLU-style questions.

    Handles models that use <think>...</think> tags for reasoning.
    """
    questions = [
        {
            "q": "What is the capital of Japan?",
            "a": "tokyo"
        },
        {
            "q": "What is the chemical symbol for gold?",
            "a": "au"
        },
        {
            "q": "In what year did World War II end?",
            "a": "1945"
        },
        {
            "q": "What is the largest planet in our solar system?",
            "a": "jupiter"
        },
        {
            "q": "What is the speed of light in a vacuum, approximately in km/s?",
            "a": "300000"  # Accepts 300,000 or ~300000
        }
    ]

    correct = 0
    results = []
    uses_think_tags = False

    for item in questions:
        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Answer with just the answer value. Be brief."},
                        {"role": "user", "content": item["q"] + " /no_think"}  # Disable thinking for concise answers
                    ],
                    "max_tokens": 100,  # Concise answers don't need much
                    "temperature": 0
                },
                timeout=60
            )

            if resp.status_code == 200:
                raw_answer = resp.json()["choices"][0]["message"]["content"]
                answer, had_think = extract_final_answer(raw_answer)
                if had_think:
                    uses_think_tags = True

                answer_lower = answer.strip().lower().replace(",", "")
                expected_lower = item["a"].lower()

                # For numeric answers, be more flexible
                is_correct = expected_lower in answer_lower
                if is_correct:
                    correct += 1
                results.append({
                    "q": item["q"][:50],
                    "expected": item["a"],
                    "got": answer[:50],
                    "correct": is_correct,
                    "had_think_tags": had_think
                })
        except Exception as e:
            results.append({"q": item["q"][:50], "error": str(e)})

    return {
        "score": f"{correct}/{len(questions)}",
        "percentage": round(100 * correct / len(questions), 1),
        "uses_think_tags": uses_think_tags,
        "details": results
    }


def test_tool_calling(api_base: str, model: str) -> dict:
    """Test if model can generate valid tool call format.

    Tests both 'auto' tool_choice and explicit function selection.
    Provides diagnostics for common failure modes.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    result = {
        "supported": False,
        "auto_mode": None,
        "forced_mode": None,
        "diagnostics": []
    }

    # Test 1: Auto tool choice
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 100,
                "temperature": 0.6,  # Model card: DON'T use temp=0 with thinking mode
                "top_p": 0.95
            },
            timeout=60
        )

        if resp.status_code == 400:
            error_detail = resp.text[:500]
            result["diagnostics"].append(f"400 Error with tool_choice=auto: {error_detail}")

            # Check for common vLLM issues
            if "enable_auto_tool_choice" in error_detail.lower():
                result["diagnostics"].append("HINT: vLLM requires --enable-auto-tool-choice flag")
            if "chat_template" in error_detail.lower() or "tool" in error_detail.lower():
                result["diagnostics"].append("HINT: Model may need a tool-capable chat template")

        elif resp.status_code == 200:
            data = resp.json()
            message = data["choices"][0]["message"]

            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                    result["auto_mode"] = {
                        "success": True,
                        "tool_name": tool_call["function"]["name"],
                        "arguments": args
                    }
                    result["supported"] = True
                except json.JSONDecodeError as e:
                    result["auto_mode"] = {"success": False, "error": f"Invalid JSON: {e}"}
            else:
                result["auto_mode"] = {
                    "success": False,
                    "reason": "Model responded with text instead of tool call",
                    "response": message.get("content", "")[:100]
                }
        else:
            result["diagnostics"].append(f"HTTP {resp.status_code}: {resp.text[:200]}")

    except Exception as e:
        result["diagnostics"].append(f"Auto mode exception: {str(e)}")

    # Test 2: Forced tool choice
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "max_tokens": 100,
                "temperature": 0.6,  # Model card: DON'T use temp=0 with thinking mode
                "top_p": 0.95
            },
            timeout=60
        )

        if resp.status_code == 400:
            error_detail = resp.text[:500]
            result["diagnostics"].append(f"400 Error with forced tool_choice: {error_detail}")

        elif resp.status_code == 200:
            data = resp.json()
            message = data["choices"][0]["message"]

            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                    result["forced_mode"] = {
                        "success": True,
                        "tool_name": tool_call["function"]["name"],
                        "arguments": args
                    }
                    result["supported"] = True
                except json.JSONDecodeError as e:
                    result["forced_mode"] = {"success": False, "error": f"Invalid JSON: {e}"}
            else:
                result["forced_mode"] = {
                    "success": False,
                    "reason": "Model did not produce tool call even when forced",
                    "response": message.get("content", "")[:100]
                }
        else:
            result["diagnostics"].append(f"Forced mode HTTP {resp.status_code}")

    except Exception as e:
        result["diagnostics"].append(f"Forced mode exception: {str(e)}")

    # Summary diagnostics
    if not result["supported"]:
        if not result["diagnostics"]:
            result["diagnostics"].append("Tool calling not supported or not properly configured")

    return result


def test_structured_output(api_base: str, model: str) -> dict:
    """Test JSON mode / structured output."""
    content = None
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
                    {"role": "user", "content": "List 3 programming languages with their main use case. Return as JSON array."}
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 200,
                "temperature": 0.6,  # Model card: DON'T use temp=0 with thinking mode
                "top_p": 0.95
            },
            timeout=60
        )

        if resp.status_code != 200:
            # Try without response_format (not all models support it)
            resp = requests.post(
                f"{api_base}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You must respond with valid JSON only, no other text."},
                        {"role": "user", "content": "List 3 programming languages with their main use case. Return as JSON array."}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.6,  # Model card: DON'T use temp=0 with thinking mode
                    "top_p": 0.95
                },
                timeout=60
            )

        if resp.status_code == 200:
            raw_content = resp.json()["choices"][0]["message"]["content"]
            # Strip think tags before parsing JSON
            content, _ = extract_final_answer(raw_content)
            # Try to parse as JSON
            parsed = json.loads(content)
            return {
                "json_mode_supported": True,
                "valid_json": True,
                "parsed": parsed if len(str(parsed)) < 200 else "...truncated..."
            }
    except json.JSONDecodeError:
        return {"json_mode_supported": False, "valid_json": False, "raw": content[:200] if content else "N/A"}
    except Exception as e:
        return {"json_mode_supported": False, "error": str(e)}


def get_model_info(api_base: str) -> dict:
    """Get model information from the endpoint."""
    try:
        resp = requests.get(f"{api_base}/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                return {"model_id": models[0]["id"], "available_models": [m["id"] for m in models]}
        return {"error": "Could not fetch models"}
    except Exception as e:
        return {"error": str(e)}


def run_benchmark(api_base: str, model: Optional[str] = None, verbose: bool = False) -> dict:
    """Run full benchmark suite."""
    api_base = api_base.rstrip("/")

    # Get model info
    info = get_model_info(api_base)
    if not model:
        model = info.get("model_id", "unknown")

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model}")
    print(f"Endpoint: {api_base}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    results = {
        "model": model,
        "endpoint": api_base,
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # Test 1: Inference Speed
    print("Testing inference speed...")
    results["tests"]["speed"] = test_inference_speed(api_base, model)
    speed = results['tests']['speed']
    print(f"  -> {speed.get('tokens_per_sec', 'N/A')} tokens/sec")
    print(f"  -> Latency: p50={speed.get('latency_p50_sec', 'N/A')}s, p90={speed.get('latency_p90_sec', 'N/A')}s, p99={speed.get('latency_p99_sec', 'N/A')}s")
    if 'ttft_avg_sec' in speed:
        print(f"  -> TTFT: avg={speed.get('ttft_avg_sec')}s")

    # Test 2: Reasoning
    print("Testing reasoning...")
    results["tests"]["reasoning"] = test_reasoning(api_base, model)
    reasoning = results['tests']['reasoning']
    think_tag_info = " (uses <think> tags)" if reasoning.get('uses_think_tags') else ""
    print(f"  -> {reasoning['score']} ({reasoning['percentage']}%){think_tag_info}")

    # Test 3: Knowledge (MMLU-style)
    print("Testing factual knowledge...")
    results["tests"]["knowledge"] = test_knowledge(api_base, model)
    knowledge = results['tests']['knowledge']
    print(f"  -> {knowledge['score']} ({knowledge['percentage']}%)")

    # Test 4: Tool Calling
    print("Testing tool calling...")
    results["tests"]["tool_calling"] = test_tool_calling(api_base, model)
    tc = results['tests']['tool_calling']
    print(f"  -> Supported: {tc.get('supported', False)}")
    if tc.get('diagnostics'):
        for diag in tc['diagnostics'][:2]:  # Show first 2 diagnostics
            print(f"     {diag[:80]}")

    # Test 5: Structured Output
    print("Testing structured output (JSON)...")
    results["tests"]["structured_output"] = test_structured_output(api_base, model)
    print(f"  -> Valid JSON: {results['tests']['structured_output'].get('valid_json', False)}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Speed: {speed.get('tokens_per_sec', 'N/A')} tok/s (p90 latency: {speed.get('latency_p90_sec', 'N/A')}s)")
    print(f"Reasoning: {reasoning['percentage']}%{think_tag_info}")
    print(f"Knowledge: {knowledge['percentage']}%")
    print(f"Tool Calling: {'Y' if tc.get('supported') else 'X'}")
    print(f"JSON Output: {'Y' if results['tests']['structured_output'].get('valid_json') else 'X'}")

    if verbose:
        print(f"\nFull results:\n{json.dumps(results, indent=2)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Quick benchmark for vLLM endpoints")
    parser.add_argument("--api-base", "-a", required=True, help="API base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--model", "-m", help="Model name (auto-detected if not specified)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--compare", "-c", help="Compare with another endpoint")

    args = parser.parse_args()

    results = [run_benchmark(args.api_base, args.model, args.verbose)]

    if args.compare:
        results.append(run_benchmark(args.compare, args.model, args.verbose))

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r['endpoint']}")
            print(f"    Model: {r['model']}")
            print(f"    Speed: {r['tests']['speed'].get('tokens_per_sec', 'N/A')} tok/s")
            print(f"    Reasoning: {r['tests']['reasoning']['percentage']}%")
            print(f"    Knowledge: {r['tests']['knowledge']['percentage']}%")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results[0] if len(results) == 1 else results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
