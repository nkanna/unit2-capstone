"""
Smoke test: verify Bedrock connectivity and Claude response before building agents.
Run with:  python smoke_test.py
"""

if __name__ == "__main__":
    from bedrock_client import invoke

    response = invoke(
        messages=[{"role": "user", "content": "Say hello in exactly one sentence."}],
        max_tokens=64,
    )

    print("=== Smoke Test Result ===")
    print(f"Response text : {response.text}")
    print(f"Input tokens  : {response.input_tokens}")
    print(f"Output tokens : {response.output_tokens}")
    print(f"Stop reason   : {response.stop_reason}")
    print("=== PASS ===")
