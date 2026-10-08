"""
Interactive CLI for the Multi-Agent RAG system.

Usage
-----
Interactive mode:
    python main.py

Single-query mode:
    python main.py "What is the password policy?"

Special commands (interactive mode):
    exit / quit   — end the session
    reset         — clear conversation history
    questions     — show example questions
"""

import sys

from agents.manager import Manager, ManagerResult

EXAMPLE_QUESTIONS = [
    # Qualitative
    "What is the password policy?",
    "How many vacation days do employees get?",
    "What is the onboarding process for new hires?",
    # Quantitative
    "What was total revenue by region in 2024?",
    "What is the customer churn rate?",
    "What is the average employee satisfaction score?",
    # Combined
    "What is the remote work policy and how many employees work remotely?",
    # Out of scope
    "What is the capital of France?",
]

_BANNER = """\
========================================
 Multi-Agent RAG — Acme Corp Intelligence
========================================
Type your question, or:
  reset     — clear conversation history
  questions — show example questions
  exit/quit — end session
"""


def _display(result: ManagerResult) -> None:
    print()
    print(f"[Route: {result.classification}]")
    print()
    print(result.final_answer)
    print()

    # Sources / SQL
    if result.qualitative_result and not result.qualitative_result.refused:
        citations = result.qualitative_result.citations
        if citations:
            print(f"Sources: {', '.join(c.get('chunk_id','?') for c in citations)}")

    if result.quantitative_result and not result.quantitative_result.refused:
        qr = result.quantitative_result
        print(f"SQL: {qr.sql}")
        if qr.result_rows:
            print(f"Rows returned: {qr.row_count}")

    # Validation
    val = result.validation
    if val.errors:
        print(f"[Validation ERRORS: {'; '.join(val.errors[:3])}]")
    if val.warnings:
        print(f"[Validation warnings: {'; '.join(val.warnings[:3])}]")

    # Missing provider usage must remain unknown in CLI totals, too.
    from tokenomics.logger import sum_known
    total_in = sum_known(r.input_tokens for r in result.call_records)
    total_out = sum_known(r.output_tokens for r in result.call_records)
    paid = sum_known(r.hypothetical_paid_tier_cost_usd for r in result.call_records)
    free = sum_known(r.free_tier_estimated_cost_usd for r in result.call_records)
    paid_label = f"${paid:.6f}" if paid is not None else "unknown"
    free_label = f"${free:.6f}" if free is not None else "unknown"
    print(f"[Gemini tokens: {total_in} in / {total_out} out]")
    print(f"[Conditional free-tier estimate: {free_label}; hypothetical paid-tier estimate: {paid_label}]")
    print()


def _run_interactive(manager: Manager) -> None:
    print(_BANNER)
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if query.lower() == "reset":
            manager.reset()
            print("[History cleared]\n")
            continue
        if query.lower() == "questions":
            print("\nExample questions:")
            for q in EXAMPLE_QUESTIONS:
                print(f"  • {q}")
            print()
            continue

        result = manager.answer(query)
        _display(result)


def _run_single(manager: Manager, query: str) -> None:
    result = manager.answer(query)
    _display(result)


if __name__ == "__main__":
    mgr = Manager()
    if len(sys.argv) > 1:
        _run_single(mgr, " ".join(sys.argv[1:]))
    else:
        _run_interactive(mgr)
