# Unit 2 Capstone: Multi-Agent RAG System with Claude

## Goal
Build a Python 3.11 CLI application that answers enterprise
documentation and business-data questions. Target Gold.

## Required Architecture
- Manager agent: classify queries as qualitative, quantitative,
  or both; route them; synthesize the final response.
- Qualitative agent: retrieve relevant document chunks from ChromaDB
  using Sentence-Transformers, then generate a grounded Claude answer
  with source citations.
- Quantitative agent: translate natural-language questions into SQL,
  validate SQL before execution, query SQLite, and use Claude to
  interpret the results.
- Validation layer: validate every Claude response before returning
  it to the user.
- Tokenomics logger: record actual input/output tokens and estimated
  cost per query.

## Technology and Authentication
- Python 3.11 with a virtual environment and requirements.txt.
- Claude Sonnet 4.6 for all LLM calls.
- AWS Bedrock through boto3, using configured AWS credentials.
- Region: us-east-1.
- Model: us.anthropic.claude-sonnet-4-6.
- ChromaDB and Sentence-Transformers for document retrieval.
- SQLite for quantitative data.
- Configuration in .env; never commit secrets.
- Document that Bedrock authentication replaces the original brief's
  direct Anthropic API key setup.

## Bronze Requirements
1. ingest.py chunks documents, embeds them, and stores them in ChromaDB.
2. Manager supports qualitative, quantitative, and combined queries.
3. Qualitative prompts require answers based only on retrieved context,
   citations, and refusal when information is unavailable.
4. Quantitative agent generates SQL through Claude.
5. validate_sql blocks non-SELECT and unsafe queries before execution.
6. Every Claude response passes through validation.
7. Tokenomics logs actual tokens and estimated cost.
8. main.py provides an interactive CLI.
9. README includes architecture, setup, and usage instructions.
10. README has a Trust-but-Verify section with at least three real
    testing examples, covering:
    - Query submitted.
    - Actual Claude output.
    - Validation findings.
    - What the learner accepted or changed, and why.
    - At least one output the learner did not immediately trust,
      and how it was resolved.
11. Submit tokenomics_log.jsonl showing at least 10 real user queries.

## Silver Requirement
Maintain conversation history so follow-up questions can refer
to earlier queries and answers.

## Gold Requirement
Meaningful unit tests for the manager, qualitative agent,
quantitative agent, and validator.

## Required Project Files
agents/manager.py
agents/qualitative.py
agents/quantitative.py
validation/validator.py
tokenomics/logger.py
data/documents/
data/database.sqlite
ingest.py
main.py
requirements.txt
README.md
tokenomics_log.jsonl

## Verification
- Verify Claude access with a smoke test before building agents.
- Verify retrieval independently before adding Claude generation.
- Test irrelevant context and unsafe SQL.
- Test all three query routes and conversation follow-ups.
- Never fabricate testing outputs, token logs, or review decisions.
- Keep changes modular and explain them for human review.