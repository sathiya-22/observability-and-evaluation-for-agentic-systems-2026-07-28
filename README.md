The Real Problem: Observability and Evaluation for Agentic Systems

Developers building agentic AI systems currently struggle with adequate tools to understand, debug, and optimize their agents. Key challenges include:
1.  **Tracing Agent Execution:** It's hard to follow the exact sequence of steps an agent takes, including tool calls, internal reasoning, and LLM interactions. This makes debugging complex behaviors extremely difficult.
2.  **Extracting User Feedback:** Automatically deriving structured feedback from natural language conversations with agents is crucial for iterative improvement but often requires manual review or custom NLP solutions.
3.  **Cost Monitoring:** Understanding the token usage and associated costs of LLM interactions is vital for financial planning and optimization, especially in production environments.

This project provides a lightweight, provider-agnostic framework for capturing and analyzing agent execution traces, including LLM interactions and tool calls. It focuses on providing immediate, actionable insights for debugging and performance evaluation.

Why this project shape/stack was chosen:

A Python package was chosen because Python is the dominant language in the AI/ML ecosystem, making it accessible to the target audience. The package structure allows for easy integration into existing agent frameworks and provides a clear separation of concerns between data capture, storage, and analysis. It's designed to be a library that agent developers can instrument their code with, rather than a separate service, reducing friction for adoption.

Setup and Usage (Zero API Keys Required):

1.  **Save the files:** Save all the provided files into a directory.
2.  **Install dependencies:**
    ```bash
    pip install click pandas tabulate
    ```
3.  **Run the demo:** The demo uses pre-recorded fixture data to simulate agent runs.
    ```bash
    python -m agent_eval.cli analyze fixtures/agent_trace_example.jsonl
    ```
    This command will process the fixture data and print an analysis of agent steps, LLM calls, and estimated costs to the console.

Optional Real-LLM Adapter:

This prototype does not include a real-LLM adapter because its core functionality (tracing, cost estimation based on *token counts*, feedback extraction from *transcripts*) operates on the *output* of LLM calls and agent steps, not the calls themselves. Token counting is done via a simple heuristic (character count / 4), which is sufficient for a prototype and provider-agnostic. The focus is on the *infrastructure for evaluation*, not on *making LLM calls*.
