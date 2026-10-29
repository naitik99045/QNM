# QNM (Quantized-Concept Network Model) Local Hybrid Engine & Autonomous Agent CLI

QNM is a 100% local-first (and optional hybrid-API) AI engine, persistent background daemon, drop-in OpenAI-compatible proxy server (`/v1/chat/completions`), and terminal CLI agent.

It features a **Dual-Bottleneck Sandwich Architecture**:
1. **Stage 1 (Literal Symbol Guard):** Extracts file paths, identifiers, flags, URLs, and code snippets into deterministic symbols (`$SYM_0`, `$SYM_1`, ...).
2. **Stage 2 (NCM Multiscale VQ-VAE Compression):** Compresses historical context and attached files by 65% to 80% into discrete coarse concept codes.
3. **Stage 3 (SQLite Concept Memory):** Persists session gists with budget-aware token retrieval (< 512 tokens).
4. **Stage 4 (LM Studio / Local GGUF Reasoning):** Operates on a minimal KV-cache and produces a concise Concept-Skeleton DSL blueprint (< 100 tokens).
5. **Stage 5 (NCM Parallel Expansion):** Decodes coarse-to-fine hierarchy levels simultaneously, restoring all protected symbols losslessly.

## Quick Start with LM Studio

1. **Launch LM Studio Server**:
   Ensure LM Studio is installed, or run:
   ```powershell
   lms server start
   ```
2. **Run QNM via Executable Script**:
   Double click or execute:
   ```cmd
   run_agent.bat
   ```
3. **Run Commands Directly via CLI**:
   - `python -m qnm.cli.main ask "Explain how QNM reduces KV-cache bloat"`
   - `python -m qnm.cli.main agent code "Write a unit test for my module"`
   - `python -m qnm.cli.main agent research "Summarize the design of QNM"`
   - `python -m qnm.cli.main agent shell "Check system status and models"`
   - `python -m qnm.cli.main serve` (boots persistent daemon on http://127.0.0.1:8000)
   - `python -m qnm.cli.main stats` (displays lifetime token savings scoreboard)
