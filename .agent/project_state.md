# Project State & Context
**Last Updated:** 2026-10-27T17:50:00+05:30  
**Project Name:** QNM (Quantized-Concept Network Model) Local Hybrid Engine & Autonomous Agent Suite  
**Engine Architecture:** Dual-Bottleneck Sandwich (Literal Symbol Guard -> NCM VQ-VAE Multiscale Compressor -> SQLite Concept Memory -> LM Studio / GGUF Reasoning -> NCM Parallel Expander -> Symbol Restoration)  
**Host Environment:** Windows 11, Python 3.12.10, LM Studio 0.3+ (`lms.exe` active on port 1234 with `qwen2.5-coder-3b-instruct` loaded)  
**Build Status:** Complete & Fully Verified (24/24 Tests Passing, 100% Green)

---

## 1. Project Overview

QNM is a local-first AI engine, persistent background daemon, OpenAI-compatible proxy server (`/v1/chat/completions`), and autonomous agent CLI. It fuses **Next-Concept Modeling (NCM)** with **Large Language Model (LLM) reasoning**, providing the intelligence and fluency of a normal AI assistant augmented with guaranteed literal symbol preservation and multi-turn concept memory compression.

### AI Agent + NCM Integration Matrix:
The interactive menu in `run_agent.bat` provides 4 primary ways to use an AI agent and NCM together:
- **Option [1] - Interactive Agent Chat REPL:** Multi-turn conversational AI with SQLite concept memory, automatic context gist pruning, and lossless literal symbol restoration. Best for natural conversations, technical Q&A, and interactive coding.
- **Option [2] - Code Specialist Agent:** Autonomous ReAct agent with sandboxed file tools (`read_file`, `write_file`, `edit_file`, `run_tests`) that navigates the codebase through the NCM pipeline, protecting paths and code tokens with `$SYM_n`.
- **Option [3] - Research Specialist Agent:** Autonomous agent with DuckDuckGo live internet search and document ingestion that leverages NCM VQ-VAE multiscale compression to digest massive documents by 65%–80% without context overflow.
- **Option [4] - Shell & DevOps Agent:** Autonomous agent with sandboxed terminal execution for running commands, verifying system health, and executing builds while guarding system paths.

---

## 2. Brainstormed Ideas & Architecture (Phase 10: Normal AI + NCM)

1. **Intelligent Dynamic Compression Thresholding:**
   - Short queries ($\le 25$ words) like greetings, one-line questions, or quick commands are preserved 100% intact without aggressive word stripping.
   - Long contexts, multi-turn history gists, and large reference files (> 25 words) undergo multiscale VQ-VAE compression down to Level 2 cutoff, reducing context by 65%–80% while retaining all `$SYM_n` tokens.
2. **Conversational Fluency & Technical Depth:**
   - Relaxed rigid telegraphic DSL constraints on general queries so the AI responds like an expert assistant (complete explanations, clean runnable code blocks, warm greetings).
   - Raised the reasoning token budget from 100 to 1024 tokens in `config/default_qnm.yaml` and `src/qnm/settings.py`.
3. **Lossless Precision on Critical Identifiers:**
   - Absolute Windows file paths, URLs, code blocks, regexes, and CLI parameters are shielded as `$SYM_0`..`$SYM_k` by the Literal Symbol Guard.
   - The LLM reasons over the gists and symbols without mutating paths or hallucinating backslashes.
   - The NCM Expander re-injects the exact original literals before presenting the output to the user.
4. **Native Windows Console Terminal Support:**
   - Initialized `ENABLE_VIRTUAL_TERMINAL_PROCESSING` (0x0004) via Win32 `kernel32.SetConsoleMode` so ANSI escape codes and Rich Unicode formatting render natively in CMD without printing raw `[1;36m` artifacts.

---

## 3. Current Progress

- Scaffolding & Configuration: Installed in editable mode (`pip install -e .`).
- Dual-Bottleneck Pipeline: Complete 5-stage pipeline with real-time telemetry.
- AI Agent Suite: `CodeAgent`, `ResearchAgent`, `ShellAgent`, plus custom YAML presets in `config/agents.yaml`.
- Persistent Daemon: FastAPI background server on port 8000 with `/v1/chat/completions` proxy.
- Executables: `run_agent.bat` and `qnm.bat` fully operational on Windows 11.
- Test Coverage: 24 automated unit and pipeline tests passing cleanly.

---

## 4. Pending Work & Next Steps

- **Enhance Web UI Interface:** Optional local browser-based GUI (Streamlit / Gradio or React frontend) to supplement the terminal CLI and batch script.
- **Multi-Agent Collaboration:** Implement a supervisor orchestrator allowing `CodeAgent` and `ResearchAgent` to work in tandem on complex projects.
- **Custom NCM Fine-Tuning:** Provide local script to fine-tune the multiscale VQ-VAE codebook on domain-specific codebases.
