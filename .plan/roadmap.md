# Project Execution Roadmap
**Status:** Completed (Phases 1 - 10)  
**Instruction for Execution Agent:** All core phases, extension phases, and AI + NCM synergy upgrades are implemented and backed by 24 passing automated tests with zero warnings.

---

## Completed Core Phases (Phases 1 - 8)
- [x] **Phase 1: Workspace Scaffolding, Dependencies & LM Studio Configuration**
- [x] **Phase 2: Literal Symbol Guard & SQLite Concept Memory Cache**
- [x] **Phase 3: NCM Multiscale Tokenizer & Parallel Expander Bridge**
- [x] **Phase 4: LM Studio Provider, GGUF Engine & Concept-Skeleton DSL**
- [x] **Phase 5: Pre-Installed AI Agent Tool Suite & Autonomous Specialists**
- [x] **Phase 6: Core QNM Orchestrator Pipeline & Telemetry Engine**
- [x] **Phase 7: Persistent Background Daemon & OpenAI-Compatible Proxy**
- [x] **Phase 8: Terminal CLI, Interactive REPL & Windows Executable Batch Scripts**

---

## Completed Extension Phase 9: Dynamic Model Switching & Custom Agent Presets
- [x] **Task 9.1: Dynamic `--model` Parameter in CLI (`ask`, `agent`, `chat`)**
- [x] **Task 9.2: Custom YAML Agent Presets Engine (`config/agents.yaml`)**
- [x] **Task 9.3: Interactive LM Studio Model Selector in `run_agent.bat`**

---

## Completed Synergy Phase 10: "Normal AI + NCM" Integration & Terminal Polish
- [x] **Task 10.1: Short-Query Loss Prevention in NCM Compressor**
  - *Action:* Updated `NCMCompressor._encode_algorithmic` to preserve short queries ($\le 25$ words) without aggressive trimming, resolving blank responses on greetings like "hi".
  - *Target Files:* `src/qnm/ncm_engine/compressor.py`, `tests/test_ncm_compression.py`
  - *Done When:* Short queries retain full wording while generating discrete codes for memory. (Verified)

- [x] **Task 10.2: Normal AI Conversational Fluency & Raised Token Ceiling**
  - *Action:* Redesigned `SkeletonDSL.SYSTEM_PROMPT` to allow natural, articulate responses and code generation without rigid telegraphic tags. Raised default `skeleton_max_tokens` to 1024.
  - *Target Files:* `src/qnm/llm_engine/skeleton_dsl.py`, `config/default_qnm.yaml`, `src/qnm/settings.py`
  - *Done When:* Model writes full conversational responses and multi-block code solutions. (Verified)

- [x] **Task 10.3: Native Windows Terminal ANSI Virtual Terminal Processing**
  - *Action:* Enabled `ENABLE_VIRTUAL_TERMINAL_PROCESSING` via `kernel32.SetConsoleMode` in `main.py` and configured Rich `Console()` for native Windows CMD rendering.
  - *Target Files:* `src/qnm/cli/main.py`
  - *Done When:* Terminal displays clean Unicode borders and colors without raw `[1;36m` artifacts. (Verified)
