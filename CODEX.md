# Codex — Tool-Specific Instructions

> **Read `AGENTS.md` first.** It holds the shared truth: architecture, the
> source-of-truth (`config.py`), the NO-AI-WATERMARK rule, workflows, and the
> two-layer memory policy. This file adds ONLY Codex's dual-graph mechanics.

## Context layering (Codex)

- `.dual-graph-context/PROJECT_CONTEXT.md` — stable project context.
- `.dual-graph-context/SESSION_CONTEXT.md` — current in-chat scope.
- `.dual-graph-context/packs/*.md` — domain-specific context packs.
- Never ask for "full context" or entire chat history. Ask only for missing
  sections, one short scoped question at a time.

## Dual-Graph retrieval order (MANDATORY)

1. **Call `graph_continue` first** — before any file exploration, grep, or reading.
2. If `needs_project=true` → call `graph_scan` with `pwd`. Don't ask.
3. If `skip=true` (<5 files) → no broad/recursive exploration; read named files only.
4. Load context layers (PROJECT/SESSION + only relevant packs) before implementation
   decisions. If critical context is missing, ask one scoped question and continue.
5. **Read `recommended_files` with `graph_read` — one call per file**, single `file`
   string. `file::symbol` reads only that symbol's lines — pass verbatim.
6. **Obey `confidence` caps strictly:** `high` → stop; `medium`/`low` → `fallback_rg`
   ≤ `max_supplementary_greps`, then `graph_read` ≤ `max_supplementary_files`, then stop.

## Rules

- No `rg`/`grep`/bash exploration before `graph_continue`. No broad/recursive exploration.
- `max_supplementary_*` are hard caps — never exceed. Don't dump full chat history.
- Don't call `graph_retrieve` more than once per turn.
- After edits, call `graph_register_edit` with changed files (use `file::symbol`).

## Memory (see AGENTS.md for the policy)

- Log decisions/tasks/blockers via `graph_add_memory(...)`. On session end, (re)write
  `CONTEXT.md` per the format in `AGENTS.md`.
