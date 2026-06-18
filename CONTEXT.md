# Session Context

**Current Task:** Committed today's analytical/hardware/validation work in clean,
no-AI-watermark chunks; set up unified cross-AI context files; ran a code review
and fixed the top two findings.

**Key Decisions:**
- Hybrid context setup: `AGENTS.md` = shared truth; `CLAUDE.md`/`CODEX.md` thin +
  tool-specific; `GEMINI.md` & `.github/copilot-instructions.md` symlink → AGENTS.md.
- No AI watermarks anywhere — commits attributed to user only (force-pushed to strip
  earlier co-authored commits).
- Memory = dual-graph MCP store (live) + this `CONTEXT.md` (session resume).

**Next Steps:**
- Optional: fix remaining 3 review findings (tendon_tension efficiency, std(ddof=1)
  on n≤1 groups, Picard silent non-convergence).
- Optional: add `setup-ai-context.sh` to recreate symlinks on a fresh clone.
