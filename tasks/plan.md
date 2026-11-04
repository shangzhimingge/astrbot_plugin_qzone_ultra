# Execution plan

1. Persona: add `qzone_bridge/persona.py`, inject into QzoneLLM, bridge effective event/session/config Persona and context-free schedules; add config `llm.persona_id`; preserve composed prompt on provider API fallback. Focused tests, compile, commit.
2. Auto-reply backend: add numeric config validation/defaults/schema and `qzone_bridge/auto_reply.py` service with durable watermark, comment statuses, account isolation, task lifecycle, fresh detail revalidation, quota/delay/locks/uncertain semantics. Integrate minimal lifecycle to `main.py` and manual reply shared lock. Tests by failure mode, compile, commit.
3. Pages AI draft: add injected callbacks and authenticated routes in `main.py`/`page_api.py`; validate opaque post/comment identities using uncached detail. Add stable buttons in existing forms; test request/response, authorization route contracts, stale targets, preserved drafts, no auto-submit. Commit.
4. Docs and final verification: README/CHANGELOG/implementation.md, Python/Node tests and compile/diff check; prepare external fixture runner and API contract for root Playwright QA. Save raw logs under `D:\Documents\杂项\output\qzone_fork_fix_20261104`. Do not push/PR; pause for Sol review and root browser QA.

## Ordering / non-goals
No project code before this spec/plan/todo. Keep each feature as a small incremental slice. Do not add network integration, migrations/dependencies, alternate auth, or unrelated refactors. If an ambiguity would require changing these contracts, stop and ask Sol planner before edits.
