# Decisions

This log records project choices that later agents should not rediscover. Each entry states what we chose and why.

## Why the model map is in the repo (2026-10-20)

Cloud agents read Cursor rules from this repository. They do not read a developer's home directory. The per-role map therefore lives at `.cursor/rules/pstack-models.mdc`.

The committed map uses the medium budget, which sets high reasoning. That choice costs less than the large budget and the unlimited budget. It also gives up some reasoning depth.

Naveed picks the budget. Do not change the map or the budget unless Naveed asks.
