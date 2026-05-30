# ADR 007: Dashboard as Integration Test Harness

**Status:** Accepted
**Date:** 2026-04-20
**Tags:** dashboard, testing, integration

## Context
Unit tests verify API contracts (one function, one path, one assertion). They do not catch problems where the contract is satisfied but the full pipeline produces visually wrong output: misaligned features, broken queries, wrong segmentation, drifted poses. Full end-to-end pipeline tests via CI are slow and brittle (dataset assumptions, hardware variability).

## Decision
The dashboard is treated as the integration test harness for the whole pipeline. Any subsystem change that ships also lands in the dashboard, where it is exercised against representative data and inspected visually before merge. Unit tests cover API contracts; the dashboard covers "does the pipeline still feel right."

## Consequences
**Positive:**
- Catches integration-level regressions that unit tests miss.
- Forces every refactor to remain runnable end-to-end.
- Doubles as developer documentation — running the dashboard demonstrates the full feature set.

**Negative:**
- Dashboard breakage becomes a release blocker.
- Requires GPU + dataset availability to validate.

**Revisit if:** a faster automated end-to-end harness (e.g. golden-output regression suite) supersedes the manual visual check.

## Alternatives Considered
- **CI-only integration tests.** Rejected: dataset + GPU requirements make them brittle in CI.
- **Skip integration testing.** Rejected: too many silent regressions slipped through unit tests alone.
