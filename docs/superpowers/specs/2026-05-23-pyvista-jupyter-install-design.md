# pyvista[jupyter] Installation Fix

**Date:** 2026-05-23

## Problem

`pyproject.toml` and `requirements.txt` declare bare `pyvista` with no extras. The interactive Jupyter stack (`trame`, `trame-vtk`, `trame-vuetify`, `ipywidgets`) happens to be installed via explicit lines below, but `pyvista` itself has no declared dependency on them. Fresh installs that skip or reorder those lines could leave the interactive backend broken.

## Solution

Swap `pyvista` → `pyvista[jupyter]` in both dependency files. Keep the explicit trame lines unchanged — they serve as version anchors and pip deduplicates.

## Changes

| File | Line | Before | After |
|------|------|--------|-------|
| `pyproject.toml` | 33 | `"pyvista"` | `"pyvista[jupyter]"` |
| `requirements.txt` | 8 | `pyvista` | `pyvista[jupyter]` |

## Scope

Main branch (`refactor/cu121`) only. Worktrees inherit on merge.

## Non-goals

- No backend auto-configuration (users call `pv.set_jupyter_backend('trame')` or use per-notebook defaults)
- No trame version changes
- No Dockerfile changes
