# MaskCLIP Reference Comparison Notebook — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement remaining tasks.

**Goal:** `docs/semantics/maskclip_reference_comparison.ipynb` — diagnostic notebook that exposes why `MaskCLIPExtractor` produces wrong similarity maps.

**Architecture:** Single notebook, 7 sections, sequential execution. Test image: `docs/semantics/dog-cat.jpg`.

**Background:** Reference notebook (`RogerQi/maskclip_onnx/clip_playground.ipynb`) uses ImageNet stats at 224px — same stats as our code. It only extracts features, no text similarity. The notebook therefore investigates three questions independently:
1. Does 1024px input (vs model-native 336px) degrade features?
2. Are patch features unit-normalized? (If not, `F.normalize` before einsum matters)
3. Are text embeddings correct?

**Status:** Notebook fully written and committed. Remaining work: open in Jupyter, run all cells, verify outputs match expected behavior, fix any runtime errors.

---

### Task: Run and verify the complete notebook

**File:** `docs/semantics/maskclip_reference_comparison.ipynb`

- [ ] Open notebook in Jupyter (or run with `jupyter nbconvert --to notebook --execute`)
- [ ] §1 Setup: model loads, IMAGE_PATH resolves to `dog-cat.jpg`, patch grid = 24×24
- [ ] §2 Resolution: both tensors shown, shapes printed, PIL images displayed
- [ ] §3 Patch features: norm histograms appear, spatial norm map shown
- [ ] §4 Text embeddings: all cosine sims print (expected ≈ 1.0)
- [ ] §5 Normalization effect: 2×3 grid of heatmaps, diff values printed
- [ ] §6 Four-column maps: 2×4 grid, Pearson correlations printed per query
- [ ] §7 Summary: Markdown table rendered with real values

**Expected §4 output:** all cosine sims ≈ 1.000000

**Expected §3/§5 finding:** if mean norm ≠ 1.0, `F.normalize` will visibly change maps

**Expected §6 finding:** 336px columns should show cleaner dog/cat localization than 1024px

### Fix runtime errors (if any)

Common issues to check:
- `resize_image` import: already imported in setup cell as `from collab_splats.utils.image import resize_image`
- `pearson()` helper defined in §6, used in §7 — §7 must run after §6
- `img_1024_pil` defined in §2, used in §5 — §5 must run after §2
- `feat_1024_chw` / `feat_1024_norm_chw` defined in §5, used in §6/§7 — sequential order required

### Commit after successful run

```bash
git add docs/semantics/maskclip_reference_comparison.ipynb \
        worklog/history/specs/2026-04-21-maskclip-reference-comparison-design.md \
        worklog/history/plans/2026-04-21-maskclip-reference-comparison.md
git commit -m "feat(semantics): complete MaskCLIP reference comparison diagnostic notebook"
```
