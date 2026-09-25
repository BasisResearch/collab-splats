# Semantics Release Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `collab_splats/semantics/` releasable: brief prose, no dead code, tunables as kwargs, silent fallbacks raise.

**Architecture:** Round 1 edits only docstrings and comments, and each commit is proven behavior-free by an AST comparison. Round 2 makes one logical code change per commit, test first, and runs the gate after each one. The end state adds `"semantics"` to `RELEASED` in the docstring contract.

**Tech Stack:** Python 3.11, torch, zarr 3.1.5, pytest, `ast`.

**Spec:** [2026-09-24-semantics-release-cleanup-design.md](../specs/2026-09-24-semantics-release-cleanup-design.md) · Rules: [decision 017](../decisions/017-release-cleanup-rules.md)

---

## Conventions for every task

- `WT=/workspace/collab-splats/.worktrees/semantics-release` (branch `clean/semantics-release`).
  Every command starts with `cd $WT &&`, because the working directory resets between Bash calls.
- `SP=/tmp/claude-0/-workspace-collab-splats/84b460d8-9feb-4184-a0e7-a553a46edc06/scratchpad`
  holds scratch files: the proof tool and the baseline. Nothing in `$SP` is committed.
- **Gate** (`G`), run after every commit-bound change:

  ```bash
  cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" \
    && cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest \
       tests/semantics tests/utils tests/test_docstring_contract.py \
       tests/test_semantics_logging.py tests/dashboard/test_pipeline.py tests/dashboard/test_viewer_lift.py \
       tests/dashboard/test_semantics_layout.py tests/wrapper/test_mask_sky.py -q -p no:cacheprovider
  ```

  - The printed path must start with `$WT/`.
  - Never pipe through `tail`, and never use `--tb=no`.
  - Compare the counts against `$SP/baseline.txt`. Every change in a count must be
    explained by the task's own test adds and deletes. Compare the SKIP count too.
  - Never run the full suite. Another session gates at the same time, and two full suites
    exceed the 46.6 GB cgroup.
  - If collection fails on `import nvdiffrast`, stub it with a pytest plugin in `$SP` and pass
    `-p` with that plugin. Never commit the stub.
- Stay inside `collab_splats/semantics/`, `tests/semantics/`, `tests/test_docstring_contract.py`
  and this plan/spec. Any caller edit outside them is **deferred** (spec § Deferred), not made.
- Commit only your own paths: `git add <paths> && git commit --only <paths> -m ...`. Other
  sessions share the git index. Never `--amend`, rebase, reset or bare `git stash`; fix a
  mistake with a new commit. Do not merge or push.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Spelling is US (`color`, `normalize`).
- Prose shape (CLAUDE.md, decision 017):
  - docstring summary on the line after `"""`, ≤100 chars, then `- ` bullets (≤6), then `Args:` / `Returns:` / `Raises:`
  - comment runs ≤4 lines; a run of 3+ lines is a header line followed by `- ` bullets
  - no `measured`, `hypothesis`, `Nx faster`, or scene ids

---

## Task 0: Verify worktree, proof tool, baseline

**Files:**
- Create (scratch, not committed): `$SP/prose_proof.py`, `$SP/baseline.txt`

The worktree and branch already exist (forked from `clean/final` at `d7440030`). Three commits
are on it: contract checks `9c7090f9`, chained bullets `9f8df7a8`, and the spec `2672fd76`.

- [ ] **Step 1: Verify the worktree, the symlinks and a clean tree**

```bash
cd $WT && git branch --show-current && git status --short && ls third_party
```

Expected output:
- `clean/semantics-release`
- empty status, apart from this plan if it is still uncommitted
- `third_party` lists `LoGeR VGGT-SLAM VGGT-X Video-Depth-Anything bae hloc vggt-omega vggt_spark xfeat`

- [ ] **Step 2: Write the prose-proof tool**

```bash
cat > $SP/prose_proof.py <<'EOF'
"""
Prove a commit range changed only docstrings and comments in the given .py files.

Usage: prose_proof.py <base-ref> <file> [<file> ...]   (run from the worktree root)
"""
import ast
import subprocess
import sys


def stripped(src: str) -> str:
    """
    AST dump with every docstring statement deleted; comments never reach the AST.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [
                s for s in body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
            ] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


base, files = sys.argv[1], sys.argv[2:]
bad = []
for f in files:
    old = subprocess.run(["git", "show", f"{base}:{f}"], capture_output=True, text=True, check=True).stdout
    new = open(f).read()
    if stripped(old) != stripped(new):
        bad.append(f)
print("CODE CHANGED:" if bad else "PROSE ONLY", *bad)
sys.exit(1 if bad else 0)
EOF
```

- [ ] **Step 3: Sanity-check the proof tool on a known-bad mutation**

```bash
cd $WT && cp collab_splats/semantics/features/base.py $SP/base.bak \
  && sed -i 's/svd_components: int = 500) -> None:/svd_components: int = 501) -> None:/' collab_splats/semantics/features/base.py \
  && /opt/venv/reconstruction/bin/python $SP/prose_proof.py HEAD collab_splats/semantics/features/base.py; \
  cp $SP/base.bak collab_splats/semantics/features/base.py && git -C $WT status --short
```

Expected: `CODE CHANGED: collab_splats/semantics/features/base.py`, then an empty `git status`.
If the tool prints `PROSE ONLY`, it is broken. Stop.

- [ ] **Step 4: Run the baseline gate and record the counts**

Run gate `G` on the untouched tree. Write two things into `$SP/baseline.txt`:
- the final summary line, e.g. `N passed, M skipped, K xfailed, J xpassed, F failed`
- every failing node id

Every later gate is compared against this file.

---

## Round 1 — prose only

Every Round 1 task ends with the same three checks:

```bash
cd $WT && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final <files touched>
```

1. The proof above must print `PROSE ONLY`.
2. Gate `G` must match the baseline, except for xfail→xpass moves in `test_docstring_contract.py`.
3. Commit with `docs(semantics): <what>`.

Line numbers below are from `clean/final`. Each edit gives an anchor quote, so earlier
edits in the same file do not throw it off. Use the Edit tool for every replacement.

### Task 1: `__init__.py` files and `utils.py` prose

**Files:**
- Modify: `collab_splats/semantics/__init__.py:1-3`
- Modify: `collab_splats/semantics/features/__init__.py:1`
- Modify: `collab_splats/semantics/segmentation/__init__.py:1-5,9,19`
- Modify: `collab_splats/semantics/utils.py` (docstring of `compute_semantic_contrast`; comments at 218, 248-249, 293-294, 343-348, 383-384, 388-392, 422-423, 439-440)

- [ ] **Step 1: Package docstring** — replace lines 1-3 of `semantics/__init__.py` with:

```python
"""
Semantic features for reconstructed scenes: extract, compress, lift, query, segment.

- features: patch-feature extractors (dinov2, maskclip, talk2dino) behind one registry
- segmentation: mask backends (insid3, mobilesamv2, sam3, skywater) behind one registry
- compression: FeatureAutoencoder, per-point codes for the lifted store
- utils: on-disk layout of the 2D cache and the lifted per-point store
"""
```

- [ ] **Step 2: `features/__init__.py`** — replace line 1 with:

```python
"""
Patch-feature extractors, looked up by name via `BaseFeatureExtractor.get`.

- dinov2: DINOv2 patch features, no text tower
- maskclip, talk2dino: queryable; they also embed text for `score_queries`
"""
```

- [ ] **Step 3: `segmentation/__init__.py`** — replace the module docstring (lines 1-5) with:

```python
"""
Segmentation backends, looked up by name via `BaseSegmentation.get`, plus mask utilities.

- insid3: in-context masks from one reference image and mask
- mobilesamv2, sam3: class-agnostic instance masks; sam3 also takes text prompts
- skywater: per-pixel sky masks; `sky_masks` caches them per scene
"""
```

Replace each `# ── <title> ──…` divider line (9 and 19) with a three-line `########` divider.
Keep the same title, in the style used in `utils.py`:

```python
########################################################################
# <same title>
########################################################################
```

- [ ] **Step 4: `compute_semantic_contrast` docstring** — replace the whole docstring with:

```python
    """
    Contrastive score per patch or point: positive queries against negative ones.

    - no negatives (num_positive == N_queries): a plain reduction over positives
    - "max": each positive scored against all negatives, then max; distinct concepts
    - "pool": positives averaged before the softmax; synonyms read as one query

    Args:
        raw_similarities: (N_queries, N) cosine similarities per patch or point.
        num_positive: rows [0:num_positive] are positive queries; the rest are negative.
        temperature: softmax temperature; lower is sharper. Unused without negatives.
        reduction: "max" or "pool".

    Returns:
        (N,) scores in [0, 1].

    Raises:
        ValueError: when `reduction` is neither "max" nor "pool".
    """
```

Before committing, check that the `Raises:` line is true. Read the function body; if it does
not raise on an unknown `reduction`, delete the `Raises:` section.

- [ ] **Step 5: `utils.py` comments.** Replace each comment run with the text shown.

In `extract_feature_cache`, the run starting `# One path at a time, never one decoded stack`:
```python
    # Decode one path at a time: a full-res decoded stack would not fit in RAM
```

The run starting `# Marker last: until these attrs land`:
```python
    # Validity attrs last: a crash mid-loop leaves a store the check above rejects
```

In `write_point_features`, the 6-line run above the width attrs:
```python
    # Width attrs tell a full-dim store from an orphaned one
    # - input_dim == latent_dim: full-dim codes, no weights needed
    # - unequal: codes need the _ae.pt to decode
    # - order codes, attrs, weights; a failure removes the store (no half-pair left)
```

In `point_features_cached`, the run starting `# No weights: usable only if`:
```python
    # No weights: cached only when the codes are full-dim; else a half-written pair
```

The 5-line run starting `# Broad except on purpose`:
```python
    # Predicate, so an unreadable store answers False instead of raising
```

In `load_point_features`:
- set the `batch_size` Arg to `batch_size: points per decode chunk; bounds peak memory.`
- replace the run above the weights branch with:
  ```python
      # Weights present: always decode, even at equal widths (the codes are still encoded)
  ```
- replace the run above the chunked decode with:
  ```python
      # Decode in chunks into one preallocated array; row-wise ops, so chunking is exact
  ```

Read each original first and keep any fact it states that the replacement drops, if that
fact is still true. In particular, keep the `write_point_features` failure-path fact exactly
as the code does it.

- [ ] **Step 6: Prove, gate, commit**

```bash
cd $WT && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final \
  collab_splats/semantics/__init__.py collab_splats/semantics/features/__init__.py \
  collab_splats/semantics/segmentation/__init__.py collab_splats/semantics/utils.py
```

Expected: `PROSE ONLY`. Run gate `G`, then:

```bash
cd $WT && P="collab_splats/semantics/__init__.py collab_splats/semantics/features/__init__.py collab_splats/semantics/segmentation/__init__.py collab_splats/semantics/utils.py" \
  && git add $P && git commit --only $P -m "docs(semantics): package docstrings and utils prose

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: `compression.py` prose

**Files:**
- Modify: `collab_splats/semantics/compression.py:55-56,60-61,70-71,127-146,183-184`

- [ ] **Step 1: Comments.** Replace each run with the text shown:
  - 55-56 → `# decoder as two submodules, not one Sequential: keeps checkpoint state_dict keys`
  - 60-61 → `# Fit quality from the last fit(), on the training set; trust only when epochs_run > 0`
  - 70-71 → `# No image decode: lifted codes are decoded per point (per_point_decode)`
  - 145-146 → `# Empty input raises: zero steps could still satisfy target_cosine`
  - 183-184 → `# N >= 1 (checked above), so n_batches >= 1`

- [ ] **Step 2: `fit` docstring.**
  - Replace the summary and body with:

    ```python
            """
            Train in place on MSE(recon, x) + (1 - cosine(recon, x)).

            - fit quality lands on `self` as recon_cosine, recon_mse, epochs_run
            - both metrics are on the training set, so optimistic at small N
    ```

  - Replace two Args:
    - `on_epoch: callback(epoch, epochs, avg_loss), once per epoch; a progress hook for UIs.`
    - `target_cosine: stop once mean reconstruction cosine reaches this; None runs all epochs.`
  - Keep every other Arg and the `Raises:` section word for word.

- [ ] **Step 3: Prove, gate, commit**

```bash
cd $WT && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final collab_splats/semantics/compression.py
```

Expected: `PROSE ONLY`. Run gate `G`, then:

```bash
cd $WT && P=collab_splats/semantics/compression.py && git add $P && git commit --only $P -m "docs(semantics): compression prose

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: `features/base.py` prose

**Files:**
- Modify: `collab_splats/semantics/features/base.py`

- [ ] **Step 1: Module docstring** (lines 1-7):

```python
"""
Base classes for patch-feature extractors.

- BaseFeatureExtractor: registry, shared preprocess, positional debiasing
- BaseQueryableExtractor: adds text embedding and contrastive query scoring
"""
```

- [ ] **Step 2: `_DEBIAS_VALIDATED` comment** — lines 27-29 become one line:

```python
# Backends whose positional debiasing has been checked; others warn in debias()
```

- [ ] **Step 3: `BaseFeatureExtractor` class and `__init__` docstrings**

```python
    """
    Image patch-feature extractor, registered by name.

    - register with `@BaseFeatureExtractor.register("name")`, look up with `.get("name")`
    - subclasses set `self._normalize` and `self.patch_size`; `preprocess` is shared
    """
```

```python
        """
        Store the shared preprocessing and debiasing configuration.

        Args:
            resize_mode: "max_size" (longest edge) or "square" (center-crop, then resize).
            image_resolution: longest-edge target for "max_size", side for "square".
            svd_components: positional-subspace rank for debias(); 500 is the INSID3 default.
        """
```

- [ ] **Step 4: Cache comment** (lines 67-69). Replace the comment and both inline comments with:

```python
        # Per patch-grid caches: positional basis (D, K) and zero-image features (D, H_p, W_p)
        self._pos_basis_cache: dict = {}
        self._zero_feats_cache: dict = {}
```

- [ ] **Step 5: `features_to_rgb`**
  - Drop the trailing inline comments on lines 146-149.
  - Keep the one-line `# Normalize to [0, 255] ...` block comment.

- [ ] **Step 6: `_build_positional_basis`.**
  - Docstring:

```python
        """
        Positional subspace at one patch grid, from a zero-pixel image (INSID3).

        - a black image carries no content, so its features are positional only
        - runs through self.forward(), so each backend preprocesses exactly as at inference
        - fills _pos_basis_cache and _zero_feats_cache at (H_p, W_p)

        Args:
            H_p: patch rows.
            W_p: patch columns.
        """
```

  - Body comments: replace every block comment, and drop every trailing inline comment.
    - `# patch_size must be set; an AttributeError beats a wrong zero image` above the `hasattr` check
    - `# Zero-pixel image through the backend's own forward()` above `zero_arr = ...`
    - `# Backend preprocessing changed the grid; bias is smooth, so interpolate` as the first line inside the `if zero_feat.shape[1:] != ...` branch
    - `# Top-K left singular vectors of the centered features span the positional subspace` above `D = zero_feat.shape[0]`
    - Delete the other block comments in this function.

- [ ] **Step 7: `_apply_debias`.**
  - Docstring:

```python
        """
        Project out the positional subspace, then re-normalize each patch.

        Args:
            fmap: (D, H_p, W_p) features at a grid with a cached basis.

        Returns:
            (D, H_p, W_p) debiased features, unit-norm per patch.
        """
```

  - Body: `# P_perp = I - U U^T removes the span(U) component` above `P_perp = ...`.
  - `# Projection breaks unit norm; cosine queries need it back` above the `F.normalize` line.
  - Drop every trailing inline comment.

- [ ] **Step 8: `get_bias_visualization` docstring.** Leave the body alone; Round 2 inlines it.

```python
        """
        RGB view of the positional bias at one patch grid (top-3 PCs of the zero image).

        - needs a prior debias() call at this grid

        Args:
            H_p: patch rows of a prior debias() call.
            W_p: patch columns of a prior debias() call.

        Returns:
            (H_p, W_p, 3) uint8 RGB.

        Raises:
            KeyError: when (H_p, W_p) is not cached.
        """
```

- [ ] **Step 9: `debias` comments**
  - The 2-line run in the warning branch becomes: `# Checked only on DINO-family models; warn so other backends stay usable`
  - The comment in the cache branch becomes: `# First call at this grid builds the basis`
  - Drop the trailing inline comments on the `_, H_p, W_p = ...` and `return` lines.

- [ ] **Step 10: `BaseQueryableExtractor` and `score_queries` docstrings**

```python
    """
    Extractor that also embeds text, so patches can be queried by cosine similarity.

    - subclasses implement `encode_text` and `forward`
    - `compute_similarity` and `score_queries` are shared
    """
```

```python
        """
        Contrastive score of positive queries against negative ones.

        Args:
            features: (C, H, W) patch feature map, or (P, D) point feature array.
            positive: queries that should score high.
            negative: queries that should score low; None uses ["object"] (Talk2DINO's
                convention), [] skips the contrast and returns raw similarity.
            temperature: softmax temperature; lower is sharper.
            reduction: "max" or "pool", see `compute_semantic_contrast`.

        Returns:
            (H, W) scores in [0, 1] for a patch map, (P,) for point features.
        """
```

- [ ] **Step 11: Prove, gate, commit**

```bash
cd $WT && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final collab_splats/semantics/features/base.py
```

Expected: `PROSE ONLY`. Run gate `G`, then:

```bash
cd $WT && P=collab_splats/semantics/features/base.py && git add $P && git commit --only $P -m "docs(semantics): feature base prose

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: `dino.py`, `talk2dino.py`, `maskclip.py` prose

**Files:**
- Modify: `collab_splats/semantics/features/dino.py`, `talk2dino.py`, `maskclip.py`

- [ ] **Step 1: Module docstrings.** In each file, replace the one-line docstring on line 1 with a
three-line form:
  - dino: `DINOv2 patch-feature backend ("dinov2"), via HuggingFace transformers.`
  - talk2dino: `Talk2DINO patch-feature backend ("talk2dino"), with a text tower for queries.`
  - maskclip: `MaskCLIP patch-feature backend ("maskclip"), with CLIP's text tower for queries.`

```python
"""
<summary above>
"""
```

- [ ] **Step 2: `DINOFeatureExtractor` class docstring** (its summary moves off the `"""` line):

```python
    """
    DINOv2 patch features via HuggingFace transformers.

    Args:
        model_name: HuggingFace model id.
        resize_mode: "max_size" (longest edge) or "square" (center-crop, then resize).
        image_resolution: longest-edge target for "max_size", side for "square".
        device: torch device; None picks one with `get_device`.
        svd_components: positional-subspace rank for debias().
    """
```

- [ ] **Step 3: `Talk2DinoExtractor`.**
  - Class docstring:

```python
    """
    Talk2DINO patch features and text embeddings, from the HuggingFace Hub.

    - backbones: Talk2DINOv3-ViTB (DINOv3, default) or Talk2DINO-ViTB (DINOv2)
    - calls forward_features directly: preprocess() already made patch-aligned tensors
    - source: https://github.com/lorebianchi98/Talk2DINO

    Args:
        model_name: HuggingFace Hub model id.
        device: torch device; None picks one with `get_device`.
        resize_mode: "max_size" (longest edge) or "square" (center-crop, then resize).
        image_resolution: longest-edge target for "max_size", side for "square".
        svd_components: positional-subspace rank for debias().
    """
```

  Keep any fact in the old class docstring that these bullets lose. Then delete the
  `__init__` docstring (its Args now live on the class).
  - Comment 52-53 becomes: `# low_cpu_mem_usage=False: Talk2DINO's load_state_dict would no-op on meta tensors`
  - Comment 61-64 becomes: `# patch_size = backbone conv stride (sets the token grid; kernel_size does not)`

- [ ] **Step 4: `MaskCLIPExtractor`.**
  - Class docstring: summary `MaskCLIP patch features and CLIP text embeddings.` on its own line.
  - Keep the existing Args, but the `cache_dir` Arg becomes `cache_dir: weights cache; None uses $TORCH_HOME, else ~/.cache/torch.`
  - Comment 44-45 becomes: `# Read $TORCH_HOME at call time, not at import`
  - Comment 50-53 becomes: `# Imported here: optional dep, needs setuptools<71 (pkg_resources.packaging)`
  - Comment 56-57 becomes: `# Drop the library's square-crop preprocess; preprocess() does CLIP normalization`
  - Delete the line-61 comment about mocks.

- [ ] **Step 5: Prove, gate, commit**

```bash
cd $WT && F="collab_splats/semantics/features/dino.py collab_splats/semantics/features/talk2dino.py collab_splats/semantics/features/maskclip.py" \
  && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final $F
```

Expected: `PROSE ONLY`. Run gate `G`, then:

```bash
cd $WT && P="collab_splats/semantics/features/dino.py collab_splats/semantics/features/talk2dino.py collab_splats/semantics/features/maskclip.py" \
  && git add $P && git commit --only $P -m "docs(semantics): feature backend prose

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: `segmentation/base.py`, `mobile_sam.py`, `sam3.py` prose

**Files:**
- Modify: `collab_splats/semantics/segmentation/base.py`, `mobile_sam.py`, `sam3.py`

- [ ] **Step 1: `base.py` module docstring**

```python
"""
Segmentation base class and mask utilities.

- BaseSegmentation: backend registry and interface
- create_composite_mask, mask_id_to_binary_mask, convert_matched_mask: integer-ID masks
- create_patch_mask, aggregate_masked_features: patch grids and per-mask feature pooling
"""
```

- [ ] **Step 2: `create_composite_mask` docstring and comment**

```python
    """
    Merge SAM results into one (H, W) uint16 mask of integer IDs.

    - higher-confidence masks paint over lower ones; a mask left with ≤10% of its pixels is dropped
    - uint16: SAM routinely yields >255 masks, and numpy>=2 raises on uint8 overflow

    Args:
        results: SAM mask-generator dicts with "segmentation" (H, W) bool and "predicted_iou".
        confidence_threshold: drop masks with predicted_iou below this (or above 1).

    Returns:
        (H, W) uint16; 1-indexed mask IDs, 0 is background.
    """
```

The 2-line run above the area check becomes:
`# ID idx was painted from masks[sorted_idxs[idx - 1]], not masks[idx - 1]`

- [ ] **Step 3: `mask_id_to_binary_mask`, `convert_matched_mask` and `aggregate_masked_features`**
  - In `mask_id_to_binary_mask`, the Arg becomes `composite_mask: (H, W) integer-ID mask; 0 is background.`
  - `convert_matched_mask` docstring:

```python
    """
    Remap sequential mask IDs 1..N to matched label IDs.

    Args:
        labels: (N,) matched label per mask ID; label k is written as k + 1.
        masks: (H, W) sequential mask IDs, 1..N.

    Returns:
        (H, W) uint16 with each ID replaced by its label + 1.
    """
```

  - In `aggregate_masked_features`, remove the extra alignment spaces after each Arg's colon,
    and lower-case the first word of each Arg description.

- [ ] **Step 4: `mobile_sam.py` module docstring**

```python
"""
MobileSAMv2 segmentation backend ("mobilesamv2").

- "object": YOLOv8 boxes prompt SAM
- "auto": SAM's automatic mask generator
"""
```

- [ ] **Step 5: `sam3.py` module docstring, class bullets, `segment` summary**

```python
"""
SAM3 text-prompted segmentation backend ("sam3"); facebook/sam3 is a gated model.
"""
```

In the class docstring, replace the two bullets with:
```
    - gated model: request access to facebook/sam3, then `huggingface-cli login`
    - Sam3Processor places the model on a device itself; no device argument
```

The `segment` summary becomes `Every object in the frame, via the generic prompt "object".`

- [ ] **Step 6: Prove, gate, commit**

```bash
cd $WT && F="collab_splats/semantics/segmentation/base.py collab_splats/semantics/segmentation/mobile_sam.py collab_splats/semantics/segmentation/sam3.py" \
  && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final $F
```

Expected: `PROSE ONLY`. Run gate `G`, then:

```bash
cd $WT && P="collab_splats/semantics/segmentation/base.py collab_splats/semantics/segmentation/mobile_sam.py collab_splats/semantics/segmentation/sam3.py" \
  && git add $P && git commit --only $P -m "docs(semantics): segmentation base, mobilesam, sam3 prose

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: `insid3.py` prose

**Files:**
- Modify: `collab_splats/semantics/segmentation/insid3.py`

- [ ] **Step 1: Module docstring**

```python
"""
INSID3 in-context segmentation backend ("insid3").

- training-free: frozen DINOv2 features, one reference image and mask per category
"""
```

- [ ] **Step 2: Helper docstrings**

`_downsample_mask`:
```python
    """
    Downsample a (H, W) bool mask to (h, w), keeping a tiny mask non-empty.

    - bilinear, then nearest, then the mask's center pixel
    - each step covers a smaller mask than the last; an empty mask stays empty

    Args:
        mask: (H, W) bool.
        h: output rows.
        w: output columns.

    Returns:
        (h, w) bool.
    """
```

`_upsample_mask`:
```python
    """
    Bilinear upsample of a bool mask.

    Args:
        mask: (h, w) bool.
        H: output rows.
        W: output columns.

    Returns:
        (H, W) bool, thresholded at 0.5.
    """
```

`_tensor_to_pil`:
```python
    """
    (C, H, W) float tensor in [0, 1] to a PIL image.

    Args:
        t: (C, H, W) float tensor in [0, 1].

    Returns:
        uint8 PIL image.
    """
```

Before committing, check each of these three against its body, and fix any fact the body
contradicts. For example, if `_downsample_mask` does not keep an empty mask empty, drop
that bullet.

- [ ] **Step 3: `_locate_candidates`.**
  - Delete the trailing `# ...` comment on each parameter line and on the return annotation.
  - Docstring:

```python
    """
    Target patches that match the reference both ways.

    - forward: positive cosine to the prototype (else the top 10% by similarity)
    - backward: the patch's nearest reference patch lies inside the reference mask

    Args:
        tgt_feat_deb: (D, Ht, Wt) debiased target features.
        ref_feat_deb: (D, Hr, Wr) debiased reference features.
        ref_mask_down: (Hr, Wr) bool reference mask at patch resolution.
        prototype: (D,) unit-norm reference prototype.

    Returns:
        (Ht, Wt) bool candidate mask.
    """
```

- [ ] **Step 4: `_seed_and_aggregate`.**
  - Delete the trailing parameter comments.
  - Docstring:

```python
    """
    Pick the seed cluster, then merge clusters whose combined score clears the threshold.

    - seed: the candidate-overlapping cluster most similar to the reference prototype
    - score: cross-image similarity x similarity to the seed x candidate overlap share

    Args:
        candidate_mask: (H_p, W_p) bool from `_locate_candidates`.
        tgt_feat: (D, H_p, W_p) target features, not debiased.
        tgt_feat_deb: (D, H_p, W_p) debiased target features.
        prototype: (D,) debiased reference prototype.
        cluster_labels: (H_p, W_p) cluster id per patch.
        K: number of clusters.
        merge_threshold: minimum combined score to join the mask.

    Returns:
        (H_p, W_p) bool; empty when no cluster overlaps the candidates.
    """
```

- [ ] **Step 5: Method summaries**
  - `set_context`: `Cache the reference features and prototype; call before segment().`
  - `segment_with_mask`: `One-shot segmentation: set_context, segment, clear_context.`
  - Keep their Args, lower-cased per contract.

- [ ] **Step 6: Prove, gate, commit**

```bash
cd $WT && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final collab_splats/semantics/segmentation/insid3.py
```

Expected: `PROSE ONLY`. Run gate `G`, then:

```bash
cd $WT && P=collab_splats/semantics/segmentation/insid3.py && git add $P && git commit --only $P -m "docs(semantics): insid3 prose

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: `sky.py` prose

**Files:**
- Modify: `collab_splats/semantics/segmentation/sky.py`

- [ ] **Step 1: Module docstring.**
  - Keep the model source.
  - Replace the "brightness detector" lore with this bullet:
    ```
    - used instead of VGGT's skyseg (facebookresearch/vggt @ a288dd0f14786c93483e45524328726ab7b1b4ce,
      visual_util.py:365-434), which tracks brightness more than sky
    ```
    Wrap it to stay within 100 chars.
  - Remove any `measured`, per-frame timing or container-specific wording.

- [ ] **Step 2: Comments**
  - Lines 60-61 become `# CUDA first when available; naming a missing provider only warns, so filter`
  - Lines 138-139 become `# Validate every wanted index up front, so a warm cache rejects junk too`
  - Lines 146-149 become:

```python
    # Segment only the cache misses, one frame at a time
    # - segment() opens the path itself, so no decode step here
    # - frames.read_frames would stack every miss in RAM at once
```

- [ ] **Step 3: `sky_masks` bullets.** Replace the two bullets with:

```
    - segments only frames without a cached PNG under cache_dir
    - cached PNGs store 255 for sky, inverted from skyseg's 255-is-NOT-sky files
```

- [ ] **Step 4: Prove, gate, commit**

```bash
cd $WT && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final collab_splats/semantics/segmentation/sky.py
```

Expected: `PROSE ONLY`. Run gate `G`, then:

```bash
cd $WT && P=collab_splats/semantics/segmentation/sky.py && git add $P && git commit --only $P -m "docs(semantics): sky prose

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Round-1 whole-package proof**

```bash
cd $WT && /opt/venv/reconstruction/bin/python $SP/prose_proof.py clean/final $(git ls-files 'collab_splats/semantics/*.py')
```

Expected: `PROSE ONLY`. Also run
`git diff --stat clean/final -- collab_splats/semantics`. It must list only files that
Tasks 1-7 touched.

---

## Round 2 — code

One commit per task, each one test first. Run the new test and watch it fail for the stated
reason, implement, then run gate `G`. The commit scope is `refactor(semantics):` when
behavior is unchanged and `fix(semantics):` when behavior changes.

### Task 8: `tokens_to_feature_map` → private, `assert` → `ValueError`

**Files:**
- Modify: `collab_splats/semantics/utils.py` (`__all__`, `tokens_to_feature_map`)
- Modify: `collab_splats/semantics/__init__.py:34,49` (import and `__all__` entry)
- Modify: `collab_splats/semantics/features/dino.py:9`, `talk2dino.py:12`, `maskclip.py:10` (imports and call sites)
- Test: `tests/semantics/test_extractor_preprocessing.py:16-43`, `tests/semantics/test_semantics_utils.py:128-137`

- [ ] **Step 1: Update the tests first**

In `test_extractor_preprocessing.py`:
- import line 16 becomes `from collab_splats.semantics.utils import _tokens_to_feature_map`
- rename every call in the three tests to `_tokens_to_feature_map`
- the wrong-count test becomes:

```python
def test_tokens_to_feature_map_wrong_count_raises():
    with pytest.raises(ValueError, match="Expected 196 tokens"):
        _tokens_to_feature_map(torch.randn(99, 8), 196, 196, 14)
```

In `test_semantics_utils.py`, replace `test_tokens_to_feature_map_is_public` with:

```python
def test_tokens_to_feature_map_is_private():
    """Only the three feature backends call it; it is not user surface."""
    assert "tokens_to_feature_map" not in su.__all__
    assert not hasattr(su, "tokens_to_feature_map")
    assert callable(su._tokens_to_feature_map)
```

- [ ] **Step 2: Run the tests; they fail**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py tests/semantics/test_semantics_utils.py -q -p no:cacheprovider
```

Expected: collection error `ImportError: cannot import name '_tokens_to_feature_map'`.

- [ ] **Step 3: Implement.** In `utils.py`, rename the function. Its body and `Raises:` become:

```python
def _tokens_to_feature_map(tokens: torch.Tensor, input_h: int, input_w: int, patch_size: int) -> torch.Tensor:
    """
    Reshape (N, D) patch tokens to (D, H_p, W_p), L2-normalized along the channel dim.

    Args:
        tokens: (N, D) patch tokens, N == (input_h // patch_size) * (input_w // patch_size).
        input_h: preprocessed image height in pixels.
        input_w: preprocessed image width in pixels.
        patch_size: pixel stride of one patch token.

    Returns:
        (D, H_p, W_p) feature map with unit-norm patch vectors.

    Raises:
        ValueError: when the token count does not match the patch grid.
    """
    ph = input_h // patch_size
    pw = input_w // patch_size
    if tokens.shape[0] != ph * pw:
        raise ValueError(
            f"Expected {ph * pw} tokens for {input_h}x{input_w} (patch_size={patch_size}), got {tokens.shape[0]}"
        )
    feat = tokens.reshape(ph, pw, -1).permute(2, 0, 1)
    return F.normalize(feat, dim=0)
```

Then:
- drop `"tokens_to_feature_map"` from `utils.__all__`
- in `semantics/__init__.py`, drop the import and its `__all__` entry
- in `dino.py`, `talk2dino.py` and `maskclip.py`, import and call `_tokens_to_feature_map`

- [ ] **Step 4: Grep for stragglers**

```bash
cd $WT && grep -rn "\btokens_to_feature_map\b" collab_splats tests docs/source configs evals scripts
```

Expected: no hits. If there are hits outside the allowed paths, stop and list them in the
report as deferred.

- [ ] **Step 5: Gate, commit**

```bash
cd $WT && P="collab_splats/semantics/utils.py collab_splats/semantics/__init__.py collab_splats/semantics/features/dino.py collab_splats/semantics/features/talk2dino.py collab_splats/semantics/features/maskclip.py tests/semantics/test_extractor_preprocessing.py tests/semantics/test_semantics_utils.py" \
  && git add $P && git commit --only $P -m "refactor(semantics): private _tokens_to_feature_map raises ValueError

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 9: `cache_store_path` raises on an ambiguous directory

**Files:**
- Modify: `collab_splats/semantics/utils.py` (`cache_store_path`)
- Test: `tests/semantics/test_semantics_utils.py`

- [ ] **Step 1: Failing test** (append):

```python
def test_cache_store_path_raises_on_two_stores(tmp_path):
    """Two 2D stores leave the extractor ambiguous; picking one silently is wrong."""
    for name in ("dinov2.zarr", "talk2dino.zarr", "talk2dino_lifted.zarr"):
        (tmp_path / name).mkdir()
    with pytest.raises(ValueError, match="dinov2.zarr.*talk2dino.zarr"):
        su.cache_store_path(tmp_path)


def test_cache_store_path_ignores_lifted_store(tmp_path):
    (tmp_path / "dinov2.zarr").mkdir()
    (tmp_path / "dinov2_lifted.zarr").mkdir()
    assert su.cache_store_path(tmp_path) == tmp_path / "dinov2.zarr"
```

- [ ] **Step 2: Run the first test; it fails with `DID NOT RAISE`**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_semantics_utils.py -k cache_store_path -q -p no:cacheprovider
```

- [ ] **Step 3: Implement.** The body after `sem_dir = Path(semantics_dir)` becomes:

```python
    # `*.zarr` also matches the lifted store; the suffix is the only thing separating them
    stores = sorted(p for p in sem_dir.glob("*.zarr") if not p.name.endswith("_lifted.zarr"))
    if not stores:
        raise FileNotFoundError(f"no 2D feature cache (*.zarr) in {sem_dir} — extract this scene's semantics first")
    if len(stores) > 1:
        raise ValueError(f"more than one 2D feature cache in {sem_dir}: {[p.name for p in stores]}")
    return stores[0]
```

Add `ValueError: when the dir holds more than one cache store.` under `Raises:`.

- [ ] **Step 4: Gate, commit**

```bash
cd $WT && P="collab_splats/semantics/utils.py tests/semantics/test_semantics_utils.py" && git add $P && git commit --only $P -m "fix(semantics): cache_store_path raises on more than one store

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 10: Narrow the store `except` clauses; per-frame debug log

**Files:**
- Modify: `collab_splats/semantics/utils.py` (module constants, `extract_feature_cache`, `point_features_cached`)
- Test: `tests/semantics/test_semantics_utils.py`

`_UNREADABLE_STORE` is `(OSError, ValueError, KeyError, TypeError)`. These are the errors that
zarr 3.1.5 raises for a missing, corrupt or half-written store:
- a missing file is `FileNotFoundError`, which is an `OSError`
- bad JSON and a missing group raise `ValueError` subclasses
- a missing attr raises `KeyError`
- `int(None)` raises `TypeError`

Anything else is a real bug and must propagate. The constant is a tuple of types, not a number,
so decision 017 allows it at module level.

- [ ] **Step 1: Failing tests** (append):

```python
def test_extract_feature_cache_propagates_unexpected_errors(tmp_path, monkeypatch):
    """A bug inside the validity check must surface, not trigger a silent re-extract."""
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / "frame_000000.png")
    (tmp_path / "fake.zarr").mkdir()

    def boom(*args, **kwargs):
        raise RuntimeError("not a store error")

    monkeypatch.setattr(su.zarr, "open", boom)
    extractor = MagicMock()
    extractor.name = "fake"
    with pytest.raises(RuntimeError, match="not a store error"):
        su.extract_feature_cache(extractor, images, tmp_path)


def test_extract_feature_cache_reextracts_corrupt_store(tmp_path):
    """A store with unreadable metadata is an expected stale cache: re-extract."""
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / "frame_000000.png")
    store = tmp_path / "fake.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{not json")

    extractor = MagicMock()
    extractor.name = "fake"
    extractor.patch_size = 2
    extractor.forward.return_value = [torch.zeros(3, 2, 2)]
    su.extract_feature_cache(extractor, images, tmp_path)
    assert zarr.open(str(store), mode="r").attrs["n_frames"] == 1


def test_point_features_cached_propagates_unexpected_errors(tmp_path, monkeypatch):
    """The predicate answers False for an unreadable store, but a bug still raises."""
    monkeypatch.setattr(su, "find_lifted_extractor", lambda d: "fake")
    monkeypatch.setattr(su, "lifted_store_path", lambda d, e: tmp_path / "fake_lifted.zarr")

    def boom(*args, **kwargs):
        raise RuntimeError("not a store error")

    monkeypatch.setattr(su.zarr, "open", boom)
    with pytest.raises(RuntimeError, match="not a store error"):
        su.point_features_cached(tmp_path)
```

Add `from unittest.mock import MagicMock` and `from PIL import Image` to the imports at the top
of the file if they are missing.

Before writing the tests, check how `point_features_cached` and `find_lifted_extractor` are
called. If `ae_path` would find a file under `tmp_path`, make sure it does not (use a fresh
`tmp_path`).

- [ ] **Step 2: Run; the two propagation tests fail** (the broad `except` swallows the
`RuntimeError`). The corrupt-store test passes both before and after; it pins the expected path.

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_semantics_utils.py -k "propagates or corrupt" -q -p no:cacheprovider
```

- [ ] **Step 3: Implement.** Below `logger = ...` in `utils.py`, add:

```python
# Errors a missing, corrupt or half-written zarr store raises; anything else is a bug
_UNREADABLE_STORE = (OSError, ValueError, KeyError, TypeError)
```

Then:
- in `extract_feature_cache`, change `except Exception:` to `except _UNREADABLE_STORE:`
- in `point_features_cached`, change `except Exception:` to `except _UNREADABLE_STORE:`, and keep the one-line comment from Task 1
- in the frame loop, replace the `if i % 10 == 0: logger.info(...)` pair with:

```python
        logger.debug("extract_feature_cache: %d/%d frames written", i + 1, N)
```

- [ ] **Step 4: Gate, commit**

```bash
cd $WT && P="collab_splats/semantics/utils.py tests/semantics/test_semantics_utils.py" && git add $P && git commit --only $P -m "fix(semantics): store checks catch only unreadable-store errors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 11: `FeatureAutoencoder.load` requires the metric keys

**Files:**
- Modify: `collab_splats/semantics/compression.py` (`load`)
- Test: `tests/semantics/test_compression_target.py:88-116`

- [ ] **Step 1: Update the tests**
  - Rename `test_load_legacy_checkpoint_without_metrics` to `test_load_checkpoint_with_literal_keys`.
  - Add `"recon_cosine": 0.5, "recon_mse": 0.25, "epochs_run": 3,` to its payload.
  - Change the three asserts to `== 0.5`, `== 0.25` and `== 3`.
  - Keep its comment about the literal state_dict keys.
  - Update its docstring to `A checkpoint's literal keys still load; two dead payload keys are ignored.`
  - Then append:

```python
def test_load_raises_without_metric_keys(tmp_path):
    """A checkpoint without fit metrics raises instead of reporting an untrained fit."""
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    payload = {"input_dim": 32, "latent_dim": 8, "state_dict": ae.state_dict()}
    weights = tmp_path / "old_ae.pt"
    torch.save(payload, weights)
    with pytest.raises(KeyError, match="recon_cosine"):
        FeatureAutoencoder.load(weights)
```

- [ ] **Step 2: Run; the new test fails with `DID NOT RAISE`**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_compression_target.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement.** In `load`, delete the `# Checkpoints predating ...` comment and write:

```python
        ae.recon_cosine = payload["recon_cosine"]
        ae.recon_mse = payload["recon_mse"]
        ae.epochs_run = payload["epochs_run"]
```

Add to the docstring:

```
        Raises:
            KeyError: when the checkpoint lacks a key `save` writes (older checkpoints must be refit).
```

- [ ] **Step 4: Gate, commit**

```bash
cd $WT && P="collab_splats/semantics/compression.py tests/semantics/test_compression_target.py" && git add $P && git commit --only $P -m "fix(semantics): FeatureAutoencoder.load raises on missing fit metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 12: Hoist `forward` and `device` into `BaseFeatureExtractor`

**Files:**
- Modify: `collab_splats/semantics/features/base.py`, `dino.py`, `talk2dino.py`, `maskclip.py`
- Test: `tests/semantics/test_extractor_preprocessing.py`

The spec says the prefix-token count is derived as `N - H_p * W_p`, so the base keeps the last
`H_p * W_p` tokens. Each backend's `_patch_tokens` returns the raw sequence:
- DINOv2 has 1 prefix token (CLS)
- DINOv3 Talk2DINO has 5 (CLS plus 4 registers)
- MaskCLIP has 0

`_patch_tokens` is **not** abstract. It raises `NotImplementedError`, because test doubles
(`_FakeExtractor`, the dashboard canary mocks) override `forward` only and must still
instantiate.

- [ ] **Step 1: Failing test** (append to `test_extractor_preprocessing.py`):

```python
class _TokenBackend(BaseFeatureExtractor):
    """Minimal backend: `_patch_tokens` returns `prefix` marker tokens, then the patch tokens."""

    def __init__(self, prefix: int):
        super().__init__(resize_mode="square", image_resolution=28)
        self.patch_size = 14
        self._normalize = T.Normalize([0.0] * 3, [1.0] * 3)
        self._device = torch.device("cpu")
        self.prefix = prefix
        self.patches = torch.randn(1, 4, 8)

    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.full((1, self.prefix, 8), 99.0), self.patches], dim=1)


@pytest.mark.parametrize("prefix", [0, 1, 5])
def test_base_forward_drops_prefix_tokens(prefix):
    """The base keeps the last H_p * W_p tokens, whatever the backend's prefix count."""
    backend = _TokenBackend(prefix)
    [feat] = backend.forward([Image.new("RGB", (28, 28))])
    expected = torch.nn.functional.normalize(backend.patches[0].reshape(2, 2, 8).permute(2, 0, 1), dim=0)
    assert feat.shape == (8, 2, 2)
    assert torch.allclose(feat, expected)


def test_base_device_property_reads_device():
    assert _TokenBackend(0).device == torch.device("cpu")
```

`_TokenBackend` is a class only because `BaseFeatureExtractor` requires subclassing. The tests
stay flat functions.

- [ ] **Step 2: Run; it fails** because `BaseFeatureExtractor.forward` is abstract, so
instantiating `_TokenBackend` raises `TypeError: Can't instantiate abstract class`.

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py -k "prefix or device_property" -q -p no:cacheprovider
```

- [ ] **Step 3: Implement the base.** In `features/base.py`, replace the abstract `forward`
(lines 71-82) with:

```python
    @property
    def device(self) -> torch.device:
        """
        Device the backend's model was moved to at construction.

        Returns:
            `self._device`, set by each backend's `__init__`.
        """
        return self._device

    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone on a preprocessed batch.

        Args:
            batch: (B, C, H, W) preprocessed images on `self.device`.

        Returns:
            (B, N, D) tokens; the last H_p * W_p are the patch tokens, in raster order.

        Raises:
            NotImplementedError: in a backend that does not override it.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _patch_tokens")

    def forward(self, images: list) -> list[torch.Tensor]:
        """
        Preprocess, run the backbone, and reshape each image's patch tokens to a grid.

        - prefix tokens (CLS, registers) are dropped: only the last H_p * W_p are kept

        Args:
            images: anything `open_image` accepts, one entry per frame.

        Returns:
            One (D, H_p, W_p) float32 CPU tensor per image, unit-norm per patch.
        """
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess and batch; torch.stack needs every image at one grid
        preprocessed = [self.preprocess(img) for img in images]
        batch = torch.stack(preprocessed).to(self.device)

        with torch.no_grad():
            tokens_all = self._patch_tokens(batch)

        # Keep the last H_p * W_p tokens and reshape each image to (D, H_p, W_p)
        results = []
        for i, t in enumerate(preprocessed):
            _, H, W = t.shape
            n_patches = (H // self.patch_size) * (W // self.patch_size)
            results.append(_tokens_to_feature_map(tokens_all[i, -n_patches:].cpu(), H, W, self.patch_size))
        return results
```

In the same file:
- remove `abstractmethod` from the imports only if nothing else uses it (`encode_text` still does)
- add `from collab_splats.semantics.utils import _tokens_to_feature_map` next to the existing `compute_semantic_contrast` import
- the `BaseQueryableExtractor` docstring bullet becomes `- subclasses implement \`encode_text\` and \`_patch_tokens\``

- [ ] **Step 4: Implement the backends.** In each backend:
  - delete its `forward` and its `device` property
  - delete the `_tokens_to_feature_map` import and any section dividers left empty
  - add the method below

`dino.py` (also delete `self.model_name = model_name`, which is never read):
```python
    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        DINOv2 last hidden state: CLS token, then patch tokens.

        Args:
            batch: (B, C, H, W) preprocessed images.

        Returns:
            (B, 1 + H_p * W_p, D) tokens.
        """
        return self.model(batch).last_hidden_state
```

`talk2dino.py`:
```python
    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Backbone forward_features, bypassing encode_image's internal square resize.

        Args:
            batch: (B, C, H, W) preprocessed images.

        Returns:
            (B, prefix + H_p * W_p, D) tokens; prefix is CLS plus registers.
        """
        return self._model.model.forward_features(batch)
```

`maskclip.py`:
```python
    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        MaskCLIP patch encodings, unit-norm per token, float32.

        Args:
            batch: (B, C, H, W) preprocessed images.

        Returns:
            (B, H_p * W_p, D) tokens; no CLS.
        """
        return F.normalize(self.model.get_patch_encodings(batch).to(torch.float32), dim=-1)
```

Drop any import these deletions leave unused, e.g. `logging`/`logger` if nothing else in the
file logs. Check with
`/opt/venv/reconstruction/bin/python -m pyflakes <file>` if pyflakes is present; otherwise grep.

- [ ] **Step 5: Existing-test check.** `test_extractor_preprocessing.py` builds fake backends with
`__new__`. Any of them that calls `forward` must now set `_device` and mock the model the new way:
- `model(batch).last_hidden_state` for DINOv2
- `_model.model.forward_features` for Talk2DINO
- `model.get_patch_encodings` for MaskCLIP

Fix only the fakes, never the assertions. If an assertion must change, stop and report it.

- [ ] **Step 6: Grep, gate, commit**

```bash
cd $WT && grep -rn "def forward\|def device\|model_name = model_name" collab_splats/semantics/features
```

Expected: exactly one `def forward` and one `def device`, both in `base.py`.

```bash
cd $WT && P="collab_splats/semantics/features/base.py collab_splats/semantics/features/dino.py collab_splats/semantics/features/talk2dino.py collab_splats/semantics/features/maskclip.py tests/semantics/test_extractor_preprocessing.py" \
  && git add $P && git commit --only $P -m "refactor(semantics): one forward and device on the feature base

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 13: `debias_validated` attribute, `get_bias_visualization` inline, `score_queries` log order

**Files:**
- Modify: `collab_splats/semantics/features/base.py`, `dino.py`, `talk2dino.py`
- Test: `tests/semantics/test_positional_debiasing.py`, `tests/semantics/test_query_api.py`

- [ ] **Step 1: Tests.**
  - In `test_positional_debiasing.py`, drop `_DEBIAS_VALIDATED` from the line-16 import.
  - `_UnvalidatedExtractor` still warns, because the attribute defaults to False. Keep it.
  - Append:

```python
def test_debias_validated_flags():
    from collab_splats.semantics.features.dino import DINOFeatureExtractor
    from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
    from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

    assert DINOFeatureExtractor.debias_validated is True
    assert Talk2DinoExtractor.debias_validated is True
    assert MaskCLIPExtractor.debias_validated is False


def test_debias_validated_silences_warning(caplog):
    class _Validated(_FakeExtractor):
        debias_validated = True

    ext = _Validated()
    with caplog.at_level("WARNING"):
        ext.debias(ext.forward([Image.new("RGB", (28, 28))]))
    assert "not yet validated" not in caplog.text


def test_get_bias_visualization_matches_features_to_rgb():
    ext = _FakeExtractor()
    [feat] = ext.forward([Image.new("RGB", (28, 28))])
    ext.debias([feat])
    _, H_p, W_p = feat.shape
    expected = ext.features_to_rgb(ext._zero_feats_cache[(H_p, W_p)])
    assert np.array_equal(ext.get_bias_visualization(H_p, W_p), expected)
```

Adapt the constructor call and image size to however `_FakeExtractor` is built in that file;
read it first. In `test_query_api.py`, append:

```python
def test_score_queries_logs_default_negative(caplog):
    """The debug log reports the negative set actually used, including the default."""
    ext = _make_queryable()  # use this file's existing queryable fake; name it as the file does
    with caplog.at_level("DEBUG", logger="collab_splats.semantics.features.base"):
        ext.score_queries(torch.randn(4, 2, 2), ["cat"])
    assert "1 negative" in caplog.text
```

If `test_query_api.py` has no queryable fake with `encode_text`, build one inline as a
`BaseQueryableExtractor` subclass whose `encode_text` returns `F.normalize(torch.randn(len(texts), 4), dim=-1)`.

- [ ] **Step 2: Run.**
  - `test_debias_validated_flags` fails with `AttributeError`.
  - `test_score_queries_logs_default_negative` fails because the log omits the negative count when `negative is None`.
  - The visualization test passes before and after; it pins the inline.

- [ ] **Step 3: Implement**
  - Delete the `_DEBIAS_VALIDATED` comment and constant, and add a class attribute on `BaseFeatureExtractor` under `_registry`:
    ```python
        # Positional debiasing checked on this backend; unchecked backends warn in debias()
        debias_validated: bool = False
    ```
  - In `debias`, the condition becomes `if not self.debias_validated:`.
  - Set `debias_validated = True` as a class attribute on `DINOFeatureExtractor` and `Talk2DinoExtractor`.
  - Replace the `get_bias_visualization` body after the `KeyError` guard with:
    ```python
            return self.features_to_rgb(self._zero_feats_cache[(H_p, W_p)])
    ```
  - In `score_queries`, move `if negative is None: negative = ["object"]` above `logger.debug(...)`.

- [ ] **Step 4: Grep, gate, commit**

```bash
cd $WT && grep -rn "_DEBIAS_VALIDATED" collab_splats tests
```

Expected: no hits.

```bash
cd $WT && P="collab_splats/semantics/features/base.py collab_splats/semantics/features/dino.py collab_splats/semantics/features/talk2dino.py tests/semantics/test_positional_debiasing.py tests/semantics/test_query_api.py" \
  && git add $P && git commit --only $P -m "refactor(semantics): debias_validated attribute; bias view reuses features_to_rgb

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 14: `maskclip_onnx` import names its install command

**Files:**
- Modify: `collab_splats/semantics/features/maskclip.py`
- Test: `tests/semantics/test_extractor_preprocessing.py`

- [ ] **Step 1: Failing test**

```python
def test_maskclip_missing_dep_names_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "maskclip_onnx", None)
    with pytest.raises(ImportError, match="pip install"):
        MaskCLIPExtractor(device="cpu")
```

Add `import sys` to the imports at the top of the file.

- [ ] **Step 2: Run; it fails.** The bare import raises an `ImportError` with no install hint.

- [ ] **Step 3: Implement.** Replace the bare import with the code below. Before writing it,
check the install line against `pyproject.toml`/`setup.sh`, e.g.
`grep -n maskclip pyproject.toml setup.sh`, and use the exact spec found there.

```python
        # Imported here: optional dep, needs setuptools<71 (pkg_resources.packaging)
        try:
            import maskclip_onnx  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "MaskCLIPExtractor needs maskclip_onnx: pip install <spec from pyproject> 'setuptools<71'"
            ) from e
```

- [ ] **Step 4: Gate, commit** — `fix(semantics): maskclip import error names the install command`
(paths: `maskclip.py`, the test file).

### Task 15: Segmentation base — `min_visible_frac`, empty raises, `convert_matched_mask` raises

**Files:**
- Modify: `collab_splats/semantics/segmentation/base.py`
- Test: `tests/semantics/test_segmentation.py`

- [ ] **Step 1: Tests**
  - Replace `test_create_composite_mask_truly_empty_results` with:

```python
def test_create_composite_mask_empty_results_raises():
    """No results means no shape; an empty (0, 0) mask would break every caller downstream."""
    with pytest.raises(ValueError, match="no results"):
        create_composite_mask([])
```

  - Append:

```python
def test_create_composite_mask_min_visible_frac():
    """The buried-mask floor is a kwarg: at 0.05, a 9.1%-visible mask survives."""
    small = np.zeros((3, 13), dtype=bool)
    small[:, 12] = True
    wide = np.zeros((3, 13), dtype=bool)
    wide[:, 0:11] = True
    medium = np.zeros((3, 13), dtype=bool)
    medium[:, 0:10] = True
    results = [
        {"segmentation": small, "predicted_iou": 0.99},
        {"segmentation": wide, "predicted_iou": 0.90},
        {"segmentation": medium, "predicted_iou": 0.95},
    ]
    assert create_composite_mask(results)[0, 10] == 0
    assert create_composite_mask(results, min_visible_frac=0.05)[0, 10] != 0


def test_convert_matched_mask_label_count_mismatch_raises():
    from collab_splats.semantics.segmentation.base import convert_matched_mask
    import torch
    masks = np.array([[1, 2], [2, 0]])
    with pytest.raises(ValueError, match="labels"):
        convert_matched_mask(torch.tensor([0]), masks)
```

- [ ] **Step 2: Run; they fail**
  - `create_composite_mask([])` returns a `(0, 0)` mask, so `DID NOT RAISE`.
  - `min_visible_frac` gives `TypeError: unexpected keyword`.
  - `convert_matched_mask` raises `AssertionError`, not `ValueError`.

- [ ] **Step 3: Implement.**
  - The signature becomes `create_composite_mask(results: list[dict], confidence_threshold: float = 0.85, min_visible_frac: float = 0.1)`.
  - Add at the top of its body:

```python
    if not results:
        raise ValueError("create_composite_mask: no results to merge")
```

  - The no-survivors branch becomes `return np.zeros_like(results[0]["segmentation"], dtype=np.uint16)`.
  - `> 0.1` becomes `> min_visible_frac`.
  - Docstring changes:
    - Add Arg `min_visible_frac: drop a mask left with this share of its pixels or less.`
    - Change the bullet's `≤10%` to `≤min_visible_frac`.
    - Add `Raises: ValueError: when results is empty.`
  - In `convert_matched_mask`, replace the assert with:

```python
    if labels.shape[0] != np.max(masks):
        raise ValueError(f"{labels.shape[0]} labels for {int(np.max(masks))} mask IDs")
```

  plus `Raises: ValueError: when the label count differs from the highest mask ID.`
  - In `BaseSegmentation.segment`, the return annotation drops `| None`. Its Returns text drops
    ", and mobilesamv2 returns None outright when nothing is detected". Task 16 makes that true.

- [ ] **Step 4: Gate, commit** — `fix(semantics): composite mask raises on empty input; min_visible_frac kwarg`
(paths: `segmentation/base.py`, `tests/semantics/test_segmentation.py`).

### Task 16: MobileSAM — private loader, strategy check in `__init__`, `box_batch_size`, empty result

**Files:**
- Modify: `collab_splats/semantics/segmentation/mobile_sam.py`, `collab_splats/semantics/segmentation/__init__.py:20,36`
- Test: `tests/semantics/test_segmentation.py`

- [ ] **Step 1: Failing tests** (append):

```python
def _mobilesam_stub(strategy="object", box_batch_size=320):
    from unittest.mock import patch, MagicMock
    from collab_splats.semantics.segmentation import mobile_sam
    with patch.object(mobile_sam, "_load_mobile_sam", return_value=(MagicMock(), MagicMock(), MagicMock())):
        return mobile_sam.MobileSAMSegmentation(strategy=strategy, box_batch_size=box_batch_size)


def test_mobilesamv2_bad_strategy_raises_at_init():
    with pytest.raises(ValueError, match="Strategy 'grid'"):
        _mobilesam_stub(strategy="grid")


def test_mobilesamv2_no_detections_returns_empty_stack():
    seg = _mobilesam_stub()
    seg.object_model.return_value = []
    masks, results = seg.segment(np.zeros((5, 7, 3), dtype=np.uint8))
    assert masks.shape == (0, 5, 7)
    assert masks.dtype == torch.float32
    assert results == []


def test_mobilesamv2_auto_empty_returns_empty_stack(monkeypatch):
    from unittest.mock import MagicMock
    from collab_splats.semantics.segmentation import mobile_sam
    gen = MagicMock()
    gen.return_value.generate.return_value = []
    monkeypatch.setattr(mobile_sam, "SamAutomaticMaskGenerator", gen)
    seg = _mobilesam_stub(strategy="auto")
    masks, results = seg.segment(np.zeros((5, 7, 3), dtype=np.uint8))
    assert masks.shape == (0, 5, 7)
    assert results == []


def test_mobilesamv2_box_batch_size_stored():
    assert _mobilesam_stub(box_batch_size=16).box_batch_size == 16
```

Add `import torch` to the imports at the top of the file.

- [ ] **Step 2: Run; they fail** with `AttributeError: ... no attribute '_load_mobile_sam'`.

- [ ] **Step 3: Implement.** In `mobile_sam.py`:
  - Rename `load_mobile_sam` to `_load_mobile_sam`, keeping its docstring.
  - Change the module docstring's `load_mobile_sam` mention, if any is left after Task 5.
  - Replace `__init__` and `segment` with:

```python
    def __init__(
        self,
        strategy: str = "object",
        device: str = "cpu",
        mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2",
        box_batch_size: int = 320,
    ):
        if strategy not in ("object", "auto"):
            raise ValueError(f"Strategy '{strategy}' not supported. Available: ['object', 'auto']")
        self.seg_model, self.object_model, self.predictor = _load_mobile_sam(mobilesam_encoder_name, device)
        self.strategy = strategy
        self.box_batch_size = box_batch_size

    def segment(self, image: np.ndarray) -> tuple[torch.Tensor, list[dict]]:
        """
        Segment one frame with the configured strategy.

        Args:
            image: (H, W, 3) uint8 array.

        Returns:
            (masks, results): masks (N, H, W) float32, N = 0 when nothing is detected;
            results are the raw SAM dicts, one per mask.
        """
        if self.strategy == "object":
            return self._segment_object(image)
        return self._segment_auto(image)
```

  - Add `box_batch_size: boxes per SAM decoder call ("object" only).` to the class Args, and a
    `Raises: ValueError: if strategy is neither "object" nor "auto".` section.
  - Add the shared stacker below the loader section:

```python
def _stack_masks(results: list[dict], height: int, width: int) -> torch.Tensor:
    """
    Stack SAM result masks as float32, (0, H, W) when there are none.

    Args:
        results: SAM dicts with a "segmentation" (H, W) array.
        height: frame rows, for the empty stack.
        width: frame columns, for the empty stack.

    Returns:
        (N, H, W) float32.
    """
    if not results:
        return torch.zeros((0, height, width), dtype=torch.float32)
    return torch.stack([torch.tensor(m["segmentation"]).to(torch.float32) for m in results])
```

  - In `_segment_auto`:
    - delete the `if len(results) == 0: return None` block
    - end with `return _stack_masks(results, *image.shape[:2]), results`
    - set the return annotation to `tuple[torch.Tensor, list[dict]]`
    - the Returns doc becomes `(masks, results), N = 0 when the generator finds nothing.`
  - `_segment_object`:
    - drop the `batch_size` parameter and use `self.box_batch_size` in both `batch_iterator` calls
    - the no-boxes early exit becomes `return _stack_masks([], height, width), []`
    - delete the final `if len(results) == 0: return None`
    - end with `return _stack_masks(results, height, width), results`
    - update the annotation and docstring the same way as `_segment_auto`, and remove the `batch_size` Arg
  - In `segmentation/__init__.py`, import only `MobileSAMSegmentation`, and drop
    `"load_mobile_sam"` from `__all__`. Task 20 handles the package-level `semantics/__init__.py`.

Task 20 also edits `semantics/__init__.py`. Until then, that file's `load_mobile_sam` import
breaks, so fix its import line **in this task** as well (drop `load_mobile_sam` from both the
import and `__all__`). Otherwise the gate fails at collection.

- [ ] **Step 4: Grep, gate, commit**

```bash
cd $WT && grep -rn "\bload_mobile_sam\b" collab_splats tests docs/source configs evals scripts
```

Expected: no hits.

Commit `fix(semantics): mobilesam returns an empty stack, validates strategy at init` with these paths:
`segmentation/mobile_sam.py segmentation/__init__.py semantics/__init__.py tests/semantics/test_segmentation.py`.

### Task 17: SAM3 imports inside `__init__`

**Files:**
- Modify: `collab_splats/semantics/segmentation/sam3.py`
- Test: `tests/semantics/test_segmentation.py:160-190`

- [ ] **Step 1: Tests.** Rewrite `test_sam3_segment_with_text_interface` to fake the modules.
  - Replace its two `patch(...)` context managers with:

```python
    import sys
    import types

    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_image_model = MagicMock()
    proc = types.ModuleType("sam3.model.sam3_image_processor")
    proc.Sam3Processor = MagicMock(return_value=mock_processor)
    fake = {
        "sam3": types.ModuleType("sam3"),
        "sam3.model": types.ModuleType("sam3.model"),
        "sam3.model_builder": builder,
        "sam3.model.sam3_image_processor": proc,
    }
    with patch.dict(sys.modules, fake):
        seg = SAM3Segmentation(confidence_threshold=0.5)
```

  - Append:

```python
def test_sam3_missing_dep_names_install():
    import sys
    from unittest.mock import patch
    from collab_splats.semantics.segmentation import SAM3Segmentation
    with patch.dict(sys.modules, {"sam3": None}):
        with pytest.raises(ImportError, match="huggingface-cli login"):
            SAM3Segmentation()
```

- [ ] **Step 2: Run.** The rewritten interface test fails, because the module-level `None`
stubs ignore the fake modules and raise `ImportError`.

- [ ] **Step 3: Implement.**
  - Delete the module-level `try/except` and its comment.
  - `__init__` becomes:

```python
    def __init__(self, confidence_threshold: float = 0.5):
        # Imported here: optional, gated dependency
        try:
            from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
            from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "sam3 is not installed. Request access at https://huggingface.co/facebook/sam3, "
                "wait for approval, run `huggingface-cli login`, then install from "
                "https://github.com/facebookresearch/sam3"
            ) from e
        self._processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=confidence_threshold)
```

  - Add `Raises: ImportError: when sam3 is not installed.` to the class docstring.

- [ ] **Step 4: Gate, commit** — `refactor(semantics): sam3 imported at construction, not module load`
(paths: `sam3.py`, `tests/semantics/test_segmentation.py`).

### Task 18: INSID3 — kwargs, `set_context` raises, `try/finally`, dead branches

**Files:**
- Modify: `collab_splats/semantics/segmentation/insid3.py`
- Test: `tests/semantics/test_insid3_segmentation.py`

- [ ] **Step 1: Tests.**
  - Delete `test_cluster_prototypes_handles_missing_cluster` and
    `test_seed_and_aggregate_missing_cluster_no_crash`. They test branches this task removes as
    unreachable: labels come from the clustering of the same points, so every id in
    `0..K-1` is present and none is negative.
  - In `_make_seg_with_mock_extractor`, add `seg._fallback_quantile = 0.9`.
  - Replace `test_set_context_empty_mask_does_not_set_context` with:

```python
def test_set_context_empty_mask_raises_and_keeps_no_context():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="empty"):
        seg.set_context(ref_image, torch.zeros(48, 48, dtype=torch.bool))
    assert seg._prototype is None
```

  - Append:

```python
def test_segment_with_mask_clears_context_when_segment_raises():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)
    ref_mask[16:32, 16:32] = True
    seg.segment = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        seg.segment_with_mask(ref_image, ref_image, ref_mask)
    assert seg._prototype is None


def test_agglomerative_labels_cover_every_cluster():
    """Pins the invariant the deleted empty-cluster branches relied on."""
    from collab_splats.semantics.segmentation.insid3 import _agglomerative_clustering
    X = F.normalize(torch.randn(40, 16), p=2, dim=1)
    labels = _agglomerative_clustering(X, tau=0.6)
    K = int(labels.max()) + 1
    assert torch.equal(labels.unique(), torch.arange(K))


def test_locate_candidates_fallback_quantile():
    """With no positive-similarity patch, the fallback keeps the top (1 - q) share."""
    from collab_splats.semantics.segmentation.insid3 import _locate_candidates
    D = 4
    prototype = F.normalize(torch.ones(D), dim=0)
    tgt = -F.normalize(torch.rand(D, 4, 5) + 0.1, dim=0)
    ref = F.normalize(torch.rand(D, 2, 2), dim=0)
    mask = torch.ones(2, 2, dtype=torch.bool)
    few = _locate_candidates(tgt, ref, mask, prototype, fallback_quantile=0.9).sum()
    many = _locate_candidates(tgt, ref, mask, prototype, fallback_quantile=0.5).sum()
    assert 0 < few < many


def test_insid3_device_defaults_to_get_device(monkeypatch):
    from collab_splats.semantics.segmentation import insid3
    seen = {}
    monkeypatch.setattr(insid3, "get_device", lambda: "cpu")
    monkeypatch.setattr(insid3, "DINOFeatureExtractor", lambda **kw: seen.update(kw))
    insid3.INSID3Segmentation()
    assert seen["device"] == "cpu"
```

- [ ] **Step 2: Run; the new tests fail**
  - the empty mask only warns
  - `segment_with_mask` has no `finally`
  - `fallback_quantile` is an unexpected keyword
  - `get_device` is not in `insid3`
  - `test_agglomerative_labels_cover_every_cluster` passes both before and after; it pins the invariant

- [ ] **Step 3: Implement.**
  - Add `from collab_splats.utils.torch_utils import get_device`.
  - `_cluster_prototypes` loop body becomes `protos.append(F.normalize(X[labels == k].mean(dim=0), p=2, dim=0).unsqueeze(0))`. Its comment becomes `# Unit-norm mean per cluster; every id 0..K-1 has members`.
  - `_locate_candidates` gains a keyword `fallback_quantile: float = 0.9` (Arg: `fallback_quantile: share of patches below the fallback cut when no patch is positive.`). The fallback uses `torch.quantile(sim_fwd.float(), fallback_quantile)`. The docstring bullet becomes `(else the top (1 - fallback_quantile) share by similarity)`.
  - `_seed_and_aggregate`, with every label valid:
    - `matched_mask = candidate_mask`
    - `all_unique, all_counts = cluster_labels.unique(return_counts=True)`
    - `cross_sim[k] = fg_sim[cluster_labels == k].mean()`
    - replace the last three lines with `return combined[cluster_labels] > merge_threshold`
    - keep the `matched_mask.sum() == 0` early return
  - `__init__` becomes:

```python
    def __init__(
        self,
        svd_components: int = 500,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        fallback_quantile: float = 0.9,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = get_device()
        self._extractor = DINOFeatureExtractor(svd_components=svd_components, device=device)
        self._tau = tau
        self._merge_threshold = merge_threshold
        self._fallback_quantile = fallback_quantile
        self._prototype: torch.Tensor | None = None
        self._ref_feat_deb: torch.Tensor | None = None
        self._ref_mask_down: torch.Tensor | None = None
```

  Class Args changes:
  - add `fallback_quantile: forwarded to candidate localization; see _locate_candidates.`
  - `device` becomes `torch device; None picks one with get_device.`

  - `set_context`:
    - delete `feat_norm = F.normalize(...)` and pass `[feat]` to `debias`, since forward output is already unit-norm per patch
    - replace the empty-mask warning block with:

```python
        # An empty mask has no prototype; keep no stale context either
        if not mask_down.any():
            raise ValueError("set_context: ref_mask is empty at patch resolution; it must cover at least one patch")
```

  with `Raises: ValueError: when ref_mask covers no patch.`
  - `segment`: same redundancy. Delete `feat_norm = F.normalize(...)` and use `feat` where
    `feat_norm` was used (debias input, clustering input, `_seed_and_aggregate` arg). Pass
    `fallback_quantile=self._fallback_quantile` to `_locate_candidates`.
  - `segment_with_mask` body:

```python
        # Context lives for this call only, even when segment() raises
        self.set_context(ref_image, ref_mask)
        try:
            return self.segment(image)
        finally:
            self.clear_context()
```

  Add `Raises: ValueError: when ref_mask covers no patch.` to its docstring.

- [ ] **Step 4: Gate, commit** — `fix(semantics): insid3 empty context raises; kwargs; unreachable branches removed`
(paths: `insid3.py`, `tests/semantics/test_insid3_segmentation.py`).

### Task 19: `sky_masks` caches probability; `threshold` kwarg

**Files:**
- Modify: `collab_splats/semantics/segmentation/sky.py` (`sky_masks`)
- Test: `tests/semantics/test_sky_segmentation.py:230-241`

- [ ] **Step 1: Tests.**
  - Read `_ConstantBackend` and `_scene()` first.
  - Change the backend's raw map so that sky pixels carry 0.9 and non-sky pixels carry 0.0.
    If other tests depend on raw == mask, add a `raw_value` parameter instead.
  - Rename and rewrite `test_sky_masks_caches_to_disk_with_255_meaning_sky`:

```python
def test_sky_masks_caches_probability_as_8bit(...):  # keep the original fixture args
    # ... same setup as the original test ...
    cached = cv2.imread(str(cache_dir / "frame_000000.png"), cv2.IMREAD_GRAYSCALE)
    assert abs(int(cached[0, 0]) - 230) <= 1   # 0.9 * 255
    assert cached[15, 15] == 0
```

  Append:

```python
def test_sky_masks_rethresholds_cache_without_model(...):  # same fixtures as above
    """A new threshold re-reads the cached probability; the model is not called again."""
    # warm the cache once, then patch BaseSegmentation.get to fail if called
    # sky_masks(..., threshold=0.95) -> all False; threshold=0.5 -> the original sky region
    ...


def test_sky_masks_reads_old_binary_cache(tmp_path):
    """A 0/255 PNG from the old format thresholds to the same mask."""
    # write frame_000000.png as 0/255 by hand next to a one-frame images/ dir, then
    # sky_masks(images_dir, cache_dir=...) with a backend that fails if called
    ...
```

  Write out the `...` bodies with this file's existing `_scene()` and backend-patching helpers,
  so both tests are complete and runnable. The re-threshold test must assert that the model
  factory was not called, e.g. by making it a `MagicMock(side_effect=AssertionError)`.

- [ ] **Step 2: Run; they fail.** The cache holds 255, not 230, and `threshold` is an unexpected keyword.

- [ ] **Step 3: Implement.**
  - The signature gains `threshold: float = 0.5`, with the Arg `threshold: sky probability above which a pixel is sky; applied on read, so changing it needs no re-segmentation.`
  - The cached-PNG bullet becomes `cached PNGs store sky probability x 255; old 0/255 masks read the same`.
  - The miss loop becomes:

```python
        for idx in todo:
            _, meta = model.segment(by_idx[idx])
            prob8 = np.rint(np.clip(meta["raw"], 0.0, 1.0) * 255).astype(np.uint8)
            cv2.imwrite(str(cache_dir / f"frame_{idx:06d}.png"), prob8)
```

  - The read becomes:

```python
    # Threshold on read, so hits and misses share one path and a new threshold reuses the cache
    return np.stack(
        [cv2.imread(str(cache_dir / f"frame_{i:06d}.png"), cv2.IMREAD_GRAYSCALE) / 255.0 > threshold for i in wanted]
    )
```

  - Confirm that `meta["raw"]` is resized to the frame (`sky.py:98`), so the PNG has the frame's shape.
  - Check `tests/wrapper/test_mask_sky.py`. It is a canary and must stay green unchanged.

- [ ] **Step 4: Gate, commit** — `fix(semantics): sky_masks caches probability, thresholds on read`
(paths: `sky.py`, `tests/semantics/test_sky_segmentation.py`).

### Task 20: Package exports

**Files:**
- Modify: `collab_splats/semantics/__init__.py`, `collab_splats/semantics/segmentation/__init__.py`
- Test: `tests/semantics/test_semantics_utils.py`

- [ ] **Step 1: Failing test** (append):

```python
def test_package_exports_segmentation_backends():
    for name in ("INSID3Segmentation", "SkyWaterSegmentation", "sky_masks"):
        assert name in semantics.__all__ and hasattr(semantics, name), name
    for name in ("load_mobile_sam", "tokens_to_feature_map"):
        assert name not in semantics.__all__ and not hasattr(semantics, name), name
```

- [ ] **Step 2: Run; it fails** on `INSID3Segmentation`.

- [ ] **Step 3: Implement.** Import the three names from `.segmentation` in `semantics/__init__.py`
and add them to `__all__`, keeping the existing grouping. If `segmentation/__init__.py` does not
export them yet, export them there too.

- [ ] **Step 4: Gate, commit** — `feat(semantics): export insid3, skywater and sky_masks`
(paths: both `__init__.py` files and the test file).

### Task 21: Contract — release semantics

**Files:**
- Modify: `tests/test_docstring_contract.py` (`RELEASED`, ~line 194)

- [ ] **Step 1: Flip the switch**

```python
RELEASED: frozenset[str] = frozenset({"semantics"})
```

- [ ] **Step 2: Run the contract**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/test_docstring_contract.py -k semantics -q -p no:cacheprovider
```

Expected: 0 failed, and no xfail for any semantics file. Each failure names a file and a rule.
Fix it in that file's prose, prove it with `prose_proof.py HEAD <file>` (must print
`PROSE ONLY`), and commit that fix separately as `docs(semantics): ...` before this task's
commit.

- [ ] **Step 3: Gate, commit** — `test(semantics): release semantics under the docstring contract`
(path: `tests/test_docstring_contract.py`). A merge conflict with the preproc branch's own
`RELEASED` line is expected later; the resolution is the union.

### Task 22: Graph, stale-name sweep, report

- [ ] **Step 1: Stale-name sweep**

```bash
cd $WT && grep -rnE "\b(tokens_to_feature_map|load_mobile_sam|_DEBIAS_VALIDATED)\b" collab_splats tests docs/source configs evals scripts
```

Expected: no hits. `_tokens_to_feature_map` does not match `\btokens_…`, because `_` is a word
character.

- [ ] **Step 2: Graph** — `cd $WT && graphify update .`

- [ ] **Step 3: Final gate and report.** Run gate `G` once more, then report:
  - counts against `$SP/baseline.txt`, with each delta explained by task (adds, deletes, xfail→pass)
  - `git log --oneline clean/final..HEAD`
  - the deferred list from the spec, unchanged: `configs/base.yaml` lore, the tutorial 04 `None` check, the move of `tests/test_semantics_logging.py`, and the grouping revival
  - that nothing was merged or pushed

---

## Self-review (against the spec)

| Spec item | Task |
|---|---|
| Round 1 prose, per file | 1-7 (proof per task, whole-package proof at 7.5) |
| `tokens_to_feature_map` private, `ValueError` | 8 |
| `cache_store_path` raises on >1 | 9 |
| narrow `except` in `extract_feature_cache`, `point_features_cached`; per-frame debug | 10 |
| `FeatureAutoencoder.load` `KeyError` | 11 |
| hoist `forward` (`N - H_p*W_p`), one `device`, delete `model_name` | 12 |
| `debias_validated`, inline `get_bias_visualization`, `score_queries` default before log | 13 |
| `maskclip_onnx` `ImportError` with install | 14 |
| `convert_matched_mask` raise, `min_visible_frac`, empty raises, `segment` drops `| None` | 15 |
| `_load_mobile_sam`, strategy check at init, `box_batch_size`, empty `(0, H, W)` | 16 |
| sam3 import in `__init__` | 17 |
| insid3 `device=None`, `fallback_quantile`, `set_context` raises + no re-normalize, `try/finally`, dead branches | 18 |
| `sky_masks` probability cache, `threshold` | 19 |
| exports | 20 |
| `RELEASED` | 21 |
| graphify, stale names, report | 22 |

Notes:
- The spec's `sam3 segment prompt=` row was dropped from the spec before this plan. `segment` is
  the promptless interface and `segment_with_text` takes prompts.
- Task 18 also drops the redundant re-normalize in `segment()`. The spec names it only for
  `set_context`, but it is the same no-op on already unit-norm features.
