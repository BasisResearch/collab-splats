# Splats Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the accumulated overengineering from `collab_splats/splats/` — collapse the 8 `if anchor_field is not None` branches in `train()` into two model classes with an identical interface, merge four files into two, retire the redundant `splats.zarr` render cache in favor of a self-contained `ckpt.pt`, and delete two dead package directories — with byte-for-byte behavior preserved.

**Architecture:** Two model classes, `Gaussians` (vanilla 3dgs/2dgs primitives) and `Scaffold` (Scaffold-GS anchors + MLP heads), expose the *same* three attributes (`params`, `optimizers`, `schedulers`, plus the derived `n_primitives`) and six methods (`render`, `pre_backward`, `post_backward`, `denormalize`, `export_gaussians`, `checkpoint`) over one classmethod constructor (`from_checkpoint`). There is deliberately **no base class** — duck typing only, verified by one parametrised interface test. `train()` becomes a 3-phase function (setup / loop / finish) that never asks which representation it is training. Downstream consumers (mesh TSDF, evals) stop reading a pre-rendered zarr store and re-render on demand from `ckpt.pt`.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), PyTorch, upstream gsplat @ `d2f5c0f` (`rasterization`, `rasterization_2dgs`, `MCMCStrategy`, `DefaultStrategy`, `fully_fused_projection`), numpy, OpenCV, zarr 3.x (being removed from this module), pytest.

---

**Source spec:** `docs/superpowers/specs/2026-09-05-splats-cleanup-design.md`

**Worktree:** `/workspace/collab-splats/.worktrees/clean-splats` (branch `clean/splats`, forked from `refactor/cu121-uv-migration` at `90af13d9`)

---

## Ground Rules

Read this section before Task 1. Every task assumes it.

### 1. Where you work

**All work happens in the worktree** `/workspace/collab-splats/.worktrees/clean-splats` on branch `clean/splats`. Never edit files under `/workspace/collab-splats/` directly — that is the main checkout and other sessions are working in it.

Confirm before starting:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && pwd && git branch --show-current
```

Expected:

```
/workspace/collab-splats/.worktrees/clean-splats
clean/splats
```

### 2. The PYTHONPATH trap — read this or every test result is a lie

The venv at `/opt/venv/reconstruction` has `collab_splats` installed **editable, hardcoded to `/workspace/collab-splats`**. A bare `pytest` inside the worktree imports the *main tree's* code and reports a false green. `PYTHONPATH` alone is **not** enough — `sys.path[0]` is the cwd and resolves first, and the shell resets between tool calls.

**The only form that works** is a single command that both `cd`s and sets `PYTHONPATH`:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
```

Expected (this exact path — if it prints `/workspace/collab-splats/collab_splats/__init__.py` you are testing the wrong tree):

```
/workspace/collab-splats/.worktrees/clean-splats/collab_splats/__init__.py
```

**Every pytest command in this plan is written in that form.** Do not shorten them.

### 3. Test-run hygiene

- Never pipe pytest through `tail` — that throws away the exit code. If you need less output use `-q`.
- A run that exceeds the Bash timeout is **backgrounded, not killed**. Prefer `run_in_background: true` for anything over ~2 minutes rather than raising the timeout.
- When you background a run, echo `PYTEST_RC=$?` **immediately after** the pytest command — the task notification reports the exit code of the *trailing* command, not pytest's.
- ~~The shared venv currently holds the **gsplat 1.4.0 PyPI wheel** while `uv.lock` pins 1.5.3 @ `d2f5c0f`...~~ **STALE — corrected 2026-09-06.** The venv now holds the pinned upstream `1.5.3 @ d2f5c0f` and `gsplat.losses` imports. `tests/splats` collects and runs clean (332 passed at `010d027b`). Do not pass `--continue-on-collection-errors` to work around an error that no longer occurs; a collection error now means something is genuinely broken. Any baseline measured before this correction is void — re-measure it.

#### What "the wide gate" means on this branch — measured 2026-09-06 at `010d027b`

**The gate is a three-path scoped subset, not the whole `tests` tree:**

```bash
pytest tests/splats tests/mesh tests/test_cu121_migration.py -q -p no:randomly
```

Measured in a pinned tree at `010d027b` with `third_party/*` symlinked in:

| scope | result |
|---|---|
| `tests/splats` | 332 passed, 0 skipped, RC=0 |
| the three-path subset | 424 passed, 0 skipped, RC=0 |
| the full `tests` tree | **43 failed**, 2330 passed, 9 skipped, RC=1, 10m49s |

**The 43 are inherited and belong to other subsystems.** None of them are in `tests/splats` — the
same pinned tree gives 332 passed / 0 failed there. They are other sessions' in-flight work plus
known failures, and this branch neither caused them nor can fix them. **Do not run the full tree as
a gate**: it costs eleven minutes and returns 43 findings that must then be triaged away. This
applies to Task 19's final gate as much as to every task before it.

Corollary for arithmetic: 424 = 385 (the scoped baseline at `c86f7c3a`) + 15 (Task 14) + 24 (Task 10
fix round 5). When attributing a delta to your own task, subtract the tasks that landed between your
baseline and your commit — the branch moves under concurrent agents, and a total diffed against a
stale baseline overstates your contribution.

### 4. Formatting

- **Never** run repo-wide `black .` or `isort .` — the venv's black is newer than the repo's pinned version and will reformat hundreds of untouched files.
- Format only the files you touched, e.g.:
  ```bash
  cd /workspace/collab-splats/.worktrees/clean-splats && /opt/venv/reconstruction/bin/black collab_splats/splats/gaussian.py && /opt/venv/reconstruction/bin/isort collab_splats/splats/gaussian.py
  ```
- `pyproject.toml`'s `[tool.flake8]` block is **inert** — flake8 fires E501 at **79** characters, not 120. isort's `profile="black"` wraps imports at **88**. Write long imports parenthesized.

### 5. Committing under concurrent sessions

Other Claude sessions share this repository's git index. A bare `git add -A` / `git commit -a` will sweep their staged and unstaged work into your commit.

- **Always** `git commit --only <explicit paths>`.
- `docs/superpowers/` may be gitignored in this repo — use `git add -f docs/superpowers/...` when committing plan or changelog edits.
- Check `.git/sequencer` does not exist before committing (a stale rebase in another worktree).
- Never bare `git stash` / `git stash pop` — the stash stack is shared.

### 6. TDD discipline

Every task follows: write the failing test → run it and **watch it fail** → write the minimal implementation → run it and watch it pass → commit. A test that passes before the implementation exists is testing nothing; if that happens, the test is wrong — fix the test, do not proceed.

### 7. Behavior is frozen

This is a refactor. The following must not change:

| Frozen surface | Note |
| --- | --- |
| `SplatsConfig` field names, types, defaults | no new keys, no removed keys |
| The `splats:` block in `configs/base.yaml` | comments may be reworded, keys may not |
| `splats.ply` contents | byte-for-byte, and measured byte-identical under a fixed seed — but **gate at tolerance on the exported values**, not md5 or vertex count; see the note below |
| `splats_quality_report.json` schema | keys and nesting unchanged |
| `train()` numerics | PSNR within 1e-3 dB, ply means `allclose(rtol=1e-4)` |

`ckpt.pt` **does** change (gains `cam_to_world`, `intrinsics`, `image_ids`, `image_size`; drops `pose_adjust`) and `splats.zarr` is **retired**. Those two are the only intended output changes.

> **How to gate `splats.ply` — measured 2026-09-06, CORRECTED the same day. Read this before
> writing a ply assertion.**
> A first version of this note claimed the ply was not md5-stable and that "run0 differs" from
> CUDA warm-up. **That was wrong**, and it was wrong in a way worth recording: the probe ran its
> three trainings **inside one process**, so state carried between them. Re-measured with each
> run in its **own** process (scaffold, `pose_opt: True`, 3 steps, `tests.splats.synthetic.
> make_scene`, seeding `torch` + `torch.cuda` + `numpy` + `random`):
>
> | | md5 | byte count | ply vertices |
> |---|---|---|---|
> | seeded, 3 fresh processes | `2c7581ac…` all three | 55631 / 55631 / 55631 | 987 / 987 / 987 |
> | unseeded, 3 fresh processes | all three differ | 44375 / 65376 / 43703 | 786 / 1161 / 774 |
>
> So: **seeded, the ply IS byte-identical across processes** — §7's "byte-for-byte" row is
> literally true under a fixed seed. Unseeded it varies in length, because anchor visibility and
> the decoded-primitive count are redrawn.
>
> **The still-correct conclusion: gate at tolerance, not at md5** — `np.allclose(rtol=1e-4)` on
> the exported means plus exact vertex-count equality, which is what the parity harness does. Not
> because md5 fails, but because md5 is the weaker gate in the one case that matters: Task 10's
> code-quality review planted a `width`/`height` swap in the `export_gaussians` call and it left
> **both the byte count and the primitive count identical while changing the file**. A size check
> or a count check alone passes that mutant. Compare the exported values.
>
> **Do not read `summary["n_gaussians"]` as the ply vertex count on a scaffold.** It is
> `model.n_primitives`, which for `Scaffold` is the **anchor** count (`scaffold.py:544`) — 163
> here against 987 ply vertices, and it is identical seeded or unseeded, so it discriminates
> nothing. Read `element vertex N` out of the ply header instead.

> **CORRECTED 2026-09-06 by Task 10 fix round 2 — `means` alone does NOT kill the swap.**
> The paragraph above prescribes "`np.allclose(rtol=1e-4)` on the exported means plus exact
> vertex-count equality". Measured against the actual mutant (seeded scaffold, 3 views,
> `make_scene(height=24, width=40)`, 3 steps, `export_gaussians` re-called at `(40, 24)` and at
> the swapped `(24, 40)` off the same trained model):
>
> | array | shape | `allclose(rtol=1e-4, atol=1e-6)` right vs swapped |
> |---|---|---|
> | `means` | (1009, 3) | **True — identical** |
> | `opacities` | (1009,) | False |
> | `sh0` | (1009, 1, 3) | False |
> | `scales` | (1009, 3) | False |
> | `quats` | (1009, 4) | False |
>
> The vertex count is identical too (1009 both sides). So the prescribed gate — means plus count
> — **survives the very mutant it was written to catch**, and so does the parity harness's
> `means_head` comparison at line 430.
>
> Why: `Scaffold.export_gaussians` builds `means = (anchors + offsets * scaling[:, None, :3])`,
> which has no view-dependent term at all. Width and height reach only `visible_anchors` (which
> selects *which* anchors decode) and the `mean_direction` fed to the MLPs (which sets opacity,
> color, covariance). At this frame size the visible set and the `neural_opacity > 0` keep mask
> happen to be unchanged, so the same 1009 means come out in the same order while every decoded
> attribute moves.
>
> **The blindness is fixture-dependent, which makes it worse, not better — controller-verified
> 2026-09-06.** Re-run on `make_scene()`'s own default frame size instead of 24x40, the swap moves
> the *count*: 987 vertices right, 979 swapped, so every array differs in shape and a count check
> would have caught it there. Same mutant, same code, opposite tell. Whether the swap shows up as
> a value change or a count change depends on which anchors happen to fall inside the projected
> frame, so **neither a count check nor a means check is reliable on its own** — a gate that
> passes on one fixture can be blind on another. Compare the decoded values, at a pinned frame
> size, always.
>
> **The gate must therefore assert on the decoded attributes, not on `means`.** Task 10's B1 test
> `test_the_scaffold_ply_holds_the_values_exported_at_the_frame_size` compares all five arrays
> (`means`, `opacities`, `sh0`, `scales`, `quats`) read back through
> `gsplat.exporter.load_ply_to_splats`, against a re-export at hard-coded `(40, 24)` literals —
> an independent absolute anchor per §2b, since a swap at the call site cannot move a literal with
> it. It also asserts `len(written["means"]) > 1`, because the all-anchors-culled fallback path
> would otherwise make every comparison vacuous. **Measured: the swap SURVIVES the pre-fix
> 252-test suite (252 passed) and is KILLED by that one test post-fix.**
>
> The `train()` numerics row of the §7 table ("ply means `allclose(rtol=1e-4)`") inherits the same
> weakness and should be read as a *necessary, not sufficient* check.

> **The `no_grad` leak out of a suspended generator, and the audit rule it forces — controller-
> verified 2026-09-06.** `render_views` is a generator. A generator parked at a `yield` inside a
> `with torch.no_grad():` has **not** exited the context manager, and PyTorch's grad-mode flag is
> thread-global rather than frame-local — so while such a generator is alive and suspended,
> autograd is off for the **whole process**. Partial consumption is the trigger (a `zip` that
> ends first, a bare `next()` on a temporary, `islice`, `break`), and a FAILING test is what keeps
> the generator alive: pytest retains the traceback, the traceback retains the frame, the frame
> retains the generator. On a passing test the local dies at teardown and the context unwinds,
> which is why this hides until something else breaks. **Nothing shipped was numerically wrong:**
> measured at `b33a7443`, `train()` returns with grad still ENABLED, because `renders` is a local
> and CPython refcounting finalizes the suspended generator on return, running `__exit__`. The
> leak was confined to callers that keep a partially-consumed generator ALIVE — pytest's retained
> tracebacks being the live case. The fix is still worth making: it removes a latent hazard for
> the mesh-stage and notebook consumers Tasks 16/17 add. **Fix at the definition** —
> `@torch.no_grad()` as a decorator, never a `with` inside the body: PyTorch's
> `_DecoratorContextManager.__call__` detects a generator function and saves/restores grad mode
> around each `next()`. Call-site listifying is not a fix; `write_outputs` cannot listify (~14 GB
> at 1080p x 300 views). Measured with mutant M49 planted: on `b33a7443`, **4 failed** — the real
> kill plus three unrelated tests dying with `RuntimeError: element 0 of tensors does not require
> grad and does not have a grad_fn`; on `348d879b`, **1 failed, 276 passed**, the real kill alone.
>
> **The before-state count is NOT reproducible run to run.** On a byte-identical pinned tree at
> `b33a7443` the same mutant gave **2 failed / 266 passed** on a first run and **4 failed / 264
> passed** on three subsequent runs — total 268 every time, so it is the same tree; two tests flip
> pass/fail across runs. The collateral rides on GC timing of the traceback-retained generator,
> which is not deterministic. Only the FIXED state is stable: `348d879b` gave 1 failed / 276 passed
> on three consecutive runs. Deterministic kill attribution is the whole point of the fix, and it
> is what the fix delivers — but it means no single run against the pre-fix tree can be quoted as
> settled.
>
> **Two consequences for every mutation audit on this branch.** First, **`--tb=no` MASKS the
> phenomenon entirely** — it suppresses the traceback retention that keeps the generator alive, so
> an audit run with it reports zero collateral and reads as "there was nothing to fix". Never use
> it here. Second, collateral makes a mutant look killed by a test that never executed against it,
> which is fatal to an acceptance criterion built on kill *attribution*. All 22 round-2 mutants
> were re-audited under default tracebacks: every one has its intended, semantically-capable
> killer among the failures, so `4306078a`'s claim stands — but 3 of 23 produced collateral, so
> the risk was real, not hypothetical. **Read that table as "re-audited on a nondeterministic
> before-state", not as settled:** each of the 22 was run once, against the substrate the
> paragraph above shows flipping two tests between runs, so a mutant whose collateral did not
> surface in its one run could have been recorded clean. Nothing downstream rests on it — every
> post-fix mutant measured since killed exactly its intended test with zero collateral.
>
> **D8 belongs in the permanent mutant list:** `if model.primitive_unit == "anchors": pass`
> planted inside `train()`. It **survived** round 2's `test_trainer_has_no_representation_branches`
> (0 failed) because that guard's name set listed only class names — not `primitive_unit`, the
> discriminator round 2 itself introduced, and therefore the likeliest spelling of a future
> violation. `REPRESENTATION_NAMES` now also carries `primitive_unit`, `scaffold` and `vanilla`,
> and D8 is killed (1 failed). A `pytest.mark.parametrize` over that set **cannot guard the set's
> own width** — narrowing it generates *fewer* cases and never fails — so the set is additionally
> pinned by an absolute-literal assertion, the same §2b remedy round 2 needed for a widened
> `REPRESENTATIONS`.

> **AMENDED 2026-09-06 after Task 10's round-3 re-review — a UNION over clauses is the same
> vacuity defect as a parametrize over a set.** `_branch_test_names` collects into a FLAT set from
> three separate clauses (`ast.Name`, `ast.Attribute`, `ast.Constant`). Round 3's meta-test planted
> the name in all three spellings **in one condition**, so the intersection with
> `REPRESENTATION_NAMES` stayed non-empty whenever **any one** clause survived, and the loss of an
> individual clause was invisible. Measured in a pinned tree at `348d879b`: delete the
> `ast.Attribute` clause AND plant a real violation `if model.primitive_unit == "anchors":` inside
> `train()` -> **277 passed, `PYTEST_RC=0`**. A genuine representation branch ships green with the
> guard hollowed out. The comment above the test read as coverage and was precisely the mechanism
> of the gap.
>
> **Remedy: parametrize the SPELLING, one case per (name, clause)** — `c86f7c3a`, `tests/splats`
> 277 -> **293 passed**. Each clause is then individually load-bearing: dropping `ast.Name`,
> `ast.Attribute` or `ast.Constant` in turn gives **8 failed / 285 passed** each time, and the
> failing case names which clause died (`[<name>-identifier]` / `[-attribute]` / `[-literal]`).
> This is now the **third** instance of one defect class on this branch — a test whose structure
> makes it unable to fail: (1) a `parametrize` over a set cannot guard the set's own width,
> (2) a union over clauses cannot guard an individual clause, (3) a sequenced regex over an
> allow-list guards order rather than membership. The general rule: **when a guard aggregates N
> things, the test must have N cases, not one case over the aggregate.**
>
> The two tests stay separate and guard different failures — `..._set_is_not_narrowed` pins the
> SET's width with an absolute literal, `..._sees_every_representation_name` pins the PREDICATE's
> completeness per clause. Merging them reintroduces one of the two holes.
>
> *Correction to the third item:* the sequenced regex `must be one of .*vanilla.*scaffold.*` was
> claimed to break on a representation **inserted** between the two. Measured: it does **not** —
> the leading `.*` spans an inserted entry. Only a **reorder** breaks it, and separately it MISSES
> a message that names only 2 of a 3-entry allow-list. `c86f7c3a` replaced it with a membership
> loop over `REPRESENTATIONS`, which is strictly stronger, not merely looser.

> **FOURTH instance of the same defect class, found by Task 10's round-4 re-review — the guard's
> NODE-TYPE axis was unguarded.** `_representation_branches` matches
> `isinstance(child, (ast.If, ast.IfExp))`: a tuple of **two** node types. The meta-test
> `..._sees_every_representation_name` planted only a statement `if`, so it exercised the `ast.If`
> arm and never the `ast.IfExp` arm. **Measured at `c86f7c3a` in a pinned tree: deleting
> `ast.IfExp` from that tuple leaves the suite fully green — 293 passed, RC=0.** A representation
> branch written as a conditional expression — `x = a if model.primitive_unit == "anchors" else b`
> — is caught today, but nothing pinned that it stays caught. The re-review raised this as
> non-blocking and pre-existing (it measures identically at `348d879b`, so round 4 neither caused
> nor worsened it); it is recorded here as a fix rather than a backlog item because it is the same
> shape as (1)-(3) one level down, and the plan had just finished stating the rule that forbids it.
>
> **Remedy: a `structure` parametrize axis** (`"statement"` / `"ternary"`), planting each
> `(name, spelling)` pair both ways — `tests/splats` 293 -> **317 passed**. Numbers below were
> measured by the controller in a pinned tree at `c86f7c3a`; the change lands as **fix round 5**. Both arms are then
> individually load-bearing, measured: dropping `ast.IfExp` gives **24 failed / 293 passed** with
> every failure name ending `-ternary`; dropping `ast.If` gives **24 failed / 293 passed** with
> every failure name ending `-statement`. Zero collateral either way, so attribution is exact.
>
> **Why the class kept recurring, stated plainly:** each instance was a test whose *comment* read
> as coverage while its *structure* could not fail. That is the tell. The check is not "does this
> test look thorough" but **"what single edit to the thing under guard would this test fail to
> notice?"** — asked once per axis the guard aggregates over: the set's members, the predicate's
> clauses, the allow-list's entries, and now the node types. A guard over an N-tuple needs N cases.

> **AMENDED 2026-09-06 after Task 10's code-quality review — `unit` comes off the model.**
> This plan originally prescribed `unit = "anchors" if type(model).__name__ == "Scaffold" else
> "gaussians"`, and Task 10 shipped exactly that at `rendering.py:390`. The review found it is
> the one remaining place where `write_outputs` dispatches on the representation, three lines
> below `**model.frame_report(render)` — the very mechanism introduced to carry per-model facts.
> A `__name__` string compare is also strictly worse than a type check: it falls through to
> `"gaussians"` for any subclass and for any rename, silently.
>
> The prescribed shape is now a class attribute beside `frame_report` — `primitive_unit =
> "gaussians"` on `Gaussians`, `"anchors"` on `Scaffold` — read as `unit = model.primitive_unit`.
> Measured in a pinned tree: 252 passed, so the change is behaviorally free. Nothing pins the
> log line today (the mutant that hardcodes `unit` survives), so **the fix must land with a test**
> that a scaffold logs `anchors` and a vanilla model logs `gaussians`.
>
> **Contrast `trainer.py:375`**, which uses `type(model).__name__` as a log *label* with no
> branch. That one stays. The distinction is reflection-as-text versus reflection-as-dispatch.

> **ADDED 2026-09-06 by Task 10 fix round 2 — a rejection test cannot see a WIDENED allow-list.**
> `REPRESENTATIONS = tuple(MODEL_CLASSES)` and `SplatsConfig` rejects anything outside it. The
> obvious test — `pytest.raises(ValueError)` on `{"representation": "octree"}` — kills a *deleted*
> check but **survives an allow-list that grew a third name**: with `REPRESENTATIONS` widened to
> `("vanilla", "scaffold", "banana")`, "octree" is still rejected and all 268 tests pass. Measured
> as mutant M8: SURVIVED. Narrowing the list is caught only incidentally (61 unrelated tests go
> red), so neither direction is actually pinned by the rejection test.
>
> Same §2b shape as everything else on this branch: `set(REPRESENTATIONS) == set(MODEL_CLASSES)`
> is a *parity* assertion and a mutant that widens both sides walks through it. The fix is an
> absolute literal — `assert set(REPRESENTATIONS) == {"vanilla", "scaffold"}` — with the parity
> assertion kept beside it so the two cannot drift apart. M8 is KILLED after that.

### 8. Commit message style

Conventional commits with scope, e.g. `refactor(splats): extract scene helpers into utils.py`. Types in use: `refactor`, `feat`, `fix`, `test`, `docs`, `chore`.

### 9. Constants rule

The spec bans scattered module-level tunables. Every literal that a caller could reasonably want to change becomes a **keyword-only argument with the current value as its default**. The complete list (do not invent others, do not miss any):

| Where | Keyword args |
| --- | --- |
| `train` | `min_points=100`, `lr_decay=0.01` |
| `Gaussians.__init__` | `knn=4`, `adam_eps=1e-15`, `lr_decay=0.01` |
| `make_strategy` | `prune_opa=0.1`, `prune_scale3d=0.5`, `refine_scale2d_stop_iter=4000` |
| `CameraOpt.from_config` | `weight_decay=1e-6` |
| `AnchorStrategy.prune` | `scale_cap=0.05` |
| `select_near_views` | `theta0=5.0`, `sigma_below=1.0`, `sigma_above=10.0` |
| `plane_depth` | `min_cosine=1e-4` |
| `project` | `min_depth=1e-6` |
| `compute_scene_scale` | `margin=1.1` |
| `view_order` | `seed=42` |
| `compute_losses` | `l1_weight=0.8`, `ssim_weight=0.2` |

`SH_C0` is the **only** surviving module-level numeric constant in the package.

### 10. Docstring style

- `"""` open and close on their own lines; summary on the line **after** the opening quotes.
- Blank line between the summary and anything that follows.
- `Args:` section, one line per argument, stating shape/dtype/unit where it is a tensor.
- `Returns:` section stating shape/dtype.
- Ported code carries `Port of <repo>@<commit> <file>:<lines>` in the docstring.
- `########`-style dividers separate major sections of long files.
- Every logical block of code gets a short block comment above it, with a blank line above the comment.

### 11. Deviations from the spec that this plan makes deliberately

Nine, all forced by implementation reality. They are called out again at the task that introduces them:

1. **`AnchorStrategy.accumulate(step, info)`**, not the spec's `accumulate(info)`. The refinement-window check the spec moves into `accumulate` needs the step number. One extra positional argument.
2. **Both models keep a private `param_optimizers: dict[str, Adam]`** alongside the spec-mandated flat `optimizers: list`. gsplat's `_update_param_with_optimizer` (used by every strategy) requires the name→optimizer mapping; the flat list is built from the same objects, so there is one optimizer per tensor either way.
3. **`evals/scripts/eval_splats.py:189-211` also reads `splats.zarr`.** The spec's consumer list misses it. Task 18 fixes it.
4. **`CameraOpt` gains a `denormalize(scale)` method** beyond the spec's three. The old `denormalize_outputs` scaled model params, cameras *and* pose-translation deltas; the spec rehomes the first two and leaves the third unassigned. It goes on the module that owns those deltas.
5. **`Scaffold.__init__` takes `lr_decay` too**, and both models take the full `SplatsConfig` (not `ScaffoldConfig`). The spec exempts Scaffold from `lr_decay`, but the anchor-position lr decays today exactly as the Gaussian means lr does, and `train()` must construct either model with one call and no branch. `knn` and `adam_eps` stay Gaussians-only — `train()` never passes them.
6. **`export_gaussians(cam_to_world, intrinsics, width, height)`**, not the spec's two-argument form. `Scaffold`'s bake runs a frustum test per camera, which needs the image size. `Gaussians` ignores all four.
7. **`image_ids` stays `list[int]`** — the spec's ckpt table says `list[str]`, but `train()` is handed no names, and the retired zarr attr held `list(range(n_views))`. Task 9 writes those same frame indices; nothing reads them but provenance.
8. **Parity runs three times, not four.** The spec asks for a full parity run after phases 3, 4, 7 and 8. Phase 7 (Task 16) changes only tests, docs, `base.yaml` comments and the CHANGELOG — no line of shipped code — so a 6-minute training run there measures nothing the Task 19 run does not. Phases 5-8 are covered by the final run in Task 19, and phase 8's specific check (the mesh adapter's tuple) is Task 17 Step 5, which compares against the reading Task 2 Step 4 takes off the store before it is retired.
9. **One class `CameraOpt`, not the spec's `PoseAppearance` holder over `CameraOptModule` + `AppearanceModule`.** The three collapse into a single `torch.nn.Module` in `cameras.py` whose two halves — pose delta and color affine — are selected independently at construction from `cfg.pose_opt` and `cfg.appearance_opt`, and whose accessors pass their input through unchanged when their half is off. The two `nn.Module`s are always sized by the same `n_views`, always built together, always stepped together and always applied one after the other in the same loop body; the holder existed only to hide that they were two. `CameraOpt` names what the class is — the camera-side optimization, both aspects of it — and reads correctly whether the run optimizes pose, appearance, both or neither. Two further consequences ride along: there is no `forward` (the halves are different transforms on different tensors, so they are called by name), and `zero_init` / `random_init` are dropped because `__init__` zero-initializes every embedding — `nn.Embedding` defaults to N(0, 1), so the vendored class was only correct if the caller remembered to call `zero_init`. The gsplat citation in the module docstring names the upstream symbol, so provenance survives the merge. Task 5 does the whole merge in one commit.

---

## File Structure

### `collab_splats/splats/` after this plan

```
collab_splats/splats/
  __init__.py    GSPLAT_COMMIT; re-exports SplatsConfig, train                       unchanged
  trainer.py     SplatsConfig (+ from_dict) and train()                    843 -> ~250 lines
  gaussian.py    SH_C0, Gaussians, make_strategy                                 NEW, ~230
  scaffold.py    ScaffoldConfig, ScaffoldMLPs, Scaffold, AnchorStrategy    774 -> ~550
  losses.py      schedule, validation, compute_losses; loss fns; registry           ~350
  pgsr.py        plane / NCC helpers, select_near_views, render_neighbor   484 -> ~420
  rendering.py   render_gaussians, render_views, write_outputs, load_checkpoint     ~330
  cameras.py     rotation_6d_to_matrix, CameraOpt (pose delta + color affine)  88 -> ~200
  utils.py       scene scale/normalization, downscale, target prep, sampler   NEW, ~140
```

### Deleted

| Path | Why |
| --- | --- |
| `collab_splats/splats/outputs.py` | merged into `rendering.py` |
| `collab_splats/splats/appearance.py` | merged into `cameras.py` |
| `collab_splats/nerfstudio/` | dead — zero tracked files |
| `tests/nerfstudio_methods/` | dead — zero tracked files |

### Tests after this plan

```
tests/splats/
  synthetic.py             unchanged (make_scene)
  test_trainer.py          SplatsConfig validation + train() end-to-end
  test_utils.py            NEW — scene scale, normalization, downscale, view_order, targets
  test_gaussian.py         NEW — Gaussians init, make_strategy, denormalize, checkpoint
  test_model_interface.py  NEW — parametrised: both models expose the same 9 members
  test_scaffold.py         Scaffold + AnchorStrategy (frustum tests deleted, LambdaLR tests added)
  test_cameras.py          CameraOpt: pose delta, color affine, aspect selection
  test_losses.py           loss fns + registry + validation (moved from test_trainer.py)
  test_rendering.py        render_gaussians, render_views, write_outputs, load_checkpoint
  test_pgsr.py             plane/NCC helpers + render_neighbor
```

Deleted test files: `tests/splats/test_appearance.py` (merged into `test_cameras.py`), `tests/splats/test_outputs.py` (merged into `test_rendering.py`), `tests/nerfstudio_methods/`.

### Files outside `collab_splats/splats/` that this plan touches

| Path | Task | Change |
| --- | --- | --- |
| `collab_splats/mesh/utils.py` | 17 | `_splats_to_tsdf_inputs` reads `ckpt.pt`, renders on demand |
| `collab_splats/wrapper/reconstructor.py` | 18 | 4 sites: `splats.zarr` -> `ckpt.pt` |
| `evals/scripts/analyze_splats.py` | 18 | `analyze_normals` renders from checkpoint |
| `evals/scripts/eval_splats.py` | 18 | `depth_vs_gt` renders from checkpoint |
| `tests/mesh/test_splats_adapter.py` | 17 | fixtures write a small `ckpt.pt` |
| `tests/wrapper/test_splats_stage.py` | 18 | zarr fixture -> ckpt fixture |
| `tests/test_cu121_migration.py` | 16 | splats module list updated |
| `docs/splats.md` | 16, 18 | new file layout; zarr references removed |
| `configs/README.md` | 18 | zarr references removed |
| `docs/known-test-failures.md` | 18 | zarr test name at :9; the "does not touch `reconstructor.py`" claim at :5-6 |
| `configs/base.yaml` | 16 | comments only |
| `CLAUDE.md` | 16 | splats tree line |
| `docs/superpowers/CHANGELOG.md` | 16, 19 | entry appended |
| `docs/source/tutorials/03_splats/train_splats.ipynb` | 18 | zarr -> ckpt |
| `docs/source/tutorials/06_mesh/splats_mesh.ipynb` | 18 | zarr -> ckpt |

---

## Task 1: Delete the dead nerfstudio directories

**Files:**
- Delete: `collab_splats/nerfstudio/`
- Delete: `tests/nerfstudio_methods/`

Both directories hold **zero tracked files** — they are leftover untracked build artefacts from the nerfstudio era. Nothing in the package imports them.

- [ ] **Step 1: Confirm both directories are untracked**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  echo "--- tracked under collab_splats/nerfstudio ---" && git ls-files collab_splats/nerfstudio | wc -l && \
  echo "--- tracked under tests/nerfstudio_methods ---" && git ls-files tests/nerfstudio_methods | wc -l && \
  echo "--- on disk ---" && ls -d collab_splats/nerfstudio tests/nerfstudio_methods 2>&1
```

Expected: both counts are `0`. If either count is **not** zero, stop — the spec's premise is wrong and a human must decide what to keep.

- [ ] **Step 2: Confirm nothing imports them**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "nerfstudio" collab_splats tests evals configs --include='*.py' --include='*.yaml' | grep -v '\.worktrees' || echo "NO REFERENCES"
```

Expected: `NO REFERENCES`. (A comment mentioning nerfstudio in prose is fine; an `import collab_splats.nerfstudio` is not — if one exists, stop.)

- [ ] **Step 3: Delete**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && rm -rf collab_splats/nerfstudio tests/nerfstudio_methods && ls -d collab_splats/nerfstudio tests/nerfstudio_methods 2>&1
```

Expected: two `No such file or directory` errors.

- [ ] **Step 4: Confirm the package still imports from the worktree**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -c "import collab_splats.splats as s; print(s.__file__)"
```

Expected:

```
/workspace/collab-splats/.worktrees/clean-splats/collab_splats/splats/__init__.py
```

- [ ] **Step 5: Commit**

Nothing was tracked, so there is nothing to commit for the deletion itself. Commit the plan instead so the branch has a starting point:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  git add -f docs/superpowers/plans/2026-09-06-splats-cleanup.md && \
  git commit --only docs/superpowers/plans/2026-09-06-splats-cleanup.md \
    -m "docs(plans): splats cleanup implementation plan"
```

---

## Task 2: Parity baseline

The whole plan rests on one claim: **behavior does not change**. This task builds the instrument that proves it, and captures the "before" reading. It runs again in Tasks 11, 13 and 19.

> **DONE, then REBUILT (2026-09-06).** The harness as written below was measuring its own noise
> and would have failed on trees that are numerically identical. It has been rebuilt in place at
> `scratchpad/splats_parity.py`: `STEPS` 300 → 50, `random.seed(0)` added beside the existing
> torch/numpy seeds, three configs → seven, and `compare()` now fails on a config-set mismatch.
> **The step text below is the superseded version.** Read
> [Harness rebuild](#harness-rebuild-2026-09-06-during-task-5s-review) at the bottom of this
> plan for the measurements, and use the `before_v3` table there as the baseline — Tasks 11, 13
> and 19 gate against `before_v3`, not `before`.

**Files:**
- Create: `/tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py` (scratchpad — **never** committed)
- Create: `/tmp/claude-0/-workspace-collab-splats/scratchpad/mesh_inputs_before.json` (scratchpad, Step 4)

- [ ] **Step 1: Write the parity script**

```bash
mkdir -p /tmp/claude-0/-workspace-collab-splats/scratchpad
cat > /tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py <<'PY'
"""
Splats refactor parity harness: train three tiny configs and dump a comparable fingerprint.

Usage:
    cd <tree> && PYTHONPATH=<tree> /opt/venv/reconstruction/bin/python \
        /tmp/.../splats_parity.py <label>

Writes /tmp/.../parity_<label>.json with, per config, the quality-report PSNR/SSIM and the
mean/std/first-row of the exported ply means. Compare two labels with --compare.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

SCRATCH = Path("/tmp/claude-0/-workspace-collab-splats/scratchpad")

# Three configs cover every branch the refactor touches: both primitives and both representations
CONFIGS = {
    "3dgs_vanilla": {"primitive": "3dgs", "representation": "vanilla"},
    "2dgs_vanilla": {"primitive": "2dgs", "representation": "vanilla"},
    "3dgs_scaffold": {"primitive": "3dgs", "representation": "scaffold", "scaffold": {}},
}

STEPS = 300


def fingerprint(out_dir):
    """
    Reduce one training run's outputs to a small comparable dict.
    """
    report = json.loads((out_dir / "splats_quality_report.json").read_text())
    ply = PlyData.read(str(out_dir / "splats.ply"))["vertex"]
    means = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float64)
    return {
        "psnr": report["summary"]["psnr"],
        "ssim": report["summary"]["ssim"],
        "n_gaussians": report["summary"]["n_gaussians"],
        "means_mean": means.mean(axis=0).tolist(),
        "means_std": means.std(axis=0).tolist(),
        "means_head": means[:5].tolist(),
    }


def run(label):
    """
    Train all three configs at a fixed seed and write parity_<label>.json.
    """
    from collab_splats.splats.trainer import SplatsConfig, train
    from tests.splats.synthetic import make_scene

    images, world_to_cam, intrinsics, points, colors, depths = make_scene()
    results = {}
    for name, overrides in CONFIGS.items():
        torch.manual_seed(0)
        np.random.seed(0)
        cfg = SplatsConfig.from_dict({**overrides, "max_steps": STEPS, "log_every": STEPS})
        out_dir = SCRATCH / f"parity_{label}" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        train(cfg, images, world_to_cam, intrinsics, points, colors, out_dir, depth_targets=depths)
        results[name] = fingerprint(out_dir)
        print(f"{name}: psnr={results[name]['psnr']:.6f} n={results[name]['n_gaussians']}")

    path = SCRATCH / f"parity_{label}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


def compare(before, after):
    """
    Fail loudly if any config drifted beyond the spec's tolerances.
    """
    a = json.loads((SCRATCH / f"parity_{before}.json").read_text())
    b = json.loads((SCRATCH / f"parity_{after}.json").read_text())
    ok = True
    for name in a:
        d_psnr = abs(a[name]["psnr"] - b[name]["psnr"])
        means_ok = np.allclose(a[name]["means_head"], b[name]["means_head"], rtol=1e-4, atol=1e-6)
        n_ok = a[name]["n_gaussians"] == b[name]["n_gaussians"]
        good = d_psnr <= 1e-3 and means_ok and n_ok
        ok = ok and good
        print(f"{'PASS' if good else 'FAIL'} {name}: dPSNR={d_psnr:.2e} means_ok={means_ok} n_ok={n_ok}")
    print("PARITY OK" if ok else "PARITY BROKEN")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.label)
PY
echo written
```

- [ ] **Step 2: Capture the baseline from the worktree (pre-refactor code)**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -u /tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py before
```

Expected: three `psnr=` lines (values will differ from the numbers below — record whatever you get, they are the baseline) then `wrote /tmp/.../parity_before.json`. Takes roughly 2-4 minutes on the A40; run it in the background if it exceeds the Bash timeout.

If this fails with `ModuleNotFoundError: No module named 'gsplat.losses'`, that is the known shared-venv skew (Ground Rules §3). Stop and report — parity cannot be established without a working baseline, and the rest of the plan depends on it.

- [ ] **Step 3: Sanity-check the baseline is non-degenerate**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/python -c "
import json
d = json.load(open('/tmp/claude-0/-workspace-collab-splats/scratchpad/parity_before.json'))
for k, v in d.items():
    print(k, 'psnr', round(v['psnr'], 3), 'n', v['n_gaussians'])
    assert v['psnr'] > 5.0, f'{k} degenerate'
    assert v['n_gaussians'] > 0, f'{k} empty'
print('BASELINE OK')
"
```

Expected: three lines then `BASELINE OK`.

- [ ] **Step 4: Capture the mesh adapter's tuple from the zarr path**

Task 9 stops writing `splats.zarr`, so this is the **last moment** the old mesh path can be
measured. The spec asks Task 17 to prove that rendering from `ckpt.pt` reproduces what the
store held; that needs a reading taken now.

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -c "
import json
import numpy as np
from pathlib import Path
from collab_splats.mesh.utils import _splats_to_tsdf_inputs

scratch = Path('/tmp/claude-0/-workspace-collab-splats/scratchpad')
depths, rgbs, c2w, K = _splats_to_tsdf_inputs(scratch / 'parity_before' / '3dgs_vanilla' / 'splats.zarr')
fingerprint = {
    'shapes': [list(a.shape) for a in (depths, rgbs, c2w, K)],
    'depth_mean': float(depths.mean()),
    'depth_nonzero_frac': float((depths > 0).mean()),
    'rgb_mean': float(rgbs.mean()),
    'c2w': c2w.tolist(),
    'K': K.tolist(),
}
(scratch / 'mesh_inputs_before.json').write_text(json.dumps(fingerprint, indent=2))
print('depth_mean', fingerprint['depth_mean'], 'nonzero', fingerprint['depth_nonzero_frac'])
print('wrote mesh_inputs_before.json')
"
```

Expected: two lines, the last `wrote mesh_inputs_before.json`. If the parity run's output
directory has no `splats.zarr`, you have run Task 9 out of order — stop, `git stash`-free reset
to the pre-Task-9 commit is not needed, but you must re-run Task 2 Step 2 from that commit to
regenerate the store.

- [ ] **Step 5: Record the baseline in the plan**

Append the three PSNR values you measured to this file so later tasks (and a fresh reader) can see them without re-running:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && cat >> docs/superpowers/plans/2026-09-06-splats-cleanup.md <<'EOF'

<!-- parity baseline captured in Task 2 — fill in the measured values -->
EOF
```

Then edit that comment to hold the actual numbers, e.g. `3dgs_vanilla psnr=21.4137 n=1024`.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  git add -f docs/superpowers/plans/2026-09-06-splats-cleanup.md && \
  git commit --only docs/superpowers/plans/2026-09-06-splats-cleanup.md \
    -m "docs(plans): record splats parity baseline"
```

The parity script itself lives in the scratchpad and is **never** committed.

---

## Task 3: Create `utils.py` — scene scale, normalization, downscale, targets

Six free functions currently buried in `trainer.py` move to a new module. They are pure, they have nothing to do with the training loop, and moving them is what lets `trainer.py` drop from 843 lines to ~250.

`denormalize_outputs` and `denormalize_anchors` do **not** move here — their model half becomes `Gaussians.denormalize` / `Scaffold.denormalize` (Tasks 7, 8), their camera half becomes `denormalize_cameras` below, and their pose-delta half becomes `CameraOpt.denormalize` (Task 5).

**Files:**
- Create: `collab_splats/splats/utils.py`
- Create: `tests/splats/test_utils.py`
- Modify: `collab_splats/splats/trainer.py` (delete the moved functions, import them instead)
- Modify: `tests/splats/test_trainer.py` (it imports four of the moved helpers by name and calls
  `prepare_training_target`; the rename mechanically forces this file — it is in scope, not scope creep)

- [ ] **Step 1: Write the failing test**

```bash
cat > tests/splats/test_utils.py <<'PY'
"""
Scene-geometry and per-view preparation helpers for splat training.
"""

import numpy as np
import pytest
import torch

from collab_splats.splats.utils import (
    compute_scene_scale,
    denormalize_cameras,
    downscale_factor,
    downscale_view,
    prepare_target,
    scene_normalization,
)


def _ring_cameras(n_views=4, radius=2.0):
    """
    n_views camera-to-world poses on a ring of the given radius in the xz plane.
    """
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(n_views)])
    for view in range(n_views):
        angle = 2 * np.pi * view / n_views
        poses[view, :3, 3] = [radius * np.cos(angle), 0.0, radius * np.sin(angle)]
    return poses


def test_compute_scene_scale_is_margin_times_max_radius():
    poses = torch.from_numpy(_ring_cameras(n_views=4, radius=2.0))
    # Centroid is the origin, every camera sits at radius 2, default margin 1.1
    assert compute_scene_scale(poses) == pytest.approx(2.2, rel=1e-6)


def test_compute_scene_scale_margin_is_a_keyword_argument():
    poses = torch.from_numpy(_ring_cameras(n_views=4, radius=2.0))
    assert compute_scene_scale(poses, margin=1.0) == pytest.approx(2.0, rel=1e-6)


def test_scene_normalization_centers_and_scales_to_unit_linf():
    poses = _ring_cameras(n_views=4, radius=2.0)
    poses[:, :3, 3] += np.array([10.0, 0.0, -5.0], dtype=np.float32)
    center, scale = scene_normalization(poses)

    assert center == pytest.approx([10.0, 0.0, -5.0], abs=1e-5)
    # L-inf spread of the ring is the radius, so the scale is its reciprocal
    assert scale == pytest.approx(0.5, rel=1e-6)


def test_scene_normalization_rejects_coincident_cameras():
    poses = np.stack([np.eye(4, dtype=np.float32)] * 3)
    with pytest.raises(ValueError, match="cameras coincide"):
        scene_normalization(poses)


def test_denormalize_cameras_inverts_scene_normalization():
    poses = _ring_cameras(n_views=4, radius=2.0)
    poses[:, :3, 3] += np.array([10.0, 0.0, -5.0], dtype=np.float32)
    center, scale = scene_normalization(poses)

    normalized = torch.from_numpy(poses.copy())
    normalized[:, :3, 3] = (normalized[:, :3, 3] - torch.from_numpy(center)) * scale
    denormalize_cameras(normalized, center, scale)

    assert torch.allclose(normalized, torch.from_numpy(poses), atol=1e-5)


def test_downscale_factor_walks_the_coarse_to_fine_schedule():
    # num_downscales=2, resolution_schedule=3000: 4x until 3000, 2x until 6000, 1x after
    assert downscale_factor(0, 2, 3000) == 4
    assert downscale_factor(2999, 2, 3000) == 4
    assert downscale_factor(3000, 2, 3000) == 2
    assert downscale_factor(6000, 2, 3000) == 1
    assert downscale_factor(99999, 2, 3000) == 1


def test_downscale_factor_is_one_when_disabled():
    assert downscale_factor(0, 0, 3000) == 1


def test_downscale_view_halves_image_and_intrinsics():
    image = np.zeros((64, 32, 3), np.uint8)
    intrinsics = torch.tensor([[[10.0, 0.0, 16.0], [0.0, 10.0, 32.0], [0.0, 0.0, 1.0]]])

    small, K_small = downscale_view(image, intrinsics, 2)

    assert small.shape == (32, 16, 3)
    assert K_small[0, 0, 0] == pytest.approx(5.0)
    assert K_small[0, 1, 2] == pytest.approx(16.0)
    # Bottom row is untouched
    assert K_small[0, 2, 2] == pytest.approx(1.0)
    # The caller's intrinsics must not be mutated
    assert intrinsics[0, 0, 0] == pytest.approx(10.0)


def test_downscale_view_passes_through_at_factor_one():
    image = np.zeros((8, 8, 3), np.uint8)
    intrinsics = torch.eye(3)[None]
    small, K_small = downscale_view(image, intrinsics, 1)

    assert small is image
    assert K_small is intrinsics


def test_prepare_target_scales_rgb_to_unit_range():
    image = np.full((4, 6, 3), 255, np.uint8)
    target = prepare_target(image, None, "cpu")

    assert target["rgb"].shape == (1, 4, 6, 3)
    assert torch.allclose(target["rgb"], torch.ones(1, 4, 6, 3))
    assert target["depth"] is None


def test_prepare_target_resizes_depth_to_the_image_grid():
    image = np.zeros((8, 8, 3), np.uint8)
    depth = np.full((4, 4), 2.5, np.float32)
    target = prepare_target(image, depth, "cpu")

    assert target["depth"].shape == (1, 8, 8, 1)
    assert torch.allclose(target["depth"], torch.full((1, 8, 8, 1), 2.5))
PY
echo written
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_utils.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'collab_splats.splats.utils'`.

- [ ] **Step 3: Write `utils.py`**

```bash
cat > collab_splats/splats/utils.py <<'PY'
"""
Scene-geometry and per-view preparation helpers for splat training.

Nothing here touches a model, an optimizer or a loss: these are the pure functions the
trainer calls before and after the step loop, plus the view schedule the loop pops from.
"""

import logging
import math
import random
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


########################################
# Scene geometry
########################################


def compute_scene_scale(cam_to_world: Tensor, *, margin: float = 1.1) -> float:
    """
    gsplat's scene-extent proxy: the largest camera distance from the camera centroid, times a margin.

    Args:
        cam_to_world: (N, 4, 4) camera-to-world poses.
        margin: multiplier on the measured spread; 1.1 matches gsplat's simple_trainer.

    Returns:
        Scene extent as a python float, in the units of `cam_to_world`.
    """
    positions = cam_to_world[:, :3, 3]
    centroid = positions.mean(0)
    spread = (positions - centroid).norm(dim=-1).max()
    return float(spread) * margin


def scene_normalization(cam_to_world: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Splatfacto's Sim3: center = mean camera position, scale = 1 / max |camera coordinate - center|.

    Matches nerfstudio ``center_method="poses"`` + ``auto_scale_poses`` (L-inf, not L2); the
    "up" re-orientation is skipped because no loss or lr depends on the world's rotation.

    Args:
        cam_to_world: (N, 4, 4) float camera-to-world poses in world units.

    Returns:
        (center (3,) float32 in world units, scale as a python float in 1 / world units).
    """
    positions = cam_to_world[:, :3, 3]
    center = positions.mean(0)
    spread = float(np.abs(positions - center).max())
    if spread <= 0:
        raise ValueError("splats: cannot normalize a scene whose cameras coincide")
    return center.astype(np.float32), 1.0 / spread


def denormalize_cameras(cam_to_world: Tensor, center: np.ndarray, scale: float) -> None:
    """
    Undo ``scene_normalization`` on camera translations, in place.

    Args:
        cam_to_world: (N, 4, 4) poses in the normalized frame; mutated to world units.
        center: (3,) the center `scene_normalization` returned.
        scale: the scale `scene_normalization` returned.

    Returns:
        None — `cam_to_world` is modified in place.
    """
    center_t = torch.as_tensor(center, dtype=torch.float32, device=cam_to_world.device)
    with torch.no_grad():
        cam_to_world[:, :3, 3] = cam_to_world[:, :3, 3] / scale + center_t


########################################
# Coarse-to-fine views
########################################


def downscale_factor(step: int, num_downscales: int, resolution_schedule: int) -> int:
    """
    Coarse-to-fine divisor at a step: 2 ** max(0, num_downscales - step // resolution_schedule).

    Args:
        step: current training step.
        num_downscales: how many halvings the run starts at; 0 disables the schedule.
        resolution_schedule: steps between halvings.

    Returns:
        Integer divisor, 1 once the schedule has run out.
    """
    if num_downscales <= 0:
        return 1
    return 2 ** max(0, num_downscales - step // resolution_schedule)


def downscale_view(image: np.ndarray, intrinsics: Tensor, factor: int) -> tuple[np.ndarray, Tensor]:
    """
    Image (bilinear) and K scaled by 1 / factor; passthrough at factor 1.

    Args:
        image: (H, W, 3) uint8 image.
        intrinsics: (1, 3, 3) camera matrix in pixels.
        factor: integer divisor from `downscale_factor`.

    Returns:
        ((H // factor, W // factor, 3) image, (1, 3, 3) scaled intrinsics). Both inputs are
        returned unchanged at factor 1; neither input is ever mutated.
    """
    if factor == 1:
        return image, intrinsics

    height, width = image.shape[:2]
    small = cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR)
    K_small = intrinsics.clone()
    K_small[:, :2, :] /= factor
    return small, K_small


def prepare_target(image: np.ndarray, depth: np.ndarray | None, device: str) -> dict:
    """
    One view's supervision targets as tensors on `device`.

    Args:
        image: (H, W, 3) uint8 image.
        depth: (h, w) float depth target, possibly at a different resolution, or None.
        device: torch device string.

    Returns:
        {"rgb": (1, H, W, 3) float in [0, 1], "depth": (1, H, W, 1) float or None}. Depth is
        resized nearest so that zeros (meaning "no target") stay exactly zero.
    """
    rgb = torch.from_numpy(image).to(device).float()[None] / 255.0
    if depth is None:
        return {"rgb": rgb, "depth": None}

    height, width = image.shape[:2]
    depth_nchw = torch.from_numpy(depth).to(device)[None, None]
    depth_nchw = F.interpolate(depth_nchw, size=(height, width), mode="nearest")
    return {"rgb": rgb, "depth": depth_nchw.permute(0, 2, 3, 1)}
PY
echo written
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_utils.py -q
```

Expected: `11 passed`.

- [ ] **Step 5: Delete the moved functions from `trainer.py` and import them instead**

`trainer.py` currently defines `compute_scene_scale` (line ~244), `scene_normalization` (~254), `downscale_factor` (~470), `downscale_view` (~480), `prepare_training_target` (~494). Delete all five definitions. `denormalize_outputs` and `denormalize_anchors` stay for now — Tasks 7 and 8 replace them.

Add to the import block at the top of `trainer.py`:

```python
from collab_splats.splats.utils import (
    compute_scene_scale,
    downscale_factor,
    downscale_view,
    prepare_target,
    scene_normalization,
)
```

Then rename the two call sites of the old name inside `train()`:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  sed -i 's/prepare_training_target(/prepare_target(/g' collab_splats/splats/trainer.py && \
  grep -n "prepare_target\|prepare_training_target" collab_splats/splats/trainer.py
```

Expected: only `prepare_target` appears, in the import block and at its call site in the step loop.

- [ ] **Step 6: Verify nothing else imported the moved names**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "prepare_training_target\|from collab_splats.splats.trainer import" collab_splats tests evals docs --include='*.py' --include='*.ipynb'
```

Expected: only `SplatsConfig` / `train` imports from `trainer`. Any hit on a moved helper must be repointed at `collab_splats.splats.utils`.

- [ ] **Step 7: Run the splats suite**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats -q
```

Expected: all pass. Any failure here is a missed call site, not a real behavior change.

- [ ] **Step 8: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/utils.py collab_splats/splats/trainer.py tests/splats/test_utils.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/utils.py collab_splats/splats/trainer.py tests/splats/test_utils.py && \
  git add collab_splats/splats/utils.py tests/splats/test_utils.py && \
  git commit --only collab_splats/splats/utils.py collab_splats/splats/trainer.py tests/splats/test_utils.py \
    -m "refactor(splats): extract scene and view helpers into utils.py"
```

---

## Task 4: Replace `ViewSampler` with a `view_order` generator

The `ViewSampler` class holds three fields and exposes one method — a generator does the same job in four lines. This is the spec's "if a param is always default, ask whether it should exist" rule applied to a whole class.

**Parity matters here and it is exact.** `ViewSampler.next()` shuffles a list and `pop()`s from the *end*, so the sequence it produces is the shuffled permutation **reversed**. `yield from reversed(order)` reproduces it exactly, seed for seed. The literals in the test below were measured against the current `ViewSampler` — they are the contract.

**Files:**
- Modify: `collab_splats/splats/utils.py`
- Modify: `tests/splats/test_utils.py`
- Modify: `collab_splats/splats/trainer.py` (delete `ViewSampler`, use the generator)
- Modify: `tests/splats/test_trainer.py` (it holds the `ViewSampler` tests Step 6 orders deleted,
  and Step 8 already stages it — in scope, not scope creep)

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_utils.py`:

```python
def test_view_order_reproduces_the_measured_shuffle_and_pop_sequence():
    # Measured against the ViewSampler this replaces: random.Random(42), shuffle, pop from the end
    assert list(islice(view_order(4), 12)) == [0, 3, 1, 2, 1, 0, 2, 3, 0, 2, 3, 1]


def test_view_order_reproduces_the_sequence_for_an_odd_view_count():
    assert list(islice(view_order(3), 9)) == [2, 0, 1, 0, 1, 2, 0, 2, 1]


def test_view_order_visits_every_view_once_per_epoch():
    drawn = list(islice(view_order(5), 15))
    for start in (0, 5, 10):
        assert sorted(drawn[start : start + 5]) == [0, 1, 2, 3, 4]


def test_view_order_seed_is_a_keyword_argument():
    assert list(islice(view_order(4, seed=7), 4)) != list(islice(view_order(4, seed=42), 4))
```

and extend the imports at the top of the file:

```python
from itertools import islice
```
```python
from collab_splats.splats.utils import (
    compute_scene_scale,
    denormalize_cameras,
    downscale_factor,
    downscale_view,
    prepare_target,
    scene_normalization,
    view_order,
)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_utils.py -q
```

Expected: collection error, `ImportError: cannot import name 'view_order'`.

- [ ] **Step 3: Add `view_order` to `utils.py`**

`view_order` uses `random.Random` and annotates `-> Iterator[int]`, but `utils.py` imports
neither. **Add `import random` and `from collections.abc import Iterator`** (py3.11 — not
`typing.Iterator`); let isort place them. Then append under a new divider:

```python
########################################
# View schedule
########################################


def view_order(n_views: int, *, seed: int = 42) -> Iterator[int]:
    """
    Splatfacto's view schedule: an endless stream of seeded shuffled epochs.

    Guarantees every view trains max_steps / n_views (+-1) times, against a +-17% spread
    from sampling with replacement. Each epoch is yielded in reverse shuffle order, which is
    what the shuffle-and-pop sampler this replaces produced — the sequences are identical
    seed for seed.

    Port of nerfstudio @ 50e0e3c full_images_datamanager (random.Random shuffle + pop).

    Args:
        n_views: number of training views.
        seed: RNG seed; 42 is the value every existing run was trained at.

    Yields:
        View indices in [0, n_views), forever.
    """
    rng = random.Random(seed)
    while True:
        order = list(range(n_views))
        rng.shuffle(order)
        yield from reversed(order)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_utils.py -q
```

Expected: `15 passed`. **If the two literal-sequence tests fail, do not "fix" the literals** — they came from the code being replaced. A mismatch means `view_order` is not reproducing `ViewSampler` and the loop would train views in a different order.

- [ ] **Step 5: Switch `trainer.py` over**

Delete the whole `class ViewSampler` block (`trainer.py:354-375` after Task 3 shrank the file
from 843 to 784 lines — re-grep rather than trusting the number). Add `view_order` to the
`collab_splats.splats.utils` import.

> **KEEP `import random` in `trainer.py`.** It is load-bearing beyond `ViewSampler`:
> `trainer.py:661` (PGSR multi-view neighbor pick) calls
> `near = near_ids[view][random.randrange(len(near_ids[view]))]`. Deleting the import raises
> `NameError` on the PGSR path, which no parity config exercises.

In `train()`, replace:

```python
    view_sampler = ViewSampler(n_views)
```

with:

```python
    views = view_order(n_views)
```

and inside the step loop replace:

```python
        view = view_sampler.next()
```

with:

```python
        view = next(views)
```

- [ ] **Step 6: Verify no `ViewSampler` references survive**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "ViewSampler\|view_sampler" collab_splats tests evals docs --include='*.py' --include='*.ipynb' || echo "NONE"
```

Expected: `NONE`. Any hit in `tests/splats/test_trainer.py` must be deleted — the behavior it covered now lives in `test_utils.py`.

- [ ] **Step 7: Run the splats suite**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats -q
```

Expected: all pass.

- [ ] **Step 8: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/utils.py collab_splats/splats/trainer.py tests/splats/test_utils.py tests/splats/test_trainer.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/utils.py collab_splats/splats/trainer.py tests/splats/test_utils.py tests/splats/test_trainer.py && \
  git commit --only collab_splats/splats/utils.py collab_splats/splats/trainer.py tests/splats/test_utils.py tests/splats/test_trainer.py \
    -m "refactor(splats): replace ViewSampler with a view_order generator"
```

---

## Task 5: Collapse `cameras.py` and `appearance.py` into one `CameraOpt` module

Three classes today do one job between them: the vendored `CameraOptModule` (per-camera SE(3) delta), `AppearanceModule` in its own 28-line file (per-image color affine), and — in the spec — a `PoseAppearance` dataclass wrapping both. They are constructed, stepped and applied at exactly the same three places, and `train()` carries four parallel optionals (`pose_refiner`, `pose_optimizer`, `appearance_module`, `appearance_optimizer`) plus an `is not None` at every use.

**They become one class: `CameraOpt`, a `torch.nn.Module` in `cameras.py`.** It owns both halves, and each half is selected independently at construction:

| Constructor argument | Reads from | Owns | Applied by |
|---|---|---|---|
| `optimize_pose` | `cfg.pose_opt` (defaults true) | `translation` (n, 3) + `rotation` (n, 6) embeddings | `camera(cam_to_world, ids)` |
| `optimize_appearance` | `cfg.appearance_opt` (defaults false) | `appearance` (n, 6) embedding | `color(rgb, ids)` |

All four combinations are legal and each is tested. An unselected half is `None` and its accessor returns its input object unchanged, so the training loop calls both unconditionally and never branches. `optimizers` and `schedulers` are lists on the module holding one entry per selected half — the same shape `Gaussians` and `Scaffold` carry, so `train()` steps all of them without knowing what is on. No new config key: the selection surface is the two booleans already in `base.yaml` (Ground Rules §7 forbids adding one).

> **Deviation from the spec (1 of 3 in this task; Ground Rules §11 #9):** the spec keeps `CameraOptModule` and `AppearanceModule` as separate classes under a `PoseAppearance` dataclass. This merges all three into `CameraOpt`. Two `nn.Module`s that are always sized by the same `n_views`, always constructed together, always stepped together and always applied one after the other in the same loop body are one object; the wrapper only existed to hide the fact that they were two. The gsplat citation in the module docstring names the upstream symbol, so provenance survives the merge.

> **Deviation from the spec (2 of 3 in this task; Ground Rules §11 #9):** `zero_init()` and `random_init(std)` are dropped. `torch.nn.Embedding` initializes to N(0, 1), so the vendored class was only ever correct if the caller remembered `zero_init()` — a live footgun. `CameraOpt.__init__` zero-initializes every embedding, so a fresh module is the identity on both counts, matching what `AppearanceModule` already did. `random_init` had no production caller.

> **Deviation from the spec (3 of 3 in this task; Ground Rules §11 #4):** `CameraOpt` gains `denormalize(scale)` beyond the spec's three methods. The old `denormalize_outputs` scaled model params, cameras *and* pose-translation deltas; the spec rehomes the first two and leaves the third unassigned. It goes on the module that owns those deltas.

**Files:**
- Modify: `collab_splats/splats/cameras.py`
- Delete: `collab_splats/splats/appearance.py`
- Modify: `tests/splats/test_cameras.py`
- Delete: `tests/splats/test_appearance.py`
- Modify: `collab_splats/splats/trainer.py` (imports; `make_pose_refiner` / `make_appearance_module` deleted)
- Modify (call-site fixes only): `collab_splats/splats/outputs.py`, `tests/splats/test_trainer.py`, `tests/splats/test_outputs.py`, `configs/base.yaml:178`, `configs/README.md:374`

> **Corrections measured against the tree (2026-09-06).** The design below is right; these call
> sites are missing or wrong in the steps that follow. Line numbers are post-Task-3 — re-grep.
>
> 1. **`denormalize_outputs` / `denormalize_anchors` will crash.** Both end with
>    `if pose_refiner is not None: pose_refiner.translation.weight /= scale`. That guard is sound
>    only while `pose_refiner` is `None` when `pose_opt` is off. `refine` is *always* a module, and
>    `refine.translation` is `None` when `pose_opt: false`, so any `normalize_scene: true,
>    pose_opt: false` run raises `AttributeError`. **All three parity configs default `pose_opt`
>    true, so parity would pass and production would still break.** Fix per Deviation 3: drop the
>    `pose_refiner` parameter and those two lines from both functions, and call
>    `refine.denormalize(scale)` in `train()`'s `if cfg.normalize_scene:` block. Update both
>    docstrings and the caller in `tests/splats/test_trainer.py`.
> 2. **Step 7 rewrites one of two appearance lines.** `trainer.py:711-712` is
>    `render["rgb"] = appearance(...)` **and** `render["appearance"] = appearance.params(camera_id)`.
>    The second feeds `appearance_reg_loss` (`losses.py:151` reads `render.get("appearance")`);
>    dropping it kills `appearance_reg` silently. `CameraOpt` renames `.params` to `.appearance`:
>    ```python
>    render["rgb"] = refine.color(render["rgb"], camera_id)
>    if refine.appearance is not None:
>        render["appearance"] = refine.appearance(camera_id)
>    ```
> 3. **Do not append `refine.schedulers` to the shared `schedulers` list.** `make_pose_refiner` /
>    `make_appearance_module` append into it today (`:552`, `:559`) and it is stepped at `:744`.
>    Stepping `refine.schedulers` there *and* in its own loop decays the camera learning rates at
>    `lr_gamma**2` — a real numeric-parity break. Step them exactly once.
> 4. **`outputs.py:117` is an appearance forward call** (`rendered_rgb = appearance(rendered_rgb,
>    camera_id)` -> `refine.color(...)`), not just the pose calls Step 5 names. For the checkpoint
>    at `:269-270`, two constraints must hold until Task 9 deletes the file: `set(ckpt)` stays
>    `{"splats", "pose_adjust", "appearance", "config"}` (`test_outputs.py:53`), and
>    `ckpt["pose_adjust"]` must contain no `appearance.weight` or the strict `load_state_dict` into
>    a pose-only `CameraOpt(8)` at `test_outputs.py:73-78` fails.
> 5. **`tests/splats/test_appearance.py` holds two tests that are not about the module** —
>    `test_appearance_reg_reads_render_params_or_skips` (port to `test_losses.py`; it is the only
>    test that would catch correction 2) and `test_config_appearance_fields` (port to
>    `test_trainer.py`). Port both before deleting the file.
>
> **Found during implementation (commit `97c2fae0`); all five corrections above measured true.**
>
> 6. **Step 6 cannot pass before Step 7 — the step order as written is impossible.**
>    `collab_splats/splats/__init__.py` eagerly imports `.trainer`, which imports the camera
>    symbol, so `test_cameras.py` is a collection error after Step 3 no matter how correct
>    `cameras.py` is. Step 7 (trainer call sites) must run before Step 6 (run the tests).
> 7. **`train()` has TWO `pose_refiner(...)` call sites, not one.** Besides the main render, the
>    PGSR neighbor render refines `near_cam_to_world`. Step 7 names only the first; missing the
>    second renders the neighbor view from unrefined poses — a silent numeric break confined to
>    `pgsr_multiview` runs, which no parity config exercises.
> 8. **`tests/splats/test_scaffold.py` is an undeclared caller of `denormalize_anchors`** and
>    passes `None` positionally for the parameter correction 1 deletes. It is absent from the
>    Files list above; it belongs in the commit.
>
> **Carried into Task 9:** correction 4's minimal two-liner derives both checkpoint keys from
> the same `refine.state_dict()`, so when *both* halves are on `ckpt["pose_adjust"]` also carries
> `appearance.weight`. Both `test_outputs.py` constraints still hold (its configs are pose-only)
> and `outputs.py` dies in Task 9, so this was left minimal rather than split by key prefix —
> but whatever rehomes the checkpoint in Task 9 should split the two state dicts properly.
>
> **Also noted, out of scope:** with the camera schedulers moved off, the shared `schedulers`
> list in `train()` now holds exactly one element.

- [ ] **Step 1: Rewrite the test file**

`tests/splats/test_cameras.py` exists and holds four tests against the vendored class. Two of them (`test_zero_init_leaves_poses_unchanged`, `test_random_init_changes_poses`) test methods this task deletes, and all four call the module through `forward`, which `CameraOpt` does not have. Replace the whole file:

```python
"""
CameraOpt: pose deltas, color affine, aspect selection, and 6D rotations.
"""

import pytest
import torch

from collab_splats.splats.cameras import CameraOpt, rotation_6d_to_matrix
from collab_splats.splats.trainer import SplatsConfig


def test_rotation_6d_gives_proper_rotations():
    torch.manual_seed(0)
    rotations = rotation_6d_to_matrix(torch.randn(5, 6))
    identity = torch.eye(3).expand(5, 3, 3)
    gram = rotations @ rotations.transpose(-1, -2)
    determinants = torch.linalg.det(rotations)
    assert torch.allclose(gram, identity, atol=1e-5)
    assert torch.allclose(determinants, torch.ones(5), atol=1e-5)


def test_construction_is_the_identity_on_both_halves():
    # Embedding defaults to N(0, 1) — the constructor must zero every weight itself
    module = CameraOpt(3, optimize_pose=True, optimize_appearance=True)
    cam_to_world = torch.eye(4).expand(3, 4, 4).clone()
    cam_to_world[:, :3, 3] = torch.arange(3).float()[:, None]
    rgb = torch.rand(3, 4, 4, 3)
    ids = torch.arange(3)

    assert torch.allclose(module.camera(cam_to_world, ids), cam_to_world)
    assert torch.allclose(module.color(rgb, ids), rgb)


def test_camera_applies_the_pose_delta():
    module = CameraOpt(2)
    with torch.no_grad():
        module.translation.weight[1] = torch.tensor([0.1, -0.2, 0.3])
    cam_to_world = torch.eye(4).expand(2, 4, 4).clone()

    refined = module.camera(cam_to_world, torch.arange(2))

    # View 0 has no delta, view 1 is translated in its own camera frame
    assert torch.allclose(refined[0], cam_to_world[0])
    assert torch.allclose(refined[1, :3, 3], torch.tensor([0.1, -0.2, 0.3]))


def test_camera_matches_the_module_dtype():
    module = CameraOpt(2).double()
    with torch.no_grad():
        module.translation.weight.normal_(std=0.1)
    cam_to_world = torch.eye(4, dtype=torch.float64).expand(2, 4, 4).clone()

    assert module.camera(cam_to_world, torch.arange(2)).dtype == torch.float64


def test_color_applies_per_image_gain_and_bias():
    module = CameraOpt(2, optimize_pose=False, optimize_appearance=True)
    with torch.no_grad():
        module.appearance.weight[1] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    rgb = torch.ones(1, 1, 1, 3)

    out = module.color(rgb, torch.tensor([1]))

    # Channel 0 gain 1 + 1 = 2, channel 1 bias +0.5, channel 2 untouched
    assert out[0, 0, 0].tolist() == pytest.approx([2.0, 1.5, 1.0])


def test_both_halves_pass_through_when_neither_is_selected():
    module = CameraOpt(1, optimize_pose=False, optimize_appearance=False)
    cam_to_world = torch.eye(4)[None]
    rgb = torch.rand(1, 2, 2, 3)
    ids = torch.tensor([0])

    # Object identity, not just value equality — nothing is copied on a disabled half
    assert module.camera(cam_to_world, ids) is cam_to_world
    assert module.color(rgb, ids) is rgb


def test_denormalize_rescales_translation_deltas_only():
    module = CameraOpt(2)
    with torch.no_grad():
        module.translation.weight.fill_(2.0)
        module.rotation.weight.fill_(3.0)

    module.denormalize(scale=0.5)

    assert torch.allclose(module.translation.weight, torch.full((2, 3), 4.0))
    assert torch.allclose(module.rotation.weight, torch.full((2, 6), 3.0))


def test_denormalize_is_a_no_op_without_the_pose_half():
    module = CameraOpt(2, optimize_pose=False, optimize_appearance=True)
    module.denormalize(scale=0.5)


def test_from_config_builds_pose_only_when_appearance_is_off():
    cfg = SplatsConfig.from_dict({"pose_opt": True, "appearance_opt": False})

    module = CameraOpt.from_config(
        cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu"
    )

    assert module.translation is not None and module.rotation is not None
    assert module.appearance is None
    # One optimizer and one scheduler for the pose half, none for appearance
    assert len(module.optimizers) == 1
    assert len(module.schedulers) == 1


def test_from_config_builds_appearance_only_when_pose_is_off():
    cfg = SplatsConfig.from_dict({"pose_opt": False, "appearance_opt": True})

    module = CameraOpt.from_config(
        cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu"
    )

    # The two halves are selected independently — appearance on with pose off is a legal run
    assert module.translation is None and module.rotation is None
    assert module.appearance is not None
    assert len(module.optimizers) == 1
    assert len(module.schedulers) == 1


def test_from_config_builds_both_when_both_are_on():
    cfg = SplatsConfig.from_dict({"pose_opt": True, "appearance_opt": True})

    module = CameraOpt.from_config(
        cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu"
    )

    assert module.rotation is not None and module.appearance is not None
    assert len(module.optimizers) == 2
    assert len(module.schedulers) == 2


def test_from_config_builds_nothing_when_both_are_off():
    cfg = SplatsConfig.from_dict({"pose_opt": False, "appearance_opt": False})

    module = CameraOpt.from_config(
        cfg, n_views=4, world_extent=2.0, scene_scale=1.0, lr_gamma=0.999, device="cpu"
    )

    assert module.translation is None and module.rotation is None and module.appearance is None
    assert module.optimizers == [] and module.schedulers == []


def test_from_config_scales_the_two_pose_learning_rates_differently():
    cfg = SplatsConfig.from_dict({"pose_opt": True})

    module = CameraOpt.from_config(
        cfg, n_views=4, world_extent=8.0, scene_scale=2.0, lr_gamma=0.999, device="cpu"
    )

    rotation_group, translation_group = module.optimizers[0].param_groups
    # Rotation follows the world extent (unit-free), translation follows the training frame
    assert rotation_group["lr"] == pytest.approx(cfg.pose_lr * 8.0)
    assert translation_group["lr"] == pytest.approx(cfg.pose_lr * 2.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_cameras.py -q
```

Expected: collection error, `ImportError: cannot import name 'CameraOpt' from 'collab_splats.splats.cameras'`.

- [ ] **Step 3: Rewrite `cameras.py`**

Replace the whole file. `rotation_6d_to_matrix` is unchanged from what is there now; everything below it is new.

```python
"""
Per-camera optimization for splat training: pose deltas and per-image color correction.

``rotation_6d_to_matrix`` and the pose half of ``CameraOpt`` are vendored from
nerfstudio-project/gsplat @ d2f5c0f, examples/utils.py lines 132-153 and 27-63 (upstream name
``CameraOptModule``) — ``examples/`` is not shipped in the gsplat wheel, so the pieces we need
are copied. Local changes: the per-camera 9-vector is stored as two embeddings (translation 3,
rotation 6) so the trainer can give the two groups different learning rates — translation is in
scene units, rotation is not; every embedding is zero-initialized at construction; and the
per-image color affine (ex ``appearance.py``) lives on the same module.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """
    Gram-Schmidt 6D rotation representation (Zhou et al. 2019) -> (..., 3, 3) rotation matrices.
    """
    # First basis vector: normalized first triple
    first_triple = rotation_6d[..., :3]
    second_triple = rotation_6d[..., 3:]
    basis_x = F.normalize(first_triple, dim=-1)

    # Second basis vector: second triple with its projection onto basis_x removed
    projection = (basis_x * second_triple).sum(-1, keepdim=True) * basis_x
    basis_y = F.normalize(second_triple - projection, dim=-1)

    # Third basis vector completes the right-handed frame
    basis_z = torch.cross(basis_x, basis_y, dim=-1)
    return torch.stack((basis_x, basis_y, basis_z), dim=-2)


########################################
# Camera-side optimization
########################################


class CameraOpt(torch.nn.Module):
    """
    Learned per-camera SE(3) delta and per-image color affine, either half optional.

    - The two halves are selected independently at construction; an unselected half is `None`
      and its accessor returns its input object unchanged; `has_pose` and `has_appearance` are
      the predicates for callers that must branch, and `color_params` owns the raw per-image
      color parameters `appearance_reg` needs.
    - Every embedding is zero-initialized, so a fresh module is the identity on both counts.
    - `optimizers` and `schedulers` carry one entry per selected half, the same shape the model
      classes use, so `train()` steps them without knowing which halves are on.
    """

    def __init__(self, n_views: int, *, optimize_pose: bool = True, optimize_appearance: bool = False):
        super().__init__()

        # Pose delta per camera: translation (3, scene units) + rotation in 6D form (6, unit-free).
        # Two embeddings, not one 9-vector, so the trainer can give them different learning rates.
        self.translation = None
        self.rotation = None
        if optimize_pose:
            self.translation = torch.nn.Embedding(n_views, 3)
            self.rotation = torch.nn.Embedding(n_views, 6)
            torch.nn.init.zeros_(self.translation.weight)
            torch.nn.init.zeros_(self.rotation.weight)

            # Identity rotation in 6D form; the learned rotation delta is added to it
            self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        # Color affine per image: 3 gain deltas + 3 biases, zero = identity
        self.appearance = None
        if optimize_appearance:
            self.appearance = torch.nn.Embedding(n_views, 6)
            torch.nn.init.zeros_(self.appearance.weight)

        # Populated by from_config; empty when the module is constructed directly (tests, inference)
        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []

    @classmethod
    def from_config(
        cls,
        cfg,
        n_views: int,
        world_extent: float,
        scene_scale: float,
        lr_gamma: float,
        device: str,
        *,
        weight_decay: float = 1e-6,
    ) -> "CameraOpt":
        """
        Build the halves the config selected, with their optimizers and exponential lr decay.

        Args:
            cfg: the run's SplatsConfig. `pose_opt` and `appearance_opt` select the two halves
                independently; `pose_lr` and `appearance_lr` set their base learning rates.
            n_views: number of training views; sizes every embedding.
            world_extent: camera extent of the UN-normalized cameras, in world units. Rotation lr
                scales by this because rotation is unit-free and must not follow the training frame.
            scene_scale: camera extent of the training frame. Translation lr scales by this so
                translation steps keep their world-unit size whether or not the scene is normalized.
            lr_gamma: per-step multiplicative decay, shared with the model's schedulers.
            device: torch device string.
            weight_decay: Adam weight decay on the pose deltas; 1e-6 as in gsplat @ d2f5c0f
                examples/simple_trainer.py pose_optimizers.

        Returns:
            A CameraOpt whose `optimizers` and `schedulers` hold one entry per selected half;
            both lists are empty when neither is selected.
        """
        module = cls(
            n_views,
            optimize_pose=cfg.pose_opt,
            optimize_appearance=cfg.appearance_opt,
        ).to(device)

        # Rotation follows the world extent, translation the training frame — see the arg docs
        if module.rotation is not None:
            optimizer = torch.optim.Adam(
                [
                    {"params": module.rotation.parameters(), "lr": cfg.pose_lr * world_extent},
                    {"params": module.translation.parameters(), "lr": cfg.pose_lr * scene_scale},
                ],
                weight_decay=weight_decay,
            )
            module.optimizers.append(optimizer)
            module.schedulers.append(ExponentialLR(optimizer, gamma=lr_gamma))

        # Appearance is one flat embedding at one learning rate
        if module.appearance is not None:
            optimizer = torch.optim.Adam(module.appearance.parameters(), lr=cfg.appearance_lr)
            module.optimizers.append(optimizer)
            module.schedulers.append(ExponentialLR(optimizer, gamma=lr_gamma))

        return module

    def camera(self, cam_to_world: Tensor, camera_ids: Tensor) -> Tensor:
        """
        Apply the learned pose deltas on the right of camera-to-world.

        The delta transform is built in the embedding's dtype, so a dtype-mismatched
        `cam_to_world` raises.

        Args:
            cam_to_world: (..., 4, 4) camera-to-world poses.
            camera_ids: (...) long view indices, matching `cam_to_world`'s batch shape.

        Returns:
            (..., 4, 4) refined poses — the same tensor object when the pose half is off.
        """
        if self.rotation is None:
            return cam_to_world
        assert (
            cam_to_world.shape[:-2] == camera_ids.shape
        ), f"cam_to_world batch {cam_to_world.shape[:-2]} != camera_ids {camera_ids.shape}"
        batch_shape = cam_to_world.shape[:-2]

        # Look up each camera's translation and rotation deltas
        translation_delta = self.translation(camera_ids)
        rotation_delta = self.rotation(camera_ids)
        identity_6d = self.identity.expand(*batch_shape, -1)
        rotation = rotation_6d_to_matrix(rotation_delta + identity_6d)

        # Build the 4x4 delta transform and compose it onto the input pose
        delta_transform = torch.eye(4, device=translation_delta.device, dtype=translation_delta.dtype).repeat(
            (*batch_shape, 1, 1)
        )
        delta_transform[..., :3, :3] = rotation
        delta_transform[..., :3, 3] = translation_delta
        return torch.matmul(cam_to_world, delta_transform)

    def color(self, rgb: Tensor, camera_ids: Tensor) -> Tensor:
        """
        Apply the learned per-image affine color correction.

        Args:
            rgb: (B, H, W, 3) rendered colors.
            camera_ids: (B,) long view indices.

        Returns:
            (B, H, W, 3) rgb * (1 + gain) + bias — the same tensor object when appearance is off.
        """
        if self.appearance is None:
            return rgb
        params = self.appearance(camera_ids)
        gain = 1.0 + params[:, None, None, :3]
        bias = params[:, None, None, 3:]
        return rgb * gain + bias

    def denormalize(self, scale: float) -> None:
        """
        Undo scene normalization on the pose translation deltas, in place.

        The deltas live in the camera frame (`cam_to_world @ delta`), so only the 1 / scale
        applies; rotations are unit-free and untouched.

        Args:
            scale: the scale `utils.scene_normalization` returned.

        Returns:
            None — the module is modified in place. No-op when the pose half is off.
        """
        if self.translation is None:
            return
        with torch.no_grad():
            self.translation.weight /= scale
```

Note what is gone: `zero_init`, `random_init`, `forward`. There is no `forward` because the two halves are different transforms on different tensors — call `camera(...)` and `color(...)` by name. `AppearanceModule` and `CameraOptModule` no longer exist anywhere.

- [ ] **Step 4: Delete the old files**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  git rm collab_splats/splats/appearance.py tests/splats/test_appearance.py
```

- [ ] **Step 5: Fix every call site outside the trainer**

Four files still name the retired classes or call the retired `forward`. Find them:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "CameraOptModule\|AppearanceModule\|splats.appearance\|splats import appearance\|zero_init\|random_init" \
    collab_splats tests evals configs docs/splats.md --include='*.py' --include='*.ipynb' --include='*.yaml' --include='*.md'
```

`collab_splats/splats/outputs.py` — two `CameraOptModule | None` annotations (lines ~37 and ~226) become `CameraOpt | None`, and the import becomes `from collab_splats.splats.cameras import CameraOpt`. Any `pose_refiner(...)` call becomes `pose_refiner.camera(...)`. (`outputs.py` dies in Task 9; it has to stay green until then.)

`tests/splats/test_outputs.py:73-78` — the checkpointed-deltas replay:

```python
    refiner = CameraOpt(8)
    refiner.load_state_dict(ckpt["pose_adjust"])
    cam_to_world_in_tensor = torch.from_numpy(cam_to_world_in).float()
    camera_ids = torch.arange(8)
    with torch.no_grad():
        cam_to_world_replayed = refiner.camera(cam_to_world_in_tensor, camera_ids).numpy()
```

The state-dict keys are unchanged (`translation.weight`, `rotation.weight`, `identity`), so an existing `pose_adjust` payload still loads.

`tests/splats/test_trainer.py:316-340` — inside `test_denormalize_outputs_round_trips_gaussians_cameras_and_pose_deltas`, the import becomes `from collab_splats.splats.cameras import CameraOpt`, then:

```python
    refiner = CameraOpt(4)
    with torch.no_grad():
        refiner.translation.weight[1] = torch.tensor([0.1, -0.2, 0.3])
    refined_normalized = refiner.camera(normalized_cams[1:2], torch.tensor([1]))[0]
```

The `refiner.zero_init()` line is deleted — the constructor does it.

`configs/base.yaml:178` and `configs/README.md:374` name `CameraOptModule` in a comment and a table cell. Both become `CameraOpt`. Neither is a config key, so Ground Rules §7 is untouched (Task 16 Step 7's `IDENTICAL` proof compares parsed values, not comments).

Verify nothing is left:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "CameraOptModule\|AppearanceModule\|zero_init\|random_init" \
    collab_splats tests evals configs docs/splats.md \
    | grep -v "upstream name" \
  || echo "MERGE COMPLETE"
```

Expected: `MERGE COMPLETE` — except for `collab_splats/splats/trainer.py`, whose `make_pose_refiner` still calls `zero_init`. Step 7 deletes that function; the tree is knowingly inconsistent between here and there, and Step 8 is the gate. (`docs/superpowers/` is excluded throughout: older specs and plans record what was true when they were written, and the `grep -v` spares the gsplat citation in `cameras.py`'s docstring, the one place the upstream name must survive.)

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_cameras.py -q
```

Expected: 13 passed.

- [ ] **Step 7: Switch `train()` onto `CameraOpt`**

Delete `make_pose_refiner` (`trainer.py:~435`) and `make_appearance_module` (`trainer.py:~459`), and change the import to `from collab_splats.splats.cameras import CameraOpt`. In `train()`, replace the pose/appearance setup block with:

```python
    # Camera-side optimization: pose deltas, color affine, both or neither
    refine = CameraOpt.from_config(cfg, n_views, world_extent, scene_scale, lr_gamma, device)
```

In the step loop, replace the guarded pose refine with an unconditional call:

```python
        cam_to_world_view = refine.camera(cam_to_world[view : view + 1], camera_ids)
```

and the guarded appearance application with:

```python
        rgb = refine.color(rgb, camera_ids)
```

The optimizer step block becomes:

```python
        for optimizer in refine.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in refine.schedulers:
            scheduler.step()
```

In the denormalize block at the end of `train()`, `pose_refiner` becomes `refine` and the translation rescale becomes `refine.denormalize(scale)`.

- [ ] **Step 8: Run the splats suite**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats -q
```

Expected: all pass.

- [ ] **Step 9: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/cameras.py collab_splats/splats/trainer.py collab_splats/splats/outputs.py tests/splats/test_cameras.py tests/splats/test_trainer.py tests/splats/test_outputs.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/cameras.py collab_splats/splats/trainer.py collab_splats/splats/outputs.py tests/splats/test_cameras.py tests/splats/test_trainer.py tests/splats/test_outputs.py && \
  git commit --only collab_splats/splats/cameras.py collab_splats/splats/appearance.py collab_splats/splats/trainer.py collab_splats/splats/outputs.py \
    tests/splats/test_cameras.py tests/splats/test_appearance.py tests/splats/test_trainer.py tests/splats/test_outputs.py \
    configs/base.yaml configs/README.md \
    -m "refactor(splats): one CameraOpt module for pose and appearance"
```

`--only` is deliberate — the index is shared with the other worktrees (see Ground Rules §2), so never `git add -A` here.

---

## Task 6: Reorder `losses.py` and move loss validation out of `trainer.py`

Two problems. First, `losses.py` reads bottom-up: eight loss implementations, then the registry, then the scheduling logic that drives them. Second, `SplatsConfig.from_dict` in `trainer.py` validates the loss schedule — 35 lines of loss knowledge in the trainer, importing `OPTIONAL_LOSSES` and `LOSS_SPEC_KEYS` from `losses.py` to do it.

New order in `losses.py`, top to bottom: the schedule API a reader wants first (`default_losses`, `validate_schedule`, `loss_weight`, `loss_active`, `compute_losses`), then a divider, then the eight loss functions, then a divider and the two registries at the bottom. Forward references are fine — every use of `OPTIONAL_LOSSES` is at call time, not import time.

**Files:**
- Modify: `collab_splats/splats/losses.py`
- Modify: `collab_splats/splats/trainer.py`
- Modify: `tests/splats/test_losses.py`
- Modify: `tests/splats/test_trainer.py` (loss-validation tests move out)

> **Corrections measured against the tree (2026-09-06).** The code in this task's steps is
> byte-accurate — `_default_losses`, `LOSS_SPEC_KEYS`, the `validate_schedule` body and every
> error string match what is in `trainer.py` today, so Step 9's "the error strings are
> byte-identical" claim holds. Three things the steps get wrong:
>
> 1. **Step 1's import block is an ADDITION, not a replacement.** As of Task 5,
>    `tests/splats/test_losses.py` imports seven names — `OPTIONAL_LOSSES`,
>    `appearance_reg_loss`, `compute_losses`, `loss_active`, `loss_weight`, `opacity_reg_loss`,
>    `scale_reg_loss` — and live tests use every one. Step 1's block lists only six and omits
>    all three of the `*_loss` functions, so pasting it as given fails collection on
>    `NameError`. Add only `default_losses` and `validate_schedule` to what is already there.
>    Task 5 also landed `test_appearance_reg_reads_render_params_or_skips` in this file (it is
>    the only test guarding a silently-dropped `render["appearance"]`) — leave it alone.
> 2. **Step 2's command was missing its `cd`** (now fixed in the block below). Without it
>    `sys.path[0]` is wrong and the run tests the main checkout, not this worktree.
> 3. **Line numbers drifted** (Tasks 3-4 shrank `trainer.py`). Measured now: `_default_losses`
>    `66-80`, the `LOSS_SPEC_KEYS` comment + dict `83-99`, the losses import `:37`,
>    `__post_init__`'s default `:152`, the `for name, spec in cfg.losses.items():` loop `:199`.
>    `losses.py`'s numbers are still exact: stale citation `:65`, `OPTIONAL_LOSSES` `:258`,
>    `loss_weight` `:274`, `loss_active` `:301`, `compute_losses` `:308`, `total = 0.8 * l1 +
>    0.2 * ssim` `:328`.
>
> Step 7's "everything from the loop down to just before `return cfg`" was checked against the
> tree and is exact — the loss loop, the distortion guard and the three `depth_ratio` guards run
> contiguously to `return cfg` with no unrelated validation interleaved. Delete the whole span.
>
> **Step 8 must delete NOTHING — following it literally loses coverage.** Step 8 asserts the
> loss tests in `test_trainer.py` are "now covered by `test_losses.py`". Checked test by test,
> that is false in four cases, and Step 8's own escape clause ("round-trip tests that go
> through `SplatsConfig.from_dict` ... may stay if they exercise the wiring") covers all of
> them, since every one goes through `from_dict`:
>
> - `test_config_rejects_invalid` is parametrized with six cases, only four of which are loss
>   rules; `{"primitive": "4dgs"}` and `{"unknown_key": 1}` are asserted nowhere else in the
>   suite. Deleting the test to drop its loss params silently drops those two rules.
> - `test_depth_ratio_rejected_on_another_loss` asserts `depth_ratio` is refused on the `depth`
>   loss (`LOSS_SPEC_KEYS` grants it to `normal_consistency` only). The new suite has **no**
>   equivalent — its nearest test passes `normal_consistency` with a `typo` key instead.
> - `test_bad_spec_key_message_names_the_keys_legal_for_that_loss` pins both message variants
>   byte-exactly; its new counterpart only matches the substring `depth_ratio`.
> - The three depth_ratio parametrized tests cover `[1.5, -0.5]`, `[True, False, None, "0.6",
>   [0.6]]` and zero-on-3dgs; the new suite covers `1.5` only, `True` only, and has no
>   zero-on-3dgs case. All three existing tests are strictly broader.
>
> `test_default_losses_match_primitive` likewise stays: it runs through `__post_init__`, so it
> is the wiring test for `self.losses = default_losses(self.primitive)`. Expected delta is
> **+15 collected tests in `test_losses.py`, `test_trainer.py` unchanged.**
>
> **This is also what keeps the delegation observable.** After Step 7, `from_dict`'s validation
> is one call to `validate_schedule`. The new tests call `validate_schedule` directly, so if the
> `from_dict` round-trip tests were deleted too, removing that call would leave validation dead
> in production with a green suite.
>
> **Step 4's `LOSS_SPEC_KEYS = { ... }` is a placeholder, not code** — copy the real dict and
> its comment out of `trainer.py` byte for byte, and prove it with a `diff` of the
> `sed -n '/^LOSS_SPEC_KEYS = {/,/^}/p'` span before and after.
>
> **Nothing outside `trainer.py` references `LOSS_SPEC_KEYS` or `_default_losses`** (grepped
> repo-wide), so the move has no external consumers beyond what Step 7 lists.

- [ ] **Step 1: Write the failing tests**

Append to `tests/splats/test_losses.py`:

```python
def test_default_losses_gives_3dgs_the_mcmc_regularizers():
    losses = default_losses("3dgs")

    assert losses["opacity_reg"] == {"weight": 0.01}
    assert losses["scale_reg"] == {"weight": 0.01}
    assert "distortion" not in losses


def test_default_losses_gives_2dgs_distortion():
    losses = default_losses("2dgs")

    assert losses["distortion"] == {"weight": 0.01, "start": 3000}
    assert "opacity_reg" not in losses


def test_default_losses_shares_depth_normal_and_appearance_across_primitives():
    for primitive in ("3dgs", "2dgs"):
        losses = default_losses(primitive)
        assert losses["depth"] == {"weight": 0.01}
        assert losses["normal_consistency"] == {"weight": 0.05, "start": 7000}
        assert losses["appearance_reg"] == {"weight": 1e-3}


def test_validate_schedule_accepts_the_defaults():
    for primitive in ("3dgs", "2dgs"):
        validate_schedule(default_losses(primitive), primitive)


def test_validate_schedule_rejects_an_unknown_loss():
    with pytest.raises(ValueError, match="unknown loss 'wobble'"):
        validate_schedule({"wobble": {"weight": 1.0}}, "3dgs")


def test_validate_schedule_requires_a_weight():
    with pytest.raises(ValueError, match=r"splats.losses.depth: expected"):
        validate_schedule({"depth": {"start": 100}}, "3dgs")


def test_validate_schedule_names_the_keys_legal_for_that_loss():
    with pytest.raises(ValueError, match="depth_ratio"):
        validate_schedule({"normal_consistency": {"weight": 1.0, "typo": 1}}, "2dgs")


def test_validate_schedule_rejects_a_decay_without_both_endpoints():
    with pytest.raises(ValueError, match="decay needs weight > 0"):
        validate_schedule({"depth": {"weight": 1.0, "end": 100}}, "3dgs")


def test_validate_schedule_rejects_a_decay_that_ends_before_it_starts():
    with pytest.raises(ValueError, match="end > start"):
        validate_schedule({"depth": {"weight": 1.0, "end_weight": 0.1, "start": 500, "end": 100}}, "3dgs")


def test_validate_schedule_rejects_distortion_on_3dgs():
    with pytest.raises(ValueError, match="distortion is 2dgs-only"):
        validate_schedule({"distortion": {"weight": 0.01}}, "3dgs")


def test_validate_schedule_allows_zero_weight_distortion_on_3dgs():
    validate_schedule({"distortion": {"weight": 0.0}}, "3dgs")


def test_validate_schedule_rejects_a_boolean_depth_ratio():
    with pytest.raises(ValueError, match="depth_ratio must be a number"):
        validate_schedule({"normal_consistency": {"weight": 1.0, "depth_ratio": True}}, "2dgs")


def test_validate_schedule_rejects_an_out_of_range_depth_ratio():
    with pytest.raises(ValueError, match=r"depth_ratio must be in \[0, 1\]"):
        validate_schedule({"normal_consistency": {"weight": 1.0, "depth_ratio": 1.5}}, "2dgs")


def test_validate_schedule_rejects_depth_ratio_on_3dgs():
    with pytest.raises(ValueError, match="depth_ratio > 0 is 2dgs-only"):
        validate_schedule({"normal_consistency": {"weight": 1.0, "depth_ratio": 0.5}}, "3dgs")


def test_compute_losses_photometric_weights_are_keyword_arguments():
    render = {"rgb": torch.zeros(1, 8, 8, 3)}
    target = {"rgb": torch.ones(1, 8, 8, 3), "depth": None}
    params = torch.nn.ParameterDict({"opacities": torch.nn.Parameter(torch.zeros(4))})

    total, values = compute_losses(0, render, target, params, {}, 1.0, l1_weight=1.0, ssim_weight=0.0)

    # Pure L1 between all-zeros and all-ones is 1.0
    assert float(total) == pytest.approx(values["l1"])
    assert values["l1"] == pytest.approx(1.0)
```

Extend the file's imports:

```python
from collab_splats.splats.losses import (
    OPTIONAL_LOSSES,
    compute_losses,
    default_losses,
    loss_active,
    loss_weight,
    validate_schedule,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_losses.py -q
```

Expected: collection error, `ImportError: cannot import name 'default_losses'`.

- [ ] **Step 3: Add the schedule API to the top of `losses.py`**

Immediately after the import block in `collab_splats/splats/losses.py`, before the existing `# Optional losses` divider, insert:

```python
########################################
# Schedule
########################################


def default_losses(primitive: str) -> dict[str, dict]:
    """
    Default loss schedule per primitive: MCMC regularizers for 3dgs, distortion for 2dgs.

    Args:
        primitive: "3dgs" or "2dgs".

    Returns:
        A fresh `{name: spec}` mapping the caller owns and may mutate.
    """
    losses = {
        "depth": {"weight": 0.01},
        "normal_consistency": {"weight": 0.05, "start": 7000},
        "appearance_reg": {"weight": 1e-3},
    }
    if primitive == "3dgs":
        losses["opacity_reg"] = {"weight": 0.01}
        losses["scale_reg"] = {"weight": 0.01}
    else:
        losses["distortion"] = {"weight": 0.01, "start": 3000}
    return losses


def validate_schedule(losses: dict[str, dict], primitive: str) -> None:
    """
    Reject unknown losses, ill-formed specs, and primitive/loss mismatches.

    Args:
        losses: the yaml `splats.losses` mapping, `{name: {weight[, start, end, end_weight, ...]}}`.
        primitive: "3dgs" or "2dgs"; decides the distortion and depth_ratio guards.

    Returns:
        None. Raises ValueError naming the offending key on any problem.
    """
    for name, spec in losses.items():
        if name not in OPTIONAL_LOSSES:
            raise ValueError(f"splats.losses: unknown loss '{name}'; allowed {sorted(OPTIONAL_LOSSES)}")

        # Some losses carry their own tuning keys; each belongs to exactly one loss
        allowed_spec_keys = {"weight", "start", "end", "end_weight"} | LOSS_SPEC_KEYS.get(name, set())
        unknown_spec_keys = set(spec) - allowed_spec_keys
        if unknown_spec_keys or "weight" not in spec:
            # Report the keys legal for THIS loss, so a depth_ratio typo isn't told the key doesn't exist
            optional_keys = ", ".join(sorted(allowed_spec_keys - {"weight"}))
            raise ValueError(f"splats.losses.{name}: expected {{weight[, {optional_keys}]}}, got {sorted(spec)}")

        # Decay entries need both endpoints positive (log-linear) and a non-empty interval
        if "end" in spec:
            end_weight = spec.get("end_weight")
            if end_weight is None or spec["weight"] <= 0 or end_weight <= 0 or spec["end"] <= spec.get("start", 0):
                raise ValueError(
                    f"splats.losses.{name}: decay needs weight > 0, end_weight > 0 and end > start, got {spec}"
                )

    # The distortion map only exists for 2DGS
    distortion_weight = losses.get("distortion", {}).get("weight", 0.0)
    if primitive == "3dgs" and distortion_weight > 0:
        raise ValueError("splats.losses.distortion is 2dgs-only; set its weight to 0 or use primitive: 2dgs")

    # bool is an int subclass, so `depth_ratio: yes` would coerce to a silent full median blend;
    # reject bools and non-numbers (a quoted '0.6' too) before any coercion, naming the key
    raw_depth_ratio = losses.get("normal_consistency", {}).get("depth_ratio", 0.0)
    if isinstance(raw_depth_ratio, bool) or not isinstance(raw_depth_ratio, (int, float)):
        raise ValueError(
            f"splats.losses.normal_consistency.depth_ratio must be a number in [0, 1], got {raw_depth_ratio!r}"
        )

    # Median depth only exists for 2DGS, so a non-zero blend on 3dgs is a config error
    depth_ratio = float(raw_depth_ratio)
    if not 0.0 <= depth_ratio <= 1.0:
        raise ValueError(f"splats.losses.normal_consistency.depth_ratio must be in [0, 1], got {depth_ratio}")
    if depth_ratio > 0 and primitive != "2dgs":
        raise ValueError(
            "splats.losses.normal_consistency.depth_ratio > 0 is 2dgs-only "
            "(median depth is a rasterization_2dgs output); set it to 0 or use primitive: 2dgs"
        )
```

- [ ] **Step 4: Move `loss_weight`, `loss_active` and `compute_losses` up, and the registries down**

Move the existing `# Weighted sum` section (`loss_weight`, `loss_active`, `compute_losses` — currently the tail of the file) so it sits **directly under** the new `validate_schedule`. Move `OPTIONAL_LOSSES` (currently line ~258) to the **bottom** of the file under a new divider, and move `LOSS_SPEC_KEYS` from `trainer.py:88-100` next to it, keeping its explanatory comment verbatim:

```python
########################################
# Registries
########################################

# Name in the yaml `losses:` block -> function. Also the allow-list for config validation.
OPTIONAL_LOSSES = {
    "depth": depth_loss,
    "normal_consistency": normal_consistency_loss,
    "distortion": distortion_loss,
    "opacity_reg": opacity_reg_loss,
    "scale_reg": scale_reg_loss,
    "appearance_reg": appearance_reg_loss,
    "pgsr_normal": pgsr_normal_loss,
    "pgsr_multiview": pgsr_multiview_loss,
}

# Per-loss tuning keys beyond {weight, start, end, end_weight}. depth_ratio is the RaDe-GS
# median-normal blend; the pgsr_* keys are PGSR's own hyperparameters (GS-SR
# gssr/scene/pgsr_scene.py:52-70), kept on the loss spec rather than promoted to SplatsConfig
# because they mean nothing when the loss is off.
LOSS_SPEC_KEYS = { ... }   # copied verbatim from trainer.py:83-99, contents unchanged
```

The resulting file order is: docstring, imports, `# Schedule` (default_losses, validate_schedule, loss_weight, loss_active, compute_losses), `# Optional losses` (the eight functions, unchanged), `# Registries`.

- [ ] **Step 5: Make the photometric weights keyword arguments**

In `compute_losses`, change the signature and the one line that uses the literals:

```python
def compute_losses(
    step: int,
    render: dict,
    target: dict,
    gaussians: torch.nn.ParameterDict,
    loss_schedule: dict[str, dict],
    scene_scale: float,
    *,
    l1_weight: float = 0.8,
    ssim_weight: float = 0.2,
) -> tuple[Tensor, dict[str, float]]:
```

```python
    total = l1_weight * l1 + ssim_weight * ssim
```

Extend its docstring with the two new arguments:

```python
        l1_weight: photometric L1 weight; 0.8 is the value every existing run trained at.
        ssim_weight: photometric (1 - SSIM) weight; 0.2 likewise.
```

- [ ] **Step 6: Fix the stale worktree citation**

`losses.py:65` cites a path inside another session's worktree, which no reader can open:

```
      (`.worktrees/streaming/collab_splats/nerfstudio/models/rade_gs.py:254`); omitted here on
```

Replace with the upstream citation:

```
      (BaowenZ/RaDe-GS @ main, scene/gaussian_model.py — background median depth is refilled with
      its max before differencing); omitted here on
```

- [ ] **Step 7: Strip the loss knowledge out of `trainer.py`**

- Delete `_default_losses` (`trainer.py:66-80`) and `LOSS_SPEC_KEYS` with its comment (`trainer.py:83-99`).
- `__post_init__`'s default becomes `self.losses = default_losses(self.primitive)`.
- In `from_dict`, delete the whole `for name, spec in cfg.losses.items():` block, the distortion guard, and the three `depth_ratio` guards — everything from the loop down to just before `return cfg`. Replace with one call, placed where the loop was:

```python
        # Loss schedule shape and primitive compatibility
        validate_schedule(cfg.losses, cfg.primitive)
```

- The `losses` import becomes:

```python
from collab_splats.splats.losses import compute_losses, default_losses, loss_active, validate_schedule
```

`OPTIONAL_LOSSES` and `LOSS_SPEC_KEYS` are no longer needed in `trainer.py`.

- [ ] **Step 8: Move the loss-validation tests out of `test_trainer.py`**

Any test in `tests/splats/test_trainer.py` that asserts on a `splats.losses...` error message is now covered by `test_losses.py`. Delete those tests from `test_trainer.py`; the round-trip tests that go through `SplatsConfig.from_dict` and check the same message may stay if they exercise the wiring rather than the rule. Find them with:

```bash
grep -n "splats.losses" tests/splats/test_trainer.py
```

- [ ] **Step 9: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_losses.py tests/splats/test_trainer.py -q
```

Expected: all pass. The error strings are byte-identical to before, so any test that matched them still matches.

- [ ] **Step 10: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/losses.py collab_splats/splats/trainer.py tests/splats/test_losses.py tests/splats/test_trainer.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/losses.py collab_splats/splats/trainer.py tests/splats/test_losses.py tests/splats/test_trainer.py && \
  git commit --only collab_splats/splats/losses.py collab_splats/splats/trainer.py tests/splats/test_losses.py tests/splats/test_trainer.py \
    -m "refactor(splats): move loss schedule validation into losses.py"
```

---

## Task 7: Create `gaussian.py` — the `Gaussians` model

This is the first half of the plan's centerpiece. Everything the trainer currently does to vanilla Gaussians — initialize them, build their optimizers, pick their densification strategy, render them, hook the strategy before and after backward, undo scene normalization, write them to a checkpoint — becomes methods on one class.

The class deliberately **does not import `SplatsConfig`**: `trainer.py` imports `gaussian.py`, so the reverse would be circular. It reads what it needs off `cfg` in `__init__` and stores three plain attributes; `from_checkpoint` reads the same three out of the checkpoint's plain-dict `config`.

> **Deviation from the spec:** the class keeps a private `param_optimizers: dict[str, Adam]` alongside the spec-mandated flat `optimizers: list`. gsplat's strategies call `_update_param_with_optimizer`, which needs the name→optimizer mapping. The flat list is built from the same objects — there is still exactly one Adam per tensor.

**Files:**
- Create: `collab_splats/splats/gaussian.py`
- Create: `tests/splats/test_gaussian.py`

> **Re-checked against the tree (2026-09-06, after Task 5 landed). Step 5 has been REMOVED —
> read the correction block below before starting.** As originally written this task was *not*
> purely additive: Step 5 deleted `SH_DC_NORMALIZER` from `rendering.py` while every one of its
> readers lives in **other** files. With Step 5 dropped the task is genuinely additive — it
> creates two new files, modifies nothing, carries no stale line numbers, and cannot collide
> with any sibling task.
>
> It still gates Task 8, which imports `SH_C0` from the module created here (`SH_C0` occurs zero
> times in the tree today), so **Task 8 must not start until this one is committed.**

> **STATUS: COMPLETE (2026-09-06).** Implemented at `235c954e`, then three code-quality fix rounds:
> `c414eff7`, `31a2495b`, `8a4fc746`. Spec review **SPEC COMPLIANT** (zero blockers; its Notes 1-6
> are triaged into Tasks 8/9/10 — see `C10-15`, `N2`, `N3`). Code-quality review **APPROVED** at
> round 4, measured on a pinned tree at `8a4fc746`: `tests/splats` **247 passed, `PYTEST_RC=0`,
> 0 skipped**; `test_gaussian.py` 27 passed. Ten mutants killed, each by exactly one test with no
> collateral; `N1` (dropping `pre_backward`'s `isinstance` guard) correctly **SURVIVES** as an
> equivalent mutant — upstream MCMC's pre-backward hook is a no-op, so the guard is documentation.
> `gaussian.py` is deliberately **no longer** byte-identical to `c414eff7`: round 3 added exactly one
> comment line (`git diff --numstat` = `1 0`). Four remaining `render_gaussians` argument gaps were
> measured and handed to Task 10 as **C10-15**; they are follow-ups, not blockers.

> **CORRECTIONS — measured against the tree, these override the step text below**
>
> **C7-1 — Step 5 is deleted, not adjusted.** It instructed you to delete the
> `SH_DC_NORMALIZER` definition at `rendering.py:25`. Its readers are `outputs.py:24` and
> `:213`, `trainer.py:41` and `:315`, and `tests/splats/test_trainer.py:13` and `:221` —
> **none of them are in `rendering.py`**. Deleting the definition without touching those six
> sites makes `import collab_splats.splats` raise `ImportError` and the entire `tests/splats`
> suite fail to *collect* (zero tests collected, not 185 failures). **Task 9 Step 4 already
> removes `SH_DC_NORMALIZER` as part of folding `outputs.py` into `rendering.py`, with all its
> readers in scope.** Leave the constant exactly where it is. `gaussian.py` defines its own
> `SH_C0`; the two coexist until Task 9.
>
> **C7-2 — Expected test count is `16 passed`, not `17`.** The Step 1 heredoc defines sixteen
> `def test_` functions. Counted, not estimated.
>
> **C7-3 — `rendering.py` is not in the Files list and must not be.** Step 7's `black`,
> `isort` and `git commit --only` lines have had `collab_splats/splats/rendering.py` removed
> for the same reason.
>
> **C7-4 — Two knowing duplications, both temporary; do not try to resolve them here.**
> `make_strategy` already exists at `trainer.py:280-315` (measured at `235c954e`; `349-384` was stale) and this task creates a second one in
> `gaussian.py`; `Gaussians.activate()` and `Gaussians.render()` duplicate
> `rendering.py:57-70 activate_vanilla` and `:238-264 render_view`. Task 10 deletes the
> trainer-side copies when `train()` is rewritten onto the model interface. Step 6's note
> already says this for `init_gaussians_from_points`; it applies to all three.

- [ ] **Step 1: Write the failing test**

```bash
cat > tests/splats/test_gaussian.py <<'PY'
"""
Vanilla Gaussian primitives: initialization, strategy selection, denormalization, checkpointing.
"""

import math

import numpy as np
import pytest
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from collab_splats.splats.gaussian import SH_C0, Gaussians, make_strategy
from collab_splats.splats.trainer import SplatsConfig


def _seed_cloud(n_points=32):
    """
    A small deterministic seed cloud and its colors.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, (n_points, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (n_points, 3)).astype(np.uint8)
    return points, colors


def _model(**overrides):
    """
    A CPU Gaussians model over the seed cloud.
    """
    points, colors = _seed_cloud()
    cfg = SplatsConfig.from_dict({"max_steps": 100, **overrides})
    return Gaussians(cfg, points, colors, scene_scale=2.0, n_views=4, device="cpu")


def test_sh_c0_is_the_degree_zero_spherical_harmonic():
    assert SH_C0 == pytest.approx(0.5 / math.sqrt(math.pi))
    # The value every existing checkpoint's sh0 was written against
    assert SH_C0 == pytest.approx(0.28209479177387814)


def test_gaussians_start_one_per_seed_point():
    model = _model()

    assert model.n_primitives == 32
    assert model.params["means"].shape == (32, 3)
    assert model.params["quats"].shape == (32, 4)
    assert model.params["opacities"].shape == (32,)


def test_gaussians_store_color_in_the_degree_zero_sh_band():
    points, colors = _seed_cloud()
    cfg = SplatsConfig.from_dict({"max_steps": 100})
    model = Gaussians(cfg, points, colors, scene_scale=2.0, n_views=4, device="cpu")

    expected = (torch.from_numpy(colors).float() / 255.0 - 0.5) / SH_C0
    assert torch.allclose(model.params["sh0"][:, 0, :], expected, atol=1e-5)
    # Higher bands start at zero
    assert torch.all(model.params["shN"] == 0)


def test_gaussians_sh_band_count_follows_sh_degree():
    model = _model(sh_degree=2)

    # (degree + 1)^2 coefficients, one held out as the DC band
    assert model.params["sh0"].shape[1] == 1
    assert model.params["shN"].shape[1] == (2 + 1) ** 2 - 1


def test_gaussians_opacities_start_at_the_configured_logit():
    model = _model(init_opacity=0.1)

    assert torch.allclose(torch.sigmoid(model.params["opacities"]), torch.full((32,), 0.1), atol=1e-6)


def test_gaussians_build_one_optimizer_per_tensor():
    model = _model()

    # Six parameters, six Adams, so the strategy can grow and prune optimizer state per tensor
    assert len(model.optimizers) == 6
    assert {id(o) for o in model.optimizers} == {id(o) for o in model.param_optimizers.values()}


def test_gaussians_scale_the_means_learning_rate_by_the_scene():
    cfg = SplatsConfig.from_dict({"max_steps": 100})
    points, colors = _seed_cloud()
    model = Gaussians(cfg, points, colors, scene_scale=3.0, n_views=4, device="cpu")

    assert model.param_optimizers["means"].param_groups[0]["lr"] == pytest.approx(cfg.means_lr * 3.0)


def test_gaussians_expose_one_scheduler_on_the_means_optimizer():
    model = _model()

    assert len(model.schedulers) == 1
    assert model.schedulers[0].optimizer is model.param_optimizers["means"]


def test_gaussians_activate_raw_parameters_for_the_rasterizer():
    model = _model()
    activated = model.activate()

    assert torch.allclose(activated["scales"], torch.exp(model.params["scales"]))
    assert torch.allclose(activated["opacities"], torch.sigmoid(model.params["opacities"]))
    # SH bands are concatenated back into one (N, K, 3) tensor
    assert activated["colors"].shape == (32, (model.sh_degree + 1) ** 2, 3)


def test_gaussians_denormalize_inverts_the_sim3():
    model = _model()
    center = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    scale = 0.25
    means_before = model.params["means"].detach().clone()
    scales_before = model.params["scales"].detach().clone()

    model.denormalize(center, scale)

    assert torch.allclose(model.params["means"], means_before / scale + torch.from_numpy(center), atol=1e-5)
    assert torch.allclose(model.params["scales"], scales_before - math.log(scale), atol=1e-5)


def test_gaussians_export_returns_the_raw_parameters():
    model = _model()
    exported = model.export_gaussians(torch.eye(4)[None], torch.eye(3)[None], 64, 64)

    assert set(exported) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    assert exported["means"] is model.params["means"]


def test_gaussians_checkpoint_round_trips():
    model = _model()
    ckpt = model.checkpoint()
    ckpt["config"] = {"primitive": "3dgs", "sh_degree": 3, "sh_degree_interval": 1000}

    restored = Gaussians.from_checkpoint(ckpt, "cpu")

    assert restored.n_primitives == model.n_primitives
    assert torch.allclose(restored.params["means"], model.params["means"])
    # A checkpoint-restored model renders; it does not train
    assert restored.optimizers == []
    assert restored.schedulers == []
    assert restored.strategy is None


def test_make_strategy_picks_mcmc_for_3dgs():
    cfg = SplatsConfig.from_dict({"primitive": "3dgs", "cap_max": 500})
    strategy = make_strategy(cfg, n_views=4)

    assert isinstance(strategy, MCMCStrategy)
    assert strategy.cap_max == 500


def test_make_strategy_picks_default_with_splatfacto_arguments_for_2dgs():
    cfg = SplatsConfig.from_dict({"primitive": "2dgs", "grow_grad2d": 2e-4})
    strategy = make_strategy(cfg, n_views=4)

    assert isinstance(strategy, DefaultStrategy)
    assert strategy.absgrad is False
    assert strategy.key_for_gradient == "gradient_2dgs"
    assert strategy.grow_grad2d == pytest.approx(2e-4)
    assert strategy.pause_refine_after_reset == 4 + 100


def test_make_strategy_caps_the_refine_pause_so_densification_still_runs(caplog):
    cfg = SplatsConfig.from_dict({"primitive": "2dgs"})
    # n_views + 100 would exceed reset_every, which silently disables refinement forever
    strategy = make_strategy(cfg, n_views=100_000)

    defaults = DefaultStrategy()
    assert strategy.pause_refine_after_reset == defaults.reset_every - defaults.refine_every
    assert "would never refine" in caplog.text


def test_make_strategy_tuning_literals_are_keyword_arguments():
    cfg = SplatsConfig.from_dict({"primitive": "2dgs"})
    strategy = make_strategy(cfg, n_views=4, prune_opa=0.2, prune_scale3d=0.7, refine_scale2d_stop_iter=1234)

    assert strategy.prune_opa == pytest.approx(0.2)
    assert strategy.prune_scale3d == pytest.approx(0.7)
    assert strategy.refine_scale2d_stop_iter == 1234
PY
echo written
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_gaussian.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'collab_splats.splats.gaussian'`.

- [ ] **Step 3: Write `gaussian.py`**

```bash
cat > collab_splats/splats/gaussian.py <<'PY'
"""
Vanilla Gaussian primitives: one Gaussian per seed point, densified by a gsplat strategy.

``Gaussians`` owns its parameters, its per-tensor optimizers, its densification strategy and
its rendering, so the trainer drives it through the same members it drives ``Scaffold`` through
and never asks which representation it is training.
"""

import logging
import math

import numpy as np
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

from collab_splats.splats.rendering import render_gaussians

logger = logging.getLogger(__name__)

# Degree-0 spherical harmonic: rgb = SH_C0 * sh0 + 0.5
SH_C0 = 0.5 / math.sqrt(math.pi)


########################################
# Densification strategy
########################################


def make_strategy(
    cfg,
    n_views: int,
    *,
    prune_opa: float = 0.1,
    prune_scale3d: float = 0.5,
    refine_scale2d_stop_iter: int = 4000,
) -> MCMCStrategy | DefaultStrategy:
    """
    MCMC for 3dgs (budgeted, no gradient heuristics); Default with splatfacto's args for 2dgs.

    Args:
        cfg: the run's SplatsConfig (reads `primitive`, `cap_max`, `grow_grad2d`).
        n_views: number of training views; sets DefaultStrategy's post-reset refine pause.
        prune_opa: opacity below which a Gaussian is pruned (2dgs only).
        prune_scale3d: world-scale above which a Gaussian is pruned (2dgs only).
        refine_scale2d_stop_iter: step after which screen-scale splitting stops (2dgs only).

    Returns:
        An unstarted gsplat strategy; the caller still calls `initialize_state`.
    """
    if cfg.primitive == "3dgs":
        return MCMCStrategy(cap_max=cfg.cap_max, verbose=False)

    # splatfacto (nerfstudio @ 50e0e3c) non-default DefaultStrategy args. absgrad stays
    # False: the 2dgs backward writes .absgrad on means2d only, never on the
    # gradient_2dgs densify tensor this strategy reads (see the parity spec's verdict),
    # and grow_grad2d 2e-4 is the measured-good non-absgrad threshold.
    # gsplat gates refine on `step % reset_every >= pause_refine_after_reset`, so
    # splatfacto's n_views + 100 silently disables densification once n_views
    # >= reset_every - 100 (2900 at gsplat defaults). Cap it and say so.
    defaults = DefaultStrategy()
    pause = n_views + 100
    max_pause = defaults.reset_every - defaults.refine_every
    if pause > max_pause:
        logger.warning(
            "pause_refine_after_reset=%d (n_views+100) >= reset_every=%d would never refine; capped to %d",
            pause,
            defaults.reset_every,
            max_pause,
        )
        pause = max_pause

    return DefaultStrategy(
        absgrad=False,
        grow_grad2d=cfg.grow_grad2d,
        key_for_gradient="gradient_2dgs",
        prune_opa=prune_opa,
        prune_scale3d=prune_scale3d,
        refine_scale2d_stop_iter=refine_scale2d_stop_iter,
        pause_refine_after_reset=pause,
        verbose=False,
    )


########################################
# Model
########################################


class Gaussians:
    """
    Vanilla 3DGS / 2DGS primitives with their optimizers, strategy and rendering.

    Exposes the same members as ``Scaffold`` — `params`, `optimizers`, `schedulers`,
    `n_primitives`, `render`, `pre_backward`, `post_backward`, `denormalize`,
    `export_gaussians`, `checkpoint`, `from_checkpoint` — so the trainer needs no branch.
    """

    def __init__(
        self,
        cfg,
        points: np.ndarray,
        colors: np.ndarray,
        scene_scale: float,
        n_views: int,
        device: str,
        *,
        knn: int = 4,
        adam_eps: float = 1e-15,
        lr_decay: float = 0.01,
    ):
        """
        One Gaussian per seed point (scale from kNN spacing, color as SH DC) plus one Adam per parameter.

        Port of create_splats_with_optimizers, gsplat @ d2f5c0f examples/simple_trainer.py.

        Args:
            cfg: the run's SplatsConfig. Only `primitive`, `sh_degree`, `sh_degree_interval`,
                `init_opacity`, the six learning rates, `max_steps`, `cap_max` and `grow_grad2d`
                are read; nothing keeps a reference to it.
            points: (N, 3) float seed positions in the training frame. Needs at least `knn` points.
            colors: (N, 3) uint8 seed colors.
            scene_scale: camera extent of the training frame; scales the means lr and seeds the strategy.
            n_views: number of training views; sets the 2dgs refine pause.
            device: torch device string.
            knn: neighbors (including self) used for the initial scale; 4 means the 3 nearest.
            adam_eps: Adam epsilon, 1e-15 as upstream.
            lr_decay: total multiplicative decay of the means lr over the run, 0.01 = 100x down.
        """
        # Plain attributes rather than a cfg reference: trainer imports this module, so importing
        # SplatsConfig back would be circular, and from_checkpoint has only a plain dict to work from
        self.primitive = cfg.primitive
        self.sh_degree = cfg.sh_degree
        self.sh_degree_interval = cfg.sh_degree_interval
        self.device = device

        n_points = len(points)

        # Initial scale: mean distance to the (knn - 1) nearest neighbors, stored as log-scale
        neighbor_dists, _ = NearestNeighbors(n_neighbors=knn).fit(points).kneighbors(points)
        neighbor_sq_dists = neighbor_dists[:, 1:] ** 2
        mean_spacing = np.sqrt(neighbor_sq_dists.mean(-1))
        spacing = torch.from_numpy(mean_spacing).float()
        log_scales = torch.log(spacing).unsqueeze(-1).repeat(1, 3)

        # Color: RGB goes into the degree-0 SH band, higher bands start at zero
        rgb = torch.from_numpy(colors).float() / 255.0
        n_sh_coeffs = (cfg.sh_degree + 1) ** 2
        sh_coeffs = torch.zeros(n_points, n_sh_coeffs, 3)
        sh_coeffs[:, 0, :] = (rgb - 0.5) / SH_C0

        # Raw parameters: random orientation, logit-opacity so sigmoid gives init_opacity
        initial_opacities = torch.logit(torch.full((n_points,), cfg.init_opacity))
        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.from_numpy(points).float()),
                "scales": torch.nn.Parameter(log_scales),
                "quats": torch.nn.Parameter(torch.rand(n_points, 4)),
                "opacities": torch.nn.Parameter(initial_opacities),
                "sh0": torch.nn.Parameter(sh_coeffs[:, :1, :]),
                "shN": torch.nn.Parameter(sh_coeffs[:, 1:, :]),
            }
        ).to(device)

        # One Adam per parameter so the densification strategy can grow/prune optimizer state per tensor
        learning_rates = {
            "means": cfg.means_lr * scene_scale,
            "scales": cfg.scales_lr,
            "quats": cfg.quats_lr,
            "opacities": cfg.opacities_lr,
            "sh0": cfg.sh0_lr,
            "shN": cfg.shN_lr,
        }
        self.param_optimizers = {
            name: torch.optim.Adam([{"params": self.params[name], "lr": lr, "name": name}], eps=adam_eps)
            for name, lr in learning_rates.items()
        }
        self.optimizers = list(self.param_optimizers.values())

        # Only the positions decay; the other groups hold their lr for the whole run
        self.means_scheduler = ExponentialLR(
            self.param_optimizers["means"], gamma=lr_decay ** (1.0 / cfg.max_steps)
        )
        self.schedulers = [self.means_scheduler]

        # Densification: MCMC is budgeted and stateless, Default carries per-Gaussian statistics
        self.strategy = make_strategy(cfg, n_views)
        self.strategy.check_sanity(self.params, self.param_optimizers)
        if isinstance(self.strategy, MCMCStrategy):
            self.strategy_state = self.strategy.initialize_state()
        else:
            self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)

    @property
    def n_primitives(self) -> int:
        """
        Number of Gaussians currently in the model.
        """
        return len(self.params["means"])

    def activate(self) -> dict[str, Tensor]:
        """
        Activate raw parameters into the tensors the rasterizer takes.

        log-scales -> scales, logit-opacities -> opacities, SH bands concatenated. `colors` are
        SH coefficients, so the caller passes an integer `sh_degree` to the rasterizer.

        Returns:
            {"means" (N,3), "quats" (N,4), "scales" (N,3), "opacities" (N,), "colors" (N,K,3)}.
        """
        return {
            "means": self.params["means"],
            "quats": self.params["quats"],
            "scales": torch.exp(self.params["scales"]),
            "opacities": torch.sigmoid(self.params["opacities"]),
            "colors": torch.cat([self.params["sh0"], self.params["shN"]], dim=1),
        }

    def render(
        self,
        cam_to_world: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
        camera_id: Tensor,
        step: int | None = None,
        render_normals: bool = True,
        render_plane: bool = False,
    ) -> tuple[dict[str, Tensor], dict]:
        """
        Rasterize one view.

        Args:
            cam_to_world: (1, 4, 4) camera-to-world pose in the training frame.
            intrinsics: (1, 3, 3) camera matrix in pixels at this render's resolution.
            width: render width in pixels.
            height: render height in pixels.
            camera_id: (1,) long view index. Unused here; ``Scaffold`` needs it for its
                appearance embedding, and the trainer calls both through this signature.
            step: current training step, which unlocks SH bands progressively. None renders
                at the full `sh_degree` — what a finished model or a checkpoint wants.
            render_normals: render the per-Gaussian normal and its finite-differenced partner.
            render_plane: add PGSR's planar signals (3dgs only).

        Returns:
            (render dict, gsplat strategy info dict) — see `rendering.render_gaussians`.
        """
        sh_degree = self.sh_degree if step is None else min(step // self.sh_degree_interval, self.sh_degree)
        absgrad = isinstance(self.strategy, DefaultStrategy) and self.strategy.absgrad
        return render_gaussians(
            self.primitive,
            self.activate(),
            cam_to_world,
            intrinsics,
            width,
            height,
            sh_degree,
            absgrad,
            render_normals=render_normals,
            render_plane=render_plane,
        )

    def pre_backward(self, step: int, info: dict) -> None:
        """
        Retain the screen-space gradients DefaultStrategy densifies on. No-op under MCMC.

        Args:
            step: current training step.
            info: the gsplat info dict this step's render returned.

        Returns:
            None.
        """
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_pre_backward(self.params, self.param_optimizers, self.strategy_state, step, info)

    def post_backward(self, step: int, info: dict) -> None:
        """
        Densify / prune / relocate, after the optimizer step.

        Refine ops rebuild the Parameters, so stepping the optimizers afterwards would see
        `.grad=None` and silently skip them — the caller must step first (upstream order).
        MCMC reads the post-decay means lr, as upstream simple_trainer does.

        Args:
            step: current training step.
            info: the gsplat info dict this step's render returned.

        Returns:
            None.
        """
        if isinstance(self.strategy, MCMCStrategy):
            means_lr = self.means_scheduler.get_last_lr()[0]
            self.strategy.step_post_backward(
                self.params, self.param_optimizers, self.strategy_state, step, info, lr=means_lr
            )
        else:
            self.strategy.step_post_backward(
                self.params, self.param_optimizers, self.strategy_state, step, info, packed=False
            )

    def denormalize(self, center: np.ndarray, scale: float) -> None:
        """
        Undo ``utils.scene_normalization`` on the Gaussians, in place.

        Args:
            center: (3,) the center `scene_normalization` returned.
            scale: the scale `scene_normalization` returned.

        Returns:
            None — `params` is modified in place.
        """
        center_t = torch.as_tensor(center, dtype=torch.float32, device=self.params["means"].device)
        with torch.no_grad():
            self.params["means"].data = self.params["means"].data / scale + center_t
            self.params["scales"].data = self.params["scales"].data - math.log(scale)

    def export_gaussians(
        self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int
    ) -> dict[str, Tensor]:
        """
        The raw parameters the ply writer wants.

        Args:
            cam_to_world: (N, 4, 4) training poses. Unused here; ``Scaffold`` bakes its anchors
                against the cameras that saw them, and the caller invokes both the same way.
            intrinsics: (N, 3, 3) camera matrices. Unused here, as above.
            width: image width in pixels. Unused here, as above.
            height: image height in pixels. Unused here, as above.

        Returns:
            {"means", "scales", "quats", "opacities", "sh0", "shN"} — the parameters themselves,
            not copies. Scales are log, opacities are logits, colors are SH coefficients.
        """
        return dict(self.params)

    def checkpoint(self) -> dict:
        """
        The model half of ckpt.pt.

        Returns:
            {"splats": ParameterDict}. The trainer adds `config`, cameras and image ids.
        """
        return {"splats": self.params}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str) -> "Gaussians":
        """
        Rebuild a render-only model from a checkpoint.

        The result has no optimizers, no schedulers and no strategy: it renders and exports,
        it does not train.

        Args:
            ckpt: a loaded ckpt.pt holding `splats` and a plain-dict `config`.
            device: torch device string.

        Returns:
            A Gaussians instance whose `params` are the checkpoint's, on `device`.
        """
        model = cls.__new__(cls)
        config = ckpt["config"]
        model.primitive = config["primitive"]
        model.sh_degree = config["sh_degree"]
        model.sh_degree_interval = config["sh_degree_interval"]
        model.device = device
        model.params = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(tensor) for name, tensor in dict(ckpt["splats"]).items()}
        ).to(device)
        model.param_optimizers = {}
        model.optimizers = []
        model.schedulers = []
        model.means_scheduler = None
        model.strategy = None
        model.strategy_state = None
        return model
PY
echo written
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_gaussian.py -q
```

Expected: `16 passed`.

- [ ] ~~**Step 5: Point `rendering.py` at the new constant**~~ — **REMOVED, see C7-1.**

Do nothing here. `SH_DC_NORMALIZER` stays defined at `rendering.py:25` with all six of its
readers untouched; Task 9 Step 4 retires it. `gaussian.py` defines its own `SH_C0` and the two
constants hold the same value side by side until then. `rendering.py` must **not** import
`SH_C0` from `gaussian.py` — `gaussian.py` imports `rendering.py`, so that would be circular.

- [ ] **Step 6: Run the splats suite**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats -q
```

Expected: all pass. `trainer.py` still uses its own `init_gaussians_from_points`; Task 10 removes it.

- [ ] **Step 7: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/gaussian.py tests/splats/test_gaussian.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/gaussian.py tests/splats/test_gaussian.py && \
  git add collab_splats/splats/gaussian.py tests/splats/test_gaussian.py && \
  git commit --only collab_splats/splats/gaussian.py tests/splats/test_gaussian.py \
    -m "feat(splats): add the Gaussians model class"
```

---

## Task 8: Rename `AnchorField` to `Scaffold` and give it the model interface

Second half of the centerpiece. `AnchorField` already owns its parameters, optimizers and decode — it just exposes them under different names than `Gaussians` does, and the trainer bridges the gap with `if anchor_field is not None`. This task closes the gap.

Two pieces of the spec's phase 4 are pulled forward into this task, because the interface cannot exist without them:

- **`update_learning_rate` becomes schedulers.** With `update_learning_rate` still a method, `train()` would need a scaffold-only call at the top of the loop — exactly the branch this plan deletes. The replacement is `LambdaLR`, which is numerically identical: `LambdaLR` applies `lambda(0)` at construction and `lambda(step)` after each end-of-loop `.step()`, so the lr in force during step *k* is `lambda(k)` either way.
- **`bake_anchor_gaussians` moves here** from `outputs.py` as `Scaffold.export_gaussians`. Task 9 deletes the original.

> **Deviations from the spec:** `Scaffold.__init__` takes the full `SplatsConfig` (it reads `cfg.scaffold_config` itself) and a `lr_decay` keyword, so `train()` constructs either model with one call; `export_gaussians` takes `width` and `height` because the bake runs a frustum test.

**Files:**
- Modify: `collab_splats/splats/scaffold.py`
- Modify: `collab_splats/splats/trainer.py` (imports and constructs `AnchorField`)
- Modify: `collab_splats/splats/outputs.py` (in the rename sweep and the commit, though see below)
- Modify: `tests/splats/test_scaffold.py`

> **STATUS: COMPLETE (2026-09-06).** Implemented at `a7358c4c` / `8c4f837c`, then two code-quality
> fix rounds: `08050ada` and `5bf13f1f`. Spec review **SPEC COMPLIANT**. Code-quality review
> **APPROVED** at round 3, measured on pinned trees at both `08050ada` and `5bf13f1f`:
> `tests/splats` **247 passed, `PYTEST_RC=0`, 0 skipped** (`third_party/*` symlinked in, so the
> skip floor is real and not the fresh-tree downgrade); `test_scaffold.py` 68 passed.
>
> Round 1 killed 20/20 named mutants; the re-review reproduced all 17 of the survivor table plus
> M2hard and M23 with **zero regressions**, including the decider **M3** (`step_post_backward`
> before `accumulate`), killed by `test_post_backward_accumulates_before_it_refines`. Round 2 then
> killed four more that the re-review had found alive — `N6`, `N13`, `N14`, `N11` — each verified to
> have SURVIVED the parent, so the fix is what killed them. `N13` is the sharp one: the same wrong
> scaling slice inside **production `decode`**, on the frozen `splats.ply` surface.
>
> Round 3's `+195 / −169` diff was fully accounted for: 149 lines are a pure `field` → `model`
> rename (proved by rewriting the old file with `sed 's/\bfield\b/model/g'` and diffing — residual
> exactly 20 lines), and the 20 are the divider, the `_reachable_state` refactor, the schedulers
> assertion, six docstring expansions and the one functional line. **No assertion, fixture
> perturbation or `torch.no_grad()` block was deleted.** Independent invariants: `assert` count
> 202 → 203, `def` count 75 → 75, sorted `def test_` name set byte-identical.
>
> Two deliberate coverage trades, both ruled correct by measurement: un-pinning `self.schedulers`'
> **order** (`M25reorder` now survives; drop and duplicate still die) is safe because every consumer
> in `collab_splats/` and `tests/` only iterates — `trainer.py:411` copies then loops, `scaffold.py:780`
> assigns, and the only indexed `.schedulers[0]` in the tree is on `Gaussians`, a different class.
> And the `_field` helper keeps its name despite the `field` purge — renaming it is a **Task 16**
> question, to be done in the same commit that updates C12-7.
>
> Three gaps were measured and handed forward rather than fixed here, all **pre-existing at both
> SHAs**: the export↔decode parity blind spot (**C12-9**), unpinned offset slot identity with a
> verified one-line fix (**C12-10**), and the corrected account of what `_reachable_state` reaches
> (under C12-8).
>
> **Superseded detail below.** The original stamp read: Commits `a7358c4c` (+406) and
> `8c4f837c` (one added test). Verdict **SPEC COMPLIANT** — nothing missing, nothing extra,
> frozen surfaces untouched. Gates: 217 -> 225 -> 226 passed, `PYTEST_RC=0`, **0 skipped**, node-id
> diff pure additions with zero deletions; flake8 RC=0 at the repo's real settings; zero lines over
> 120. `LambdaLR` replacing `expon_lr` measured a numerical no-op — 71 steps x 6 groups, max
> relative error **1.087e-15**.
>
> **Four of the correction blocks below are wrong or incomplete. Read this before trusting them.**
>
> **C8-5's last clause is WRONG — do not delete the MLP optimizer step.** C8-5 says
> `anchor_field.mlp_optimizer.step()` / `.zero_grad(...)` are "now covered by the flat `optimizers`
> list — stepping both double-steps the MLP heads". Measured at `c90eea22`: after patch C8-11(a)
> the trainer's local `optimizers` is bound to `anchor_field.param_optimizers`, a **5-entry
> name -> Adam dict** that does not contain `mlp_optimizer`, and the loop iterates
> `optimizers.values()`. Those two lines at `trainer.py:585-586` are therefore the **only** thing
> training the MLP heads. Deleting them leaves every head untrained with **no failing test**.
> C8-11(d) already says this; C8-5's clause is struck. Both the implementer and the spec reviewer
> reached this independently.
>
> **C8-11(b) must NOT delete `means_key`.** It is still read at `trainer.py:608`
> (`n_primitives = len(gaussians[means_key])`, bound at `:409`).
>
> **C8-9 undercounts.** It names three tests covering **5** of the `.optimizers` sites in
> `tests/splats/test_scaffold.py`. There are **11**; the other six (lines 168, 169, 516, 529, 653,
> 663) are unnamed by the correction and all needed the `param_optimizers` rename.
>
> **C8-6 stands, and the code comment must be trimmed to match it.** The comment committed beside
> the MLP optimizer construction reads: "No lr= here: LambdaLR multiplies each group's OWN initial
> lr, so a zero placeholder would pin every head at zero forever." Sentence two is right; sentence
> one restates the bug C8-6 proved does not exist. Verified again on the pre-change tree: with
> `lr=0.0` still present the group lrs read `0.002 / 0.004 / 0.008 / 0.05`, because the per-group
> `lr` in `mlp_groups` shadows the constructor default, so `LambdaLR` never captured
> `initial_lr=0`. Trim to sentence two.
>
> One more, minor: Step 4's inline import was correctly hoisted to module top level, which is what
> CLAUDE.md's import rule and the brief's own rule elsewhere require. The step text should say so.
>
> **`Scaffold.export_gaussians` is a faithful move of `outputs.py:145 bake_anchor_gaussians`** —
> verified by AST, the bodies are identical statement for statement after renaming the receiver
> `anchor_field` -> `self`, with two non-behavioral differences: the `with torch.no_grad():` block
> became the `@torch.no_grad()` decorator at `scaffold.py:655`, and `SH_DC_NORMALIZER` became
> `SH_C0` (0 ULP, see C12-5). The original is now dead weight; Task 9/10 deletes it.

> **Corrections measured against the tree (2026-09-06).**
>
> 1. **This task is NOT parallel-safe with Task 7 — it depends on it.** Step 9 adds
>    `from collab_splats.splats.gaussian import SH_C0` to `scaffold.py`, and `SH_C0` exists
>    nowhere in the tree today: Task 7 creates it. Run Task 7 to completion first. The two tasks
>    are file-disjoint (Task 7 only creates `gaussian.py` and `test_gaussian.py`), which is why
>    they looked pairable, but a hard import edge is not a file collision and dispatching them
>    together races Task 8's test run against Task 7's commit.
> 2. **The Files list above was missing `trainer.py` and `outputs.py`.** The steps were already
>    right — the rename `sed` and the final `git commit --only` both name all four paths — but
>    the list did not, and `trainer.py:46` imports `AnchorField` with 20-odd `anchor_field` uses
>    below it. Renaming only inside `scaffold.py` would break the trainer import and take the
>    whole splats suite down.
> 3. ~~**The `sed` is a no-op on `outputs.py`**~~ — **REFUTED, see C8-7.** `outputs.py:195`
>    contains the string `AnchorField.decode` in a comment, so the rename `sed` does rewrite the
>    file. Keep it in the command and the commit — but not because it is inert.
> 4. Landmarks confirmed present: `ScaffoldConfig:34`, `ScaffoldMLPs:135`, `expon_lr:204`,
>    `AnchorField:237`, `update_learning_rate:326`, `AnchorStrategy:486`. ~~`expon_lr` is used
>    only inside `scaffold.py` and `tests/splats/test_scaffold.py`~~ — **REFUTED, see C8-8:**
>    `expon_lr` appears in `scaffold.py` only (`:204` def, `:208` docstring, `:335`, `:337`) and
>    in **no** test. The deletion is still safe; the stated reason was wrong.

> **SECOND CORRECTION PASS — measured against `1d1d036b` (2026-09-06, after Tasks 5's quality
> fixes landed). These override both the step text and the first correction block above.**
>
> Every line number here was measured at `1d1d036b`. Task 6 and Task 7 land before this task;
> Task 6 moves loss validation out of `trainer.py` and will shift every `trainer.py` number below
> by roughly 50-70 lines. **Re-grep for the symbol; never trust the number.** The `scaffold.py`,
> `test_scaffold.py`, `test_outputs.py` and `outputs.py` numbers survive Tasks 6 and 7 untouched.
>
> **C8-1 (blocker) — Step 4's closing sentence is false.** It says "Every later use of
> `self.cfg` in the class body already means the `ScaffoldConfig`, so nothing else in the
> constructor changes." The constructor body does not read `self.cfg` — it reads the **local**
> `cfg`, twenty times, at `scaffold.py:263, 264, 266, 280, 281, 287, 291, 292, 293, 294, 295,
> 303, 304, 305, 311, 319 (×2), 320, 321, 322, 323`. Once `cfg` is a `SplatsConfig` every one of
> those changes meaning. Introduce the `cfg_scaffold = cfg.scaffold_config` local that Step 6
> already assumes exists ("where `cfg_scaffold` is a local alias for `self.cfg` assigned at the
> top of `__init__`") and rebind all twenty reads to it.
>
> Nineteen of them fail loudly with `AttributeError` — `SplatsConfig` has no `voxel_size`,
> `n_offsets`, `feat_dim`, `anchor_lr`, `offset_lr`, `mlp_opacity_lr` and so on. **One does
> not.** `appearance_lr` exists on *both* dataclasses with different values:
> `SplatsConfig.appearance_lr = 1e-3` (`trainer.py:130`) versus
> `ScaffoldConfig.appearance_lr = 5e-2` (`scaffold.py:83`). A missed rebind at `scaffold.py:311`
> or `:323` silently trains the scaffold appearance embedding at **one fiftieth** of its rate,
> with no error and no failing test — it would surface only as a parity dPSNR at Task 11.
>
> **C8-2 (blocker) — the task invalidates most of `test_scaffold.py` and says nothing about
> it.** The file holds 48 tests. The `_field` helper at `:178-183` constructs
> `AnchorField(ScaffoldConfig(...), ...)` and has **28 call sites**; five more tests construct
> the class directly at `:123`, `:142`, `:143`, `:152`, `:167`. Every one of them passes a
> `ScaffoldConfig` positionally into a constructor that now demands a `SplatsConfig`. Rewrite
> `_field` to build a `SplatsConfig` (`SplatsConfig.from_dict({"representation": "scaffold",
> "scaffold": {...}, "max_steps": ...})`) and fix the five direct constructions; the 28 call
> sites then need no change. Do this in Step 1, before the rename, or Step 9 reports a wall of
> `AttributeError`s you cannot read.
>
> **C8-3 (blocker) — `tests/splats/test_outputs.py` is an undeclared caller.** `:124`, `:125`
> and `:129` import and construct `AnchorField` in a fixture. It is not in the Files list, not
> in the `sed`, and not in the `git commit --only`. This is the exact shape of the defect that
> bit Task 5. **CUDA is live in this container** (the suite runs with zero skips), so these are
> not skipped tests — they execute and they will break. Add the file to all three places.
>
> **C8-4 (blocker) — Step 5's grep misses two hits.** `grep -n "\.optimizers\[" ` finds only
> the *subscripted* uses. `AnchorStrategy` passes the mapping **unsubscripted** at
> `scaffold.py:688` and `:743`:
> `_update_param_with_optimizer(param_fn, optimizer_fn, field.params, field.optimizers)`.
> Left alone, gsplat receives the new flat `list` where it expects the name→optimizer `dict`
> and densification dies inside the strategy. Use `\.optimizers\b` and convert all four hits
> (`:297` the assignment, `:334`, `:688`, `:743`).
>
> **C8-5 (high) — five `trainer.py` sites break and Step 10's "splats suite green" cannot hold
> without them.** Measured at `1d1d036b`: `:463` `gaussians, optimizers = anchor_field.params,
> anchor_field.optimizers` (the flat list is now correct here — check the unpack still means
> what the trainer wants), `:464` `AnchorStrategy(cfg.scaffold_config, ...)` (Step 8 moves the
> strategy inside the constructor, so this line goes away), `:522`
> `anchor_field.update_learning_rate(step)` (the method is deleted in Step 6 — replace with the
> shared scheduler loop), `:550` and `:595` `anchor_field.decode(...)`, and `:660-661`
> `anchor_field.mlp_optimizer.step()` / `.zero_grad(...)` (now covered by the flat
> `optimizers` list — stepping both double-steps the MLP heads, exactly the Task 5 Correction C
> failure mode).
>
> **C8-6 (high) — Step 6's stated reason for dropping `lr=0.0` is wrong; make the change
> anyway.** The step says a zero initial lr "would pin every head at zero forever". Measured:
> `torch.optim.Adam([{"params": [p], "lr": 0.05}, {"params": [q]}], lr=0.0)` gives group lrs
> `[0.05, 0.0]` — the default only fills groups that omit the key, and **every** mlp group dict
> carries its own `"lr"` (`scaffold.py:303-305, :311`). So `lr=0.0` was never doing anything to
> these groups. Dropping it is correct hygiene and changes no number; do not go hunting for the
> zero-lr bug the step implies, because there isn't one.
>
> **C8-7 (medium) — first-block correction 3 is refuted.** `outputs.py` is not untouched by the
> rename `sed`: `outputs.py:195` reads `# mirroring the same guard in AnchorField.decode.` and
> the `sed` rewrites that comment to `Scaffold.decode`, which is the right outcome. The file was
> already in the `sed` and the commit; only the justification changes.
>
> **C8-8 (medium) — first-block correction 4 is refuted on `expon_lr`'s test coverage.**
> `expon_lr` appears four times, all in `scaffold.py` (`:204` definition, `:208` docstring,
> `:335`, `:337`), and **zero** times in `tests/splats/test_scaffold.py` or anywhere else. It is
> safe to delete, but "and its tests" in Step 6 names nothing — see C8-9 for the tests that
> actually break.
>
> **C8-9 (medium) — two `update_learning_rate` tests must be migrated, not deleted.** Step 6
> says "Delete the now-unused `expon_lr` function and its tests"; per C8-8 `expon_lr` has no
> tests, and the tests that actually break are the three that drive `update_learning_rate`.
> `:555 test_anchors_are_frozen_by_default` (asserts the anchors group starts at `lr == 0.0`)
> needs only `optimizers` → `param_optimizers`, and it guards a real shipped default — keep it.
> `:560 test_learning_rates_decay_per_head` is the **only** coverage of the four per-head
> schedules (`mlp_color > mlp_cov > mlp_opacity` at start, color/opacity/appearance decaying,
> cov flat, offsets decaying) — rewrite it to step the new `LambdaLR`s instead of calling
> `update_learning_rate`, do not drop it. `:579
> test_learning_rate_horizon_is_the_config_not_the_run_length` likewise: it pins the
> `lr_max_steps`-vs-run-length distinction that Step 6's own comment calls out.

> **THIRD CORRECTION PASS — measured against `235c954e` (2026-09-06, after Task 7 landed
> `gaussian.py`). These override both correction blocks above and the step text.**
>
> **C8-10 (blocker) — the anchor-position lr would decay TWICE per step.** Step 6 builds
> `self.anchor_scheduler = ExponentialLR(self.param_optimizers["anchors"], gamma=lr_decay **
> (1.0 / cfg.max_steps))` inside `Scaffold.__init__`. But `train()` **already builds exactly
> that scheduler on exactly that optimizer** and already steps it every iteration (re-grep; the
> numbers below are pre-Task-6):
>
> ```
> 407     # lr decay on the anchor / Gaussian positions (0.01x over the run)
> 408     lr_gamma = 0.01 ** (1.0 / cfg.max_steps)
> 409     means_key = "anchors" if anchor_field is not None else "means"
> 410     means_optimizer = optimizers[means_key]
> 411     means_scheduler = ExponentialLR(means_optimizer, gamma=lr_gamma)
> 412     schedulers = [means_scheduler]
> ...
> 596         for scheduler in schedulers:
> 597             scheduler.step()
> ```
>
> `lr_decay` defaults to `0.01` and `gamma` is the same expression, so after this task one Adam
> group would be driven by two identical `ExponentialLR`s and its lr would fall as `gamma**(2k)`
> rather than `gamma**k`. That is a silent change to `train()`'s numerics — a frozen surface —
> with no failing test; it surfaces only as a parity dPSNR at Task 11. The trainer's scaffold
> branch must consume the model's schedulers instead of building its own. See C8-11(b).
>
> **C8-11 (blocker) — the exact `trainer.py` patch.** C8-5 names the five broken sites but not
> the code, and is wrong about one of them. This is the measured, minimal edit; it is
> scaffold-branch-only and leaves the vanilla path byte-unchanged.
>
> *(a) Construction.* `Scaffold.__init__` now takes the full `SplatsConfig` and (Step 8) builds
> the strategy itself, so the three strategy lines go away:
>
> ```python
>     anchor_field = None
>     if cfg.representation == "scaffold":
>         anchor_field = Scaffold(cfg, points, colors, scene_scale, n_views, device)
>         gaussians, optimizers = anchor_field.params, anchor_field.param_optimizers
>         strategy = anchor_field.strategy
>         strategy_state = anchor_field.strategy_state
> ```
>
> Note `param_optimizers`, not `optimizers`: the trainer subscripts this mapping by name and
> iterates `optimizers.values()`, and the new flat `.optimizers` list breaks both.
>
> *(b) Schedulers* — the C8-10 fix:
>
> ```python
>     # lr decay on the Gaussian positions (0.01x over the run). Scaffold owns its own schedulers
>     # (anchors, offsets and the four MLP heads), so the trainer only collects them.
>     lr_gamma = 0.01 ** (1.0 / cfg.max_steps)
>     if anchor_field is not None:
>         schedulers = list(anchor_field.schedulers)
>     else:
>         means_scheduler = ExponentialLR(optimizers["means"], gamma=lr_gamma)
>         schedulers = [means_scheduler]
> ```
>
> `lr_gamma` stays — `CameraOpt.from_config(cfg, n_views, world_extent, scene_scale, lr_gamma,
> device)` on the next line still needs it. `means_scheduler` is read once more, at
> `means_lr_now = means_scheduler.get_last_lr()[0]` inside the `elif isinstance(strategy,
> MCMCStrategy)` branch; that branch is unreachable when `anchor_field is not None`, so leaving
> the name unbound on the scaffold path is correct, not a latent `NameError`.
>
> *(c) Top-of-loop lr update.* Delete the block and its five-line comment entirely — the comment
> is now wrong twice over (the schedules live in `Scaffold`, and Step 6's own argument is that
> `LambdaLR(k)` applied after the end-of-loop `.step()` is the lr that was in force during step
> *k*):
>
> ```python
>         if anchor_field is not None:
>             anchor_field.update_learning_rate(step)
> ```
>
> *(d) MLP optimizer step — **C8-5 is wrong here**.* C8-5 says these two lines are "now covered
> by the flat `optimizers` list — stepping both double-steps the MLP heads". They are not.
> After (a) the trainer's local `optimizers` is `param_optimizers`, the **dict** of five anchor
> tensors; `mlp_optimizer` lives only in the flat `.optimizers` **list**, which the trainer never
> iterates. Delete these lines and the MLP heads never train at all. Keep them:
>
> ```python
>         if anchor_field is not None:
>             anchor_field.mlp_optimizer.step()
>             anchor_field.mlp_optimizer.zero_grad(set_to_none=True)
> ```
>
> *(e)* The two `anchor_field.decode(...)` sites are unchanged — `decode` keeps its name.
>
> **C8-12 (high) — drop the now-unused `AnchorStrategy` import from `trainer.py`.** After (a)
> nothing in `trainer.py` references it and flake8 fails on F401. Re-grep for `ScaffoldConfig`
> before touching that one; it may still be read by `SplatsConfig`. The line is
> `from collab_splats.splats.scaffold import AnchorField, AnchorStrategy, ScaffoldConfig`.
>
> **C8-13 (medium) — Step 10's `black`, `isort` and `git commit --only` lists are still missing
> `tests/splats/test_outputs.py`.** C8-3 says to add it in three places; Step 10 names only four
> paths in all three commands. The correct final command carries five paths. Run `flake8` over
> the same five before committing.
>
> **C8-14 (medium) — Step 9's gate is far too narrow.** It runs only
> `tests/splats/test_scaffold.py`. Every defect in C8-10 and C8-11 is invisible to that file and
> visible only where `train()` is actually driven with a scaffold:
> `tests/splats/test_trainer.py::test_train_scaffold_runs_and_writes_outputs` (parametrised) and
> three `train(...)` calls in `tests/splats/test_outputs.py`. Gate on the whole `tests/splats`
> directory. The baseline at `235c954e` is **216 passed**; measure it again at `HEAD` rather than
> trusting the number.
>
> **C8-15 (low) — only two imports are genuinely new.** Measured in `scaffold.py`: `math` (:14),
> `numpy as np` (:17), `torch.nn.functional as F` (:19) and `Tensor` (:24) are already imported,
> so Step 7's `denormalize` and `export_gaussians` need no new stdlib imports, and
> `visible_anchors` exists at `:340` as Step 7 assumes. The new imports are exactly:
>
> ```python
> from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
>
> from collab_splats.splats.gaussian import SH_C0
> from collab_splats.splats.rendering import render_gaussians
> ```
>
> Both `ExponentialLR` (anchors) and `LambdaLR` (offsets, MLP heads) are used, so neither trips
> F401. `scaffold.py` imports neither today.
>
> **C8-16 (low) — `self.lr_schedule` dies with `update_learning_rate`.** It is read only at
> `:335` and `:337`, both inside that method. Delete the dict built in `__init__` too; its
> endpoint pairs are what `mlp_endpoints` and the `_decay_lambda` calls now carry.

- [ ] **Step 1: Write the failing test**

> **STALE AS OF `08050ada` — do not copy `_scaffold_model` out of this block.** Task 8's fix
> round deleted it: its `**overrides` splatted into the **top level** of `SplatsConfig.from_dict`
> rather than into the `"scaffold"` block, so `_scaffold_model(n_offsets=8)` looked like it
> configured the scaffold and silently did not. `tests/splats/test_scaffold.py` now has one
> builder, `_field`, which takes a `run=` parameter for top-level run keys. See C12-7 under
> Task 12. The rest of this step is history; the file it describes has moved on.

Append to `tests/splats/test_scaffold.py`:

```python
def _scaffold_model(**overrides):
    """
    A CPU Scaffold over a small deterministic seed cloud.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, (64, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (64, 3)).astype(np.uint8)
    cfg = SplatsConfig.from_dict(
        {"representation": "scaffold", "scaffold": {}, "max_steps": 100, **overrides}
    )
    return Scaffold(cfg, points, colors, scene_scale=2.0, n_views=4, device="cpu")


def test_scaffold_exposes_anchor_parameters():
    model = _scaffold_model()

    assert set(model.params) == {"anchors", "offsets", "anchor_feat", "scaling", "rotation"}
    assert model.n_primitives == len(model.params["anchors"])


def test_scaffold_optimizers_is_a_flat_list_covering_every_tensor_and_the_mlps():
    model = _scaffold_model()

    # Five anchor tensors plus the one multi-group MLP optimizer
    assert len(model.optimizers) == 6
    assert model.mlp_optimizer in model.optimizers
    assert set(model.param_optimizers) == set(model.params)


def test_scaffold_lambda_schedulers_reproduce_the_exponential_curve():
    model = _scaffold_model()
    lr_max_steps = model.cfg.lr_max_steps
    offset_optimizer = model.param_optimizers["offsets"]
    lr_init = offset_optimizer.param_groups[0]["lr"]

    # Step 0: the schedule has not moved
    assert lr_init == pytest.approx(model.cfg.offset_lr * 2.0, rel=1e-6)

    # Halfway along the schedule the lr is the geometric mean of the endpoints
    for _ in range(lr_max_steps // 2):
        for scheduler in model.schedulers:
            scheduler.step()
    expected = math.sqrt((model.cfg.offset_lr * 2.0) * (model.cfg.offset_lr_final * 2.0))
    assert offset_optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-4)


def test_scaffold_lambda_schedulers_hold_at_the_final_lr_past_the_horizon():
    model = _scaffold_model()
    offset_optimizer = model.param_optimizers["offsets"]

    for _ in range(model.cfg.lr_max_steps + 50):
        for scheduler in model.schedulers:
            scheduler.step()

    assert offset_optimizer.param_groups[0]["lr"] == pytest.approx(model.cfg.offset_lr_final * 2.0, rel=1e-4)


def test_scaffold_denormalize_inverts_the_sim3():
    model = _scaffold_model()
    center = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    scale = 0.25
    anchors_before = model.params["anchors"].detach().clone()
    scaling_before = model.params["scaling"].detach().clone()
    offsets_before = model.params["offsets"].detach().clone()

    model.denormalize(center, scale)

    assert torch.allclose(model.params["anchors"], anchors_before / scale + torch.from_numpy(center), atol=1e-5)
    assert torch.allclose(model.params["scaling"], scaling_before - math.log(scale), atol=1e-5)
    # Offsets are stored in units of the anchor's own extent, so they are scale-free
    assert torch.allclose(model.params["offsets"], offsets_before)


def test_scaffold_checkpoint_carries_the_mlps_and_the_voxel_size():
    model = _scaffold_model()
    ckpt = model.checkpoint()

    assert set(ckpt) == {"splats", "mlps", "voxel_size"}
    assert ckpt["voxel_size"] == pytest.approx(model.voxel_size)


def test_scaffold_checkpoint_round_trips():
    model = _scaffold_model()
    ckpt = model.checkpoint()
    ckpt["config"] = {"primitive": "3dgs", "scaffold": {}}

    restored = Scaffold.from_checkpoint(ckpt, "cpu")

    assert restored.n_primitives == model.n_primitives
    assert torch.allclose(restored.params["anchors"], model.params["anchors"])
    assert restored.optimizers == []
    assert restored.schedulers == []
    assert restored.strategy is None


def test_scaffold_export_gaussians_bakes_every_anchor():
    model = _scaffold_model()
    cam_to_world = torch.eye(4)[None].repeat(2, 1, 1)
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)

    baked = model.export_gaussians(cam_to_world, intrinsics, 64, 64)

    assert set(baked) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    assert len(baked["means"]) > 0
    # The ply writer wants raw forms: log scales, logit opacities, a single SH DC band
    assert baked["sh0"].shape[1] == 1
    assert baked["shN"].shape == (len(baked["means"]), 0, 3)
```

Extend the file's imports:

```python
import math

import numpy as np
import pytest
import torch

from collab_splats.splats.scaffold import AnchorStrategy, Scaffold, ScaffoldConfig, ScaffoldMLPs
from collab_splats.splats.trainer import SplatsConfig
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -q
```

Expected: collection error, `ImportError: cannot import name 'Scaffold'`.

- [ ] **Step 3: Rename the class**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  sed -i 's/\bAnchorField\b/Scaffold/g' collab_splats/splats/scaffold.py collab_splats/splats/trainer.py collab_splats/splats/outputs.py tests/splats/test_scaffold.py && \
  grep -rn "AnchorField" collab_splats tests evals docs --include='*.py' --include='*.ipynb' || echo "NONE LEFT"
```

Expected: `NONE LEFT`. Then fix the class docstring's first line, which now reads oddly:

```python
class Scaffold:
    """
    Scaffold-GS anchors: the trainable state, the per-view decode into neural Gaussians, and rendering.

    - ``params`` is the ParameterDict the densification strategy grows and prunes; the MLP heads live
      in ``mlps`` and are fixed-size, so they are optimized separately and never handed to the strategy.
    - ``decode`` returns the rasterizer inputs plus ``decode_index``, the (anchor * n_offsets + offset)
      slot each emitted Gaussian came from. Anchor densification is built entirely on that index.
    - There is no anchor opacity parameter: opacity is decoded per view by mlp_opacity.
    - Exposes the same members as ``Gaussians`` so the trainer needs no representation branch.
    """
```

- [ ] **Step 4: Change the constructor to take `SplatsConfig`**

```python
    def __init__(
        self,
        cfg,
        points: np.ndarray,
        colors: np.ndarray,
        scene_scale: float,
        n_views: int,
        device: str,
        *,
        lr_decay: float = 0.01,
    ):
        """
        Voxelize the seed cloud into anchors, build the MLP heads, optimizers, schedulers and strategy.

        Args:
            cfg: the run's SplatsConfig. `cfg.scaffold_config` holds the parsed ScaffoldConfig;
                `cfg.primitive` and `cfg.max_steps` are read too. No reference is kept.
            points: (N, 3) float seed positions in the training frame.
            colors: (N, 3) uint8 seed colors. Unused — the color head learns color from scratch,
                and the argument exists so both models construct identically.
            scene_scale: camera extent of the training frame; scales the anchor and offset lrs.
            n_views: number of training views; sizes the appearance embedding.
            device: torch device string.
            lr_decay: total multiplicative decay of the anchor-position lr over the run.
        """
        self.cfg = cfg.scaffold_config
        self.primitive = cfg.primitive
        self.device = device
```

Every later use of `self.cfg` in the class body already means the `ScaffoldConfig`, so nothing else in the constructor changes except the two blocks below.

- [ ] **Step 5: Rename `self.optimizers` to `self.param_optimizers` and build the flat list**

Inside `__init__`, the dict comprehension currently assigned to `self.optimizers` becomes `self.param_optimizers`. After `self.mlp_optimizer` is built, add:

```python
        # Flat list for the trainer (one optimizer per anchor tensor plus the multi-group MLP one);
        # param_optimizers stays because gsplat's _update_param_with_optimizer needs the name mapping
        self.optimizers = [*self.param_optimizers.values(), self.mlp_optimizer]
```

Fix the internal readers:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -n "\.optimizers\[" collab_splats/splats/scaffold.py
```

Every hit (`self.optimizers["offsets"]`, `field.optimizers[...]` inside `AnchorStrategy`) becomes `param_optimizers`. There must be **no** subscripted `.optimizers[` left in the file when this step is done.

- [ ] **Step 6: Replace `update_learning_rate` with `LambdaLR` schedulers**

Delete the `update_learning_rate` method. In `__init__`, after `self.lr_schedule` is built, replace that dict and the method with:

```python
        # Anchor positions decay over the RUN, like the vanilla means lr
        self.anchor_scheduler = ExponentialLR(
            self.param_optimizers["anchors"], gamma=lr_decay ** (1.0 / cfg.max_steps)
        )

        # Offsets and the MLP heads follow upstream's own log-space curves, whose horizon is
        # cfg.lr_max_steps rather than the run length: a shorter run stops partway down the curve.
        # LambdaLR applies lambda(0) at construction and lambda(step) after each end-of-loop step(),
        # so the lr in force during step k is lambda(k) — identical to setting it at the top of the
        # iteration, which is what the update_learning_rate this replaces did.
        offset_lr = cfg_scaffold.offset_lr * scene_scale
        offset_lr_final = cfg_scaffold.offset_lr_final * scene_scale
        self.offset_scheduler = LambdaLR(
            self.param_optimizers["offsets"],
            lr_lambda=_decay_lambda(offset_lr, offset_lr_final, cfg_scaffold.lr_max_steps),
        )
        mlp_endpoints = {
            "mlp_opacity": (cfg_scaffold.mlp_opacity_lr, cfg_scaffold.mlp_opacity_lr_final),
            "mlp_cov": (cfg_scaffold.mlp_cov_lr, cfg_scaffold.mlp_cov_lr),
            "mlp_color": (cfg_scaffold.mlp_color_lr, cfg_scaffold.mlp_color_lr_final),
            "embedding_appearance": (cfg_scaffold.appearance_lr, cfg_scaffold.appearance_lr_final),
        }
        self.mlp_scheduler = LambdaLR(
            self.mlp_optimizer,
            lr_lambda=[
                _decay_lambda(*mlp_endpoints[group["name"]], cfg_scaffold.lr_max_steps)
                for group in self.mlp_optimizer.param_groups
            ],
        )
        self.schedulers = [self.anchor_scheduler, self.offset_scheduler, self.mlp_scheduler]
```

where `cfg_scaffold` is a local alias for `self.cfg` assigned at the top of `__init__` for readability, and the module-level helper replaces `expon_lr`:

```python
def _decay_lambda(lr_init: float, lr_final: float, lr_max_steps: int):
    """
    LambdaLR multiplier reproducing 3DGS's log-space lr decay from lr_init to lr_final.

    LambdaLR multiplies the optimizer's *initial* lr, so the multiplier is the ratio raised to the
    progress fraction. Non-positive endpoints hold the lr flat, matching the guard in the
    ``expon_lr`` this replaces (upstream's delay term is inert: lr_delay_steps defaults to 0).

    Args:
        lr_init: lr at step 0, already scaled by whatever the caller scales by.
        lr_final: lr at `lr_max_steps` and beyond.
        lr_max_steps: schedule horizon in steps; upstream keys every schedule to a fixed 30k.

    Returns:
        A `step -> multiplier` callable for `torch.optim.lr_scheduler.LambdaLR`.
    """
    if lr_init <= 0.0 or lr_final <= 0.0:
        return lambda step: 1.0
    ratio = lr_final / lr_init
    horizon = max(lr_max_steps, 1)
    return lambda step: ratio ** min(step / horizon, 1.0)
```

**`mlp_optimizer` must be built with each group's own `lr`**, not the current `lr=0.0` placeholder — `LambdaLR` multiplies the group's initial lr, so a zero initial lr would pin every head at zero forever. The group dicts already carry `"lr"`; drop the `lr=0.0` argument from the `Adam(...)` call:

```python
        self.mlp_optimizer = torch.optim.Adam(mlp_groups, eps=1e-15)
```

Delete the now-unused `expon_lr` function and its tests. Add to the imports:

```python
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
```

- [ ] **Step 7: Add the interface methods**

Append to the `Scaffold` class, after `decode`:

```python
    @property
    def n_primitives(self) -> int:
        """
        Number of anchors currently in the model. Decoded Gaussian count is per view and varies.
        """
        return len(self.params["anchors"])

    def render(
        self,
        cam_to_world: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
        camera_id: Tensor,
        step: int | None = None,
        render_normals: bool = True,
        render_plane: bool = False,
    ) -> tuple[dict[str, Tensor], dict]:
        """
        Decode this view's neural Gaussians and rasterize them.

        Args:
            cam_to_world: (1, 4, 4) camera-to-world pose in the training frame.
            intrinsics: (1, 3, 3) camera matrix in pixels at this render's resolution.
            width: render width in pixels.
            height: render height in pixels.
            camera_id: (1,) long view index; feeds the appearance embedding when it is on.
            step: current training step. Unused here — the SH schedule is vanilla-only, and the
                argument exists so both models render through one signature.
            render_normals: render the per-Gaussian normal and its finite-differenced partner.
            render_plane: add PGSR's planar signals (3dgs only).

        Returns:
            (render dict, gsplat strategy info dict). The render dict carries the decoded
            `log_scales` and `opacities` the regularizers read — this representation has no
            opacity or scale parameter for them to read instead. The info dict carries
            `decode_index`, `decoded_opacities` and `visible_ids` for the strategy.
        """
        decoded, decode_index = self.decode(self.primitive, cam_to_world, intrinsics, width, height, camera_id)
        render, info = render_gaussians(
            self.primitive,
            decoded,
            cam_to_world,
            intrinsics,
            width,
            height,
            sh_degree=None,
            absgrad=False,
            render_normals=render_normals,
            render_plane=render_plane,
        )

        # The regularizers read decoded quantities; the strategy reads the decode bookkeeping
        render["log_scales"] = decoded["log_scales"]
        render["opacities"] = decoded["opacities"]
        info["decode_index"] = decode_index
        info["decoded_opacities"] = decoded["opacities"]
        info["visible_ids"] = decoded["visible_ids"]
        return render, info

    def pre_backward(self, step: int, info: dict) -> None:
        """
        Retain the screen-space gradient the anchor strategy accumulates after backward.

        Args:
            step: current training step. Unused; the signature is shared with ``Gaussians``.
            info: the gsplat info dict this step's render returned.

        Returns:
            None.
        """
        info[self.strategy.key_for_gradient].retain_grad()

    def post_backward(self, step: int, info: dict) -> None:
        """
        Accumulate anchor statistics, then grow and prune.

        Reading the retained gradient here rather than before the optimizer step is safe:
        `zero_grad(set_to_none=True)` clears the *parameters'* gradients, and the tensor read
        here is a retained non-leaf whose `.grad` no optimizer touches.

        Args:
            step: current training step.
            info: the gsplat info dict this step's render returned.

        Returns:
            None.
        """
        if self.strategy.should_accumulate(step):
            self.strategy.accumulate(
                self.strategy_state, info, info["decode_index"], info["decoded_opacities"], info["visible_ids"]
            )
        self.strategy.step_post_backward(self, self.strategy_state, step)

    def denormalize(self, center: np.ndarray, scale: float) -> None:
        """
        Undo ``utils.scene_normalization`` on the anchors, in place.

        Both halves of the log ``scaling`` shift by -log(scale). Offsets are stored in units of
        the anchor's own extent, so they are scale-free and untouched. The MLP heads survive
        because their only view input is a unit direction, and every decode that writes an output
        happens after this call.

        Args:
            center: (3,) the center `scene_normalization` returned.
            scale: the scale `scene_normalization` returned.

        Returns:
            None — `params` is modified in place.
        """
        center_t = torch.as_tensor(center, dtype=torch.float32, device=self.params["anchors"].device)
        with torch.no_grad():
            self.params["anchors"].data = self.params["anchors"].data / scale + center_t
            self.params["scaling"].data = self.params["scaling"].data - math.log(scale)

    @torch.no_grad()
    def export_gaussians(
        self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int
    ) -> dict[str, Tensor]:
        """
        Decode each anchor once at its mean observed view direction, for a static viewer-loadable ply.

        Anchors seen by no training camera fall back to the direction of the nearest camera.
        Colors are baked into the degree-0 SH band; there are no higher bands to write. Lossy by
        construction: the trained model is view-dependent, and renders from the checkpoint are not.

        Args:
            cam_to_world: (N, 4, 4) training poses in the output frame.
            intrinsics: (N, 3, 3) camera matrices at the training image size.
            width: training image width in pixels.
            height: training image height in pixels.

        Returns:
            {"means" (M,3), "scales" (M,3) log, "quats" (M,4), "opacities" (M,) logit,
            "sh0" (M,1,3), "shN" (M,0,3)} — the raw forms every ply viewer expects.
        """
        device = self.params["anchors"].device
        anchors = self.params["anchors"].detach()

        # Accumulate the unit direction to every camera that can see each anchor
        direction_sum = torch.zeros_like(anchors)
        seen_count = torch.zeros(len(anchors), device=device)
        for view in range(len(cam_to_world)):
            visible = self.visible_anchors(cam_to_world[view : view + 1], intrinsics[view : view + 1], width, height)
            to_camera = anchors - cam_to_world[view, :3, 3]
            unit = to_camera / to_camera.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            direction_sum[visible] += unit[visible]
            seen_count[visible] += 1

        # Unseen anchors: use the nearest camera's direction rather than dropping them from the ply
        unseen = seen_count == 0
        if bool(unseen.any()):
            camera_centers = cam_to_world[:, :3, 3]
            nearest = torch.cdist(anchors[unseen], camera_centers).argmin(dim=1)
            to_nearest = anchors[unseen] - camera_centers[nearest]
            direction_sum[unseen] = to_nearest / to_nearest.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            seen_count[unseen] = 1

        mean_direction = direction_sum / seen_count[:, None]
        mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # One decode at that direction, bypassing the frustum filter so every anchor is written
        features = torch.cat([self.params["anchor_feat"].detach(), mean_direction], dim=-1)
        camera_id = torch.zeros(1, dtype=torch.long, device=device)
        neural_opacity, cov, color = self.mlps(features, camera_id)

        n_offsets = self.cfg.n_offsets
        scaling = torch.exp(self.params["scaling"].detach())
        offsets = self.params["offsets"].detach()
        keep = (neural_opacity > 0).reshape(-1)

        # An all-closed decode would write an empty ply, which gsplat's export_splats cannot
        # serialize (its shN reshape needs at least one splat). Keep the most opaque offset,
        # mirroring the same guard in decode.
        if not bool(keep.any()):
            keep = torch.zeros_like(keep)
            keep[neural_opacity.reshape(-1).argmax()] = True

        means = (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep]
        cov = cov.reshape(-1, 7)[keep]
        scales = scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[keep] * torch.sigmoid(cov[:, :3])
        quats = F.normalize(cov[:, 3:7], dim=-1)
        opacities = neural_opacity.reshape(-1)[keep]
        colors = color.reshape(-1, 3)[keep]

        return {
            "means": means,
            "scales": torch.log(scales.clamp_min(1e-12)),
            "quats": quats,
            "opacities": torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),
            "sh0": ((colors - 0.5) / SH_C0).unsqueeze(1),
            "shN": torch.zeros(len(means), 0, 3, device=means.device),
        }

    def checkpoint(self) -> dict:
        """
        The model half of ckpt.pt.

        Returns:
            {"splats": ParameterDict, "mlps": state_dict, "voxel_size": float}. The trainer adds
            `config`, cameras and image ids.
        """
        return {"splats": self.params, "mlps": self.mlps.state_dict(), "voxel_size": self.voxel_size}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str) -> "Scaffold":
        """
        Rebuild a render-only model from a checkpoint.

        The result has no optimizers, no schedulers and no strategy: it renders and exports,
        it does not train.

        Args:
            ckpt: a loaded ckpt.pt holding `splats`, `mlps`, `voxel_size` and a plain-dict `config`.
            device: torch device string.

        Returns:
            A Scaffold instance whose parameters and heads are the checkpoint's, on `device`.
        """
        model = cls.__new__(cls)
        config = ckpt["config"]
        model.cfg = ScaffoldConfig.from_dict(config.get("scaffold") or {})
        model.primitive = config["primitive"]
        model.device = device
        model.voxel_size = ckpt["voxel_size"]
        model.params = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(tensor) for name, tensor in dict(ckpt["splats"]).items()}
        ).to(device)

        # n_views comes from the saved appearance embedding, which is the only view-sized head
        appearance_weight = ckpt["mlps"].get("embedding_appearance.weight")
        n_views = 0 if appearance_weight is None else len(appearance_weight)
        model.mlps = ScaffoldMLPs(model.cfg, n_views=n_views).to(device)
        model.mlps.load_state_dict(ckpt["mlps"])

        model.param_optimizers = {}
        model.mlp_optimizer = None
        model.optimizers = []
        model.schedulers = []
        model.strategy = None
        model.strategy_state = None
        return model
```

Add to the module imports:

```python
from collab_splats.splats.gaussian import SH_C0
from collab_splats.splats.rendering import render_gaussians
```

**Import direction check:** `scaffold.py` -> `gaussian.py` -> `rendering.py` -> `pgsr.py`. No cycle. `gaussian.py` must never import `scaffold.py`.

- [ ] **Step 8: Build the strategy inside `Scaffold.__init__`**

At the end of `__init__`, replacing what `train()` currently does:

```python
        # Anchor densification: grows into under-covered cells and prunes anchors whose offsets shut
        self.strategy = AnchorStrategy(self.cfg, self.primitive, self.voxel_size)
        anchor_state = self.strategy.initialize_state(len(self.params["anchors"]))
        self.strategy_state = {name: value.to(device) for name, value in anchor_state.items()}
```

`AnchorStrategy` is defined below `Scaffold` in the file; that is fine, the reference is resolved at call time.

- [ ] **Step 9: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -q
```

Expected: all pass, including the 8 new tests. Tests that exercised `expon_lr` or `update_learning_rate` are deleted — the two `LambdaLR` value tests above replace them.

- [ ] **Step 10: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/scaffold.py collab_splats/splats/trainer.py collab_splats/splats/outputs.py tests/splats/test_scaffold.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/scaffold.py collab_splats/splats/trainer.py collab_splats/splats/outputs.py tests/splats/test_scaffold.py && \
  git commit --only collab_splats/splats/scaffold.py collab_splats/splats/trainer.py collab_splats/splats/outputs.py tests/splats/test_scaffold.py \
    -m "refactor(splats): rename AnchorField to Scaffold and give it the model interface"
```

Note `trainer.py` and `outputs.py` still call the old entry points in places — they are rewritten in Tasks 9 and 10. This commit only needs the splats suite green.

---

## Task 9: Fold `outputs.py` into `rendering.py` and retire `splats.zarr` — MERGED INTO TASK 10, DO NOT EXECUTE STANDALONE

`outputs.py` exists because rendering-for-training and rendering-for-artifacts were different code paths. With `model.render` they are the same call, so the writer is three functions in `rendering.py` and the file goes away.

The store goes with it. `splats.zarr` held one render per training view so the mesh stage could fuse without touching the model — but the checkpoint already holds the model, and with cameras added it holds everything the zarr did. Re-rendering 300 views takes seconds on the GPU the splats stage already requires. Two artifacts describing one model is one too many, and the zarr is where a stale render can silently disagree with the checkpoint it came from.

> **This task changes an on-disk artifact.** `<backend>/splats/splats.zarr` is no longer written, and `ckpt.pt` gains `cam_to_world`, `intrinsics`, `image_ids`, `image_size` and loses `pose_adjust`. Tasks 17 and 18 move the readers. Between this task and Task 18 the mesh stage's `source: splats` path is broken — that is expected and Task 19's gate is where it comes back.

**Files:**
- Modify: `collab_splats/splats/rendering.py`
- Delete: `collab_splats/splats/outputs.py`
- Modify: `tests/splats/test_rendering.py`
- Delete: `tests/splats/test_outputs.py`

> **DEFECT RESOLVED — re-measured at `c90eea22` (2026-09-06), after Task 8 committed.**
>
> The defect recorded here was real: `collab_splats/splats/trainer.py:43` imports
> `write_splat_outputs` and calls it at `:622`, so deleting `outputs.py` breaks
> `import collab_splats` and takes the whole `tests/splats` suite down with it. `trainer.py`
> appeared in neither this task's Files list nor its commit.
>
> **Resolution: this task is merged into Task 10.** They are one task. Three measurements
> settle it, none of which were available when the defect was first written:
>
> 1. **This task's own new code is already written against the model interface.** Its
>    `write_outputs` body calls `model.export_gaussians(...)`, `model.checkpoint()` and
>    `model.n_primitives`. `train()` binds no `model` today — at `c90eea22` it still binds
>    `anchor_field`, `gaussians`, `optimizers`, `strategy`, `strategy_state` separately and
>    branches on `anchor_field is not None` throughout. **Task 10 is what creates `model`.**
>    That is also why this task's Step 6 admitted its six new tests "cannot run yet": they
>    cannot, and no ordering fixes it except doing Task 10 first.
>
> 2. **`outputs.py` holds four of the representation branches Task 10 exists to delete**, plus a
>    fifth via `render_all_views(..., anchor_field=...)`:
>
>    ```
>    outputs.py:244  ply_baked = anchor_field is not None
>    outputs.py:288  n_anchors=(0 if anchor_field is None else len(anchor_field.params["anchors"]))
>    outputs.py:299  n_gaussians = int(len(gaussians["anchors" if anchor_field is not None else "means"]))
>    outputs.py:318  unit = "gaussians" if anchor_field is None else "anchors"
>    outputs.py:273  if anchor_field is not None:  # checkpoint["mlps"], checkpoint["voxel_size"]
>    ```
>
>    Task 10's stated premise is that **every** `if anchor_field is not None` disappears. Run
>    alone it cannot deliver that — it would have to keep handing `anchor_field=` to a branching
>    writer, i.e. move the branches rather than delete them, and invent an interim contract that
>    this task then immediately deletes.
>
> 3. **The duplication this task removes already exists.** Task 8 moved
>    `outputs.py:145 bake_anchor_gaussians` onto the model as `Scaffold.export_gaussians`
>    (`scaffold.py:656`). Verified by AST: after renaming the receiver `anchor_field` -> `self`,
>    the two bodies are **identical statement for statement**, with exactly two differences,
>    neither behavioral — the `with torch.no_grad():` block became the `@torch.no_grad()`
>    decorator at `scaffold.py:655`, and `SH_DC_NORMALIZER` became `SH_C0`, which C12-5 measured
>    at **0 ULP**. So `bake_anchor_gaussians` is dead weight the moment `train()` can call
>    `model.export_gaussians` — which is, again, Task 10.
>
> **What the merged task owns**, in this order, as one commit:
>
> 1. Rewrite `train()` to the model interface (Task 10's steps) so `model` exists.
> 2. Fold `outputs.py` into `rendering.py` as `write_outputs` / `render_views` /
>    `load_checkpoint` (this task's steps), dropping `bake_anchor_gaussians` in favor of
>    `model.export_gaussians`.
> 3. Rewire `trainer.py:43` (the import) and `trainer.py:622` (the call site) — **the seam
>    neither task owned before**, which is what made this a defect.
> 4. Delete `collab_splats/splats/outputs.py` and `tests/splats/test_outputs.py`.
> 5. Only now run this task's six new tests. They execute for the first time here.
>
> Combined Files list: modify `trainer.py`, `test_trainer.py`, `rendering.py`,
> `test_rendering.py`; delete `outputs.py`, `test_outputs.py`. About 2,100 lines in scope —
> large, but it is the smallest unit that never leaves the tree unimportable, and the whole
> point of the plan's Task 10 is that the branches die in one place.
>
> The other two candidate resolutions were measured and rejected. A thin re-export shim (b)
> adds a file whose only purpose is to be deleted one task later, and still leaves the six tests
> unrunnable. Widening this task to own the two `trainer.py` lines (c) fixes the import but not
> the tests, since `model` still would not exist.
>
> Minor, same area: `collab_splats/splats/__init__.py`'s `GSPLAT_COMMIT` comment reads
> "recorded in every splats.zarr for provenance", which becomes false once `splats.zarr` is
> retired. It is a comment, not an import, so it breaks nothing — sweep it in Task 15.

> **CORRECTION C9-1 (measured at `235c954e`, 2026-09-06) — this task's `SH_DC_NORMALIZER`
> line numbers are stale.**
>
> Task 7's spec review measured the readers directly. C7-1 (Task 7) gives them as
> `trainer.py:41` and `:315`; the tree has them at `trainer.py:46` and `:246`. The file set is
> right, the numbers are not. Full reader set at `235c954e`:
>
> ```
> collab_splats/splats/trainer.py:46, :246
> collab_splats/splats/outputs.py:24, :213
> tests/splats/test_trainer.py:13, :221
> ```
>
> Re-grep for the symbol before touching any of them — Tasks 6, 7 and 8 each moved `trainer.py`
> after those numbers were written. Same class of staleness as C7-4's `trainer.py:349-384`,
> which measured at `280-315`.
>
> Note also that `gaussian.py` (added by Task 7) now holds `SH_C0 = 0.5 / math.sqrt(math.pi)`,
> bit-identical to `rendering.py:25`'s `SH_DC_NORMALIZER` (both `0x1.20dd750429b6dp-2`, verified
> by hex repr, not just `==`). When this task retires the duplicate, `SH_C0` is the survivor —
> Ground Rules §9 names it as the only sanctioned module-level numeric constant in the package.

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_rendering.py`:

```python
def _trained_stub(tmp_path):
    """
    A 4-view scene plus a 200-point seed cloud, trained for 3 steps — enough to write outputs.
    """
    scene = make_scene(n_views=4, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "cap_max": 500, "losses": {}})
    train(
        cfg,
        scene["images"],
        scene["world_to_cam"],
        scene["intrinsics"],
        scene["points"],
        scene["colors"],
        tmp_path,
    )
    return scene, cfg


def test_write_outputs_writes_three_artifacts_and_no_zarr(tmp_path):
    _trained_stub(tmp_path)

    assert (tmp_path / "splats.ply").exists()
    assert (tmp_path / "ckpt.pt").exists()
    assert (tmp_path / "splats_quality_report.json").exists()
    assert not (tmp_path / "splats.zarr").exists()


def test_checkpoint_is_self_contained(tmp_path):
    scene, _ = _trained_stub(tmp_path)
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)

    assert set(ckpt) >= {"splats", "config", "cam_to_world", "intrinsics", "image_ids", "image_size", "appearance"}
    assert "pose_adjust" not in ckpt
    assert ckpt["cam_to_world"].shape == (4, 4, 4)
    assert ckpt["intrinsics"].shape == (4, 3, 3)
    assert ckpt["image_ids"] == [0, 1, 2, 3]
    assert tuple(ckpt["image_size"]) == (32, 32)


def test_quality_report_keeps_its_schema(tmp_path):
    _trained_stub(tmp_path)
    report = json.loads((tmp_path / "splats_quality_report.json").read_text())

    assert set(report) == {"summary", "per_frame"}
    assert set(report["summary"]) >= {"psnr", "ssim", "n_gaussians", "seconds", "final_losses", "config"}
    assert len(report["per_frame"]) == 4
    assert set(report["per_frame"][0]) == {"image_id", "psnr", "ssim"}


def test_load_checkpoint_round_trips_the_model_and_the_cameras(tmp_path):
    _trained_stub(tmp_path)

    model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width) = load_checkpoint(
        tmp_path / "ckpt.pt", "cpu"
    )

    assert isinstance(model, Gaussians)
    # appearance_opt defaults false, so the rebuilt module is the identity on both halves
    assert isinstance(camera_opt, CameraOpt)
    assert camera_opt.appearance is None and camera_opt.rotation is None
    assert cam_to_world.shape == (4, 4, 4)
    assert intrinsics.shape == (4, 3, 3)
    assert image_ids == [0, 1, 2, 3]
    assert (height, width) == (32, 32)


def test_load_checkpoint_restores_the_color_affine_but_never_the_pose_half(tmp_path):
    scene = make_scene(n_views=4, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict(
        {"max_steps": 3, "cap_max": 500, "losses": {}, "pose_opt": True, "appearance_opt": True}
    )
    train(
        cfg,
        scene["images"],
        scene["world_to_cam"],
        scene["intrinsics"],
        scene["points"],
        scene["colors"],
        tmp_path,
    )
    saved = torch.load(tmp_path / "ckpt.pt", weights_only=False)["appearance"]

    _, camera_opt, *_ = load_checkpoint(tmp_path / "ckpt.pt", "cpu")

    # The affine comes back; the pose deltas do not — write_outputs already folded them into
    # cam_to_world, so a restored pose half would apply them twice
    assert camera_opt.appearance is not None
    assert torch.allclose(camera_opt.appearance.weight, saved["weight"].cpu())
    assert camera_opt.translation is None and camera_opt.rotation is None


def test_render_views_yields_one_render_per_view(tmp_path):
    _trained_stub(tmp_path)
    model, camera_opt, cam_to_world, intrinsics, _, (height, width) = load_checkpoint(tmp_path / "ckpt.pt", "cpu")

    renders = list(render_views(model, camera_opt, cam_to_world, intrinsics, height, width))

    assert len(renders) == 4
    assert set(renders[0]) >= {"rgb", "depth", "alpha", "normal"}
    assert renders[0]["rgb"].shape == (1, 32, 32, 3)
    # A generator, not a list: the mesh stage streams 300 views through it
    assert isinstance(render_views(model, camera_opt, cam_to_world, intrinsics, height, width), types.GeneratorType)


def test_render_views_yields_median_depth_for_2dgs(tmp_path):
    scene = make_scene(n_views=2, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"primitive": "2dgs", "max_steps": 3, "losses": {}})
    train(
        cfg,
        scene["images"],
        scene["world_to_cam"],
        scene["intrinsics"],
        scene["points"],
        scene["colors"],
        tmp_path,
    )
    model, camera_opt, cam_to_world, intrinsics, _, (height, width) = load_checkpoint(tmp_path / "ckpt.pt", "cpu")

    render = list(render_views(model, camera_opt, cam_to_world, intrinsics, height, width))[0]

    assert "median_depth" in render
```

Extend the file's imports:

```python
import json
import types

import torch

from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.cameras import CameraOpt
from collab_splats.splats.rendering import load_checkpoint, render_views, write_outputs
from collab_splats.splats.trainer import SplatsConfig, train
from tests.splats.synthetic import make_scene
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_rendering.py -q
```

Expected: collection error, `ImportError: cannot import name 'render_views'`.

- [ ] **Step 3: Add the three functions to `rendering.py`**

Append to `collab_splats/splats/rendering.py`:

```python
########################################################################################
# Outputs
########################################################################################


# `@torch.no_grad()` as a DECORATOR, never a `with` block inside the body. Grad mode is
# process-global, and a generator suspended at a `yield` inside a `with` has not exited it — so a
# partially consumed generator (a `zip` that ends first, a bare `next`, a `break`) would leave
# autograd disabled for every later caller in the process. The decorator form scopes grad to the
# body only and restores it around each yield.
@torch.no_grad()
def render_views(
    model,
    camera_opt: CameraOpt,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    height: int,
    width: int,
):
    """
    Re-render every view from a trained model, one at a time.

    A generator, not a list: 300 renders at 1080p cost ~23 B/px, about 14 GB held at once
    inside a 46.6 GB container. The caller consumes each render and drops it.

    Cameras are already pose-corrected — the checkpoint stores the poses that were rendered,
    so the pose half of `camera_opt` is never applied here (it would double-apply the deltas).

    Args:
        model: a `Gaussians` or `Scaffold`, in eval use (no strategy, no optimizers needed).
        camera_opt: the run's `CameraOpt`. Its color affine is applied to `rgb` before the
            render is yielded; a module without the appearance half returns `rgb` unchanged.
        cam_to_world: (N, 4, 4) poses in the output world frame.
        intrinsics: (N, 3, 3) camera matrices at (height, width).
        height: render height in pixels.
        width: render width in pixels.

    Yields:
        One render dict per view: `rgb` (1, H, W, 3) in [0, 1], `depth`, `alpha`, `normal`,
        and `median_depth` for 2dgs.
    """
    device = cam_to_world.device
    for view in range(len(cam_to_world)):
        camera_id = torch.tensor([view], device=device)
        render, _ = model.render(
            cam_to_world[view : view + 1],
            intrinsics[view : view + 1],
            width,
            height,
            camera_id,
            step=None,
        )
        render["rgb"] = camera_opt.color(render["rgb"], camera_id).clamp(0, 1)
        yield render


def write_outputs(
    cfg,
    model,
    refine,
    images: np.ndarray,
    image_ids: list[int],
    cam_to_world: Tensor,
    intrinsics: Tensor,
    out_dir: Path,
    seconds: float,
    final_losses: dict[str, float],
    *,
    training_cam_to_world: Tensor,
) -> None:
    """
    Write splats.ply, ckpt.pt and splats_quality_report.json to out_dir.

    Args:
        cfg: the run's SplatsConfig; serialized into both the checkpoint and the report.
        model: the trained `Gaussians` or `Scaffold`, already denormalized.
        refine: the run's `CameraOpt`. Only its color affine is checkpointed — the pose
            deltas are already folded into `cam_to_world`.
        images: (N, H, W, 3) uint8 training frames, scored against the re-renders.
        image_ids: frame indices, in render order.
        cam_to_world: (N, 4, 4) pose-corrected poses in the output world frame.
        intrinsics: (N, 3, 3) camera matrices at the training resolution.
        out_dir: the stage's output directory; created if absent.
        seconds: wall-clock training time, for the report summary.
        final_losses: the last step's single-view loss snapshot — a snapshot, not an average.
        training_cam_to_world: (N, 4, 4) the same cameras WITHOUT the learned pose deltas, in
            the output world frame — the poses splats.ply is baked against. The checkpoint and
            the re-renders take the corrected `cam_to_world`; do not collapse the two.

    Returns:
        None.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(cfg)
    n_views, height, width = images.shape[:3]

    # splats.ply: standard 3DGS ply (raw log-scales / logit-opacities, as every viewer expects).
    # Scaffold has no static Gaussians, so its export bakes one Gaussian set from the anchors —
    # lossy, and the reason a scaffold ply and a scaffold render do not match.
    export_splats(
        **model.export_gaussians(cam_to_world, intrinsics, width, height),
        format="ply",
        save_to=str(out_dir / "splats.ply"),
    )

    # ckpt.pt: self-contained — model, cameras, image size and config, everything a re-render needs
    checkpoint = model.checkpoint()
    checkpoint["splats"] = {name: param.detach().cpu() for name, param in checkpoint["splats"].items()}
    checkpoint["config"] = config_dict
    checkpoint["cam_to_world"] = cam_to_world.detach().cpu()
    checkpoint["intrinsics"] = intrinsics.detach().cpu()
    checkpoint["image_ids"] = list(image_ids)
    checkpoint["image_size"] = (height, width)
    checkpoint["appearance"] = None if refine.appearance is None else refine.appearance.state_dict()
    torch.save(checkpoint, out_dir / "ckpt.pt")

    # splats_quality_report.json: per-view psnr/ssim scored in memory as the renders stream past
    per_frame = []
    renders = render_views(model, refine, cam_to_world, intrinsics, height, width)
    for view, render in zip(image_ids, progress(renders, desc="splats render", total=n_views)):
        # This leaves the generator suspended by design — which is why the `no_grad` fix must
        # live on the `def` of `render_views` and not at the call sites. Listifying here is
        # ~14 GB at 1080p x 300 views.
        target = torch.from_numpy(images[view]).to(render["rgb"].device).float()[None] / 255.0
        mse = F.mse_loss(render["rgb"], target).item()
        ssim_distance = ssim_loss(render["rgb"].permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)).item()
        per_frame.append(
            {
                "image_id": view,
                "psnr": 10 * np.log10(1.0 / max(mse, 1e-12)),
                "ssim": 1.0 - ssim_distance,
            }
        )

    mean_psnr = float(np.mean([frame["psnr"] for frame in per_frame]))
    mean_ssim = float(np.mean([frame["ssim"] for frame in per_frame]))
    summary = {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "n_gaussians": model.n_primitives,
        "seconds": round(seconds, 1),
        "final_losses": final_losses,
        "config": config_dict,
    }
    report = {"summary": summary, "per_frame": per_frame}
    (out_dir / "splats_quality_report.json").write_text(json.dumps(report, indent=2))

    # Scaffold counts ANCHORS, and its rendered primitive count is a per-view quantity, so the
    # log names the unit rather than implying every model reports the same thing. The unit is a
    # per-model fact, so it comes OFF the model as a class attribute -- see the amendment note
    # below; do not reintroduce a name check here.
    unit = model.primitive_unit
    logger.info(
        "splats: %d %s, psnr %.2f, ssim %.3f, %.0fs -> %s",
        model.n_primitives,
        unit,
        mean_psnr,
        mean_ssim,
        seconds,
        out_dir,
    )


def load_checkpoint(path: Path, device: str):
    """
    Rebuild a render-only model and its cameras from a ckpt.pt.

    Args:
        path: the checkpoint written by `write_outputs`.
        device: torch device string for the model and cameras.

    Returns:
        (model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width)).
        The model has no optimizers and no strategy: it renders and exports, it does not train.
        `camera_opt` carries the color affine only, and is the identity when the run had none.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    config = ckpt["config"]

    # Representation picks the class; both rebuild from the same key
    model_class = Scaffold if config["representation"] == "scaffold" else Gaussians
    model = model_class.from_checkpoint(ckpt, device)

    # Color affine only: the pose deltas are already baked into the saved cam_to_world, so
    # rebuilding the pose half here would apply them a second time
    camera_opt = CameraOpt(
        len(ckpt["image_ids"]),
        optimize_pose=False,
        optimize_appearance=ckpt["appearance"] is not None,
    ).to(device)
    if ckpt["appearance"] is not None:
        camera_opt.appearance.load_state_dict(ckpt["appearance"])

    cam_to_world = ckpt["cam_to_world"].to(device).float()
    intrinsics = ckpt["intrinsics"].to(device).float()
    height, width = ckpt["image_size"]
    return model, camera_opt, cam_to_world, intrinsics, list(ckpt["image_ids"]), (int(height), int(width))
```

**Import direction:** `rendering.py` now imports `Gaussians` and `Scaffold`, which import `rendering.py` back. Break the cycle by importing them **inside** `load_checkpoint` — it is the one function that needs them, this is the codebase's documented exception for imports that would otherwise be circular, and a comment says so:

Add these two lines as the first statements of the function body you just wrote, directly under
its docstring and above `ckpt = torch.load(...)`, with the comment:

```python
    # Local import: both model modules import this one for render_gaussians, so a module-level
    # import here would be circular. load_checkpoint is the only site that needs the classes.
    from collab_splats.splats.gaussian import Gaussians
    from collab_splats.splats.scaffold import Scaffold
```

Do **not** add `from collab_splats.splats.gaussian import Gaussians` at module level — the
import fails at collection time with `ImportError: cannot import name 'render_gaussians' from
partially initialized module`, and the traceback points at whichever module happened to be
imported first, not at the cycle.

Add the module-level imports the new code needs:

```python
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.exporter import export_splats
from gsplat.losses import ssim_loss

from collab_splats.splats.cameras import CameraOpt
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Delete what the new code replaces**

From `rendering.py`, delete:
- `SH_DC_NORMALIZER` — `gaussian.SH_C0` is the same constant under the name the spec asks for. Confirm nothing still reads it:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "SH_DC_NORMALIZER" collab_splats tests evals docs || echo "NONE LEFT"
```

- `activate_vanilla` — replaced by `Gaussians.activate` in Task 7.
- `render_view` — replaced by `Gaussians.render` in Task 7.

Then delete the old module and its test file:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  git rm collab_splats/splats/outputs.py tests/splats/test_outputs.py
```

Anything worth keeping from `test_outputs.py` has already been rewritten into `test_rendering.py` in Step 1; the zarr-layout tests are deleted outright, because there is no store to have a layout.

- [ ] **Step 5: Update the module docstring**

```python
"""
Rasterization and the splat stage's artifacts.

- `render_gaussians` is the one call into gsplat: 3dgs via `rasterization` (antialiased) with
  per-Gaussian normals as extra signals, 2dgs via `rasterization_2dgs`.
- `render_views` re-renders a trained model view by view; `write_outputs` writes splats.ply,
  ckpt.pt and splats_quality_report.json; `load_checkpoint` reads the checkpoint back.

Ported from gsplat @ d2f5c0f, examples/simple_trainer.py.
"""
```

- [ ] **Step 6: Run the tests to verify they pass**

`train()` still calls `write_splat_outputs` at this point — Task 10 rewrites it. To keep this task's tests runnable now, apply the one-line bridge at the bottom of `train()`:

```python
    write_outputs(
        cfg, model, refine, images, list(range(n_views)), cam_to_world, intrinsics, out_dir, seconds, final_losses
    )
```

If `train()` has not yet been converted to build `model`/`refine` (it has not — that is Task 10), this task's six new tests cannot run yet. **Run them at the end of Task 10 instead**, and for now gate on the rest of the suite still importing:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -c "
import collab_splats, collab_splats.splats.rendering as r
print(collab_splats.__file__)
print(sorted(n for n in ('render_views', 'write_outputs', 'load_checkpoint') if hasattr(r, n)))
"
```

Expected: the worktree path, then `['load_checkpoint', 'render_views', 'write_outputs']`.

- [ ] **Step 7: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/rendering.py tests/splats/test_rendering.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/rendering.py tests/splats/test_rendering.py && \
  git commit --only collab_splats/splats/rendering.py collab_splats/splats/outputs.py tests/splats/test_rendering.py tests/splats/test_outputs.py \
    -m "refactor(splats): fold outputs.py into rendering.py and retire splats.zarr"
```

---

## Task 10: Rewrite `train()` with no representation branches, and fold `outputs.py` into `rendering.py` (ABSORBS TASK 9)

The payoff. Every `if anchor_field is not None` and every `if cfg.representation == "scaffold"` in the trainer disappears, because both models now answer the same ten members. What remains is three phases: setup, loop, finish.

**Task 9 is part of this task.** Read Task 9's prose, its on-disk-artifact warning, its steps and its
correction blocks as if they appeared here — they are this task's steps 2 and 4 (see the merged
ordering below). Task 9's own head carries the measurements that forced the merge; this block carries
the execution order.

> **SPEC-COMPLIANCE REVIEW OUTCOME — recorded 2026-09-06, reviewed at `aaadb0e1`.**
> Verdict **CHANGES REQUESTED (narrow)**. Zero deviations from this task's body: every retained
> helper call keeps its arguments, order and defaults, and all 15 C10 board items are CLOSED.
> `SplatsConfig`'s 24 fields are identical, `configs/base.yaml` is untouched, and `train()`
> numerics hold (Task 11's parity gate ran early against this tree — all 7 configs PASS, worst
> dPSNR 1.98e-07). Counts: 247 passed / 0 skipped at `bed9c5c1` -> **249 passed / 0 skipped** at
> `aaadb0e1`; the caller gate is 3 failed / 31 passed on both sides with the same pre-existing
> `KeyError: 'splat_max_depth_frac'`.
>
> What it found was **three frozen-surface movements the implementer did not report**. All three
> are prescribed by this task's body, so the body and Ground Rule §7 contradicted each other.
> Adjudicated against §7's own text:
>
> | Movement | Verdict |
> | --- | --- |
> | `ckpt.pt["appearance"]` holds `refine.appearance.state_dict()` where the parent held the whole `refine.state_dict()` | **REFUTED — no action.** §7 sanctions `ckpt.pt` changes explicitly. The parent had **no reader of that key at all** (measured at `bed9c5c1`: one comment, one writer, two test asserts, zero `load_state_dict`), and `load_checkpoint` (`rendering.py:415-418`) is symmetric with the writer. |
> | `splats_quality_report.json` loses `per_frame[].n_decoded` and `summary.n_decoded_mean` for scaffold | **CONFIRMED — blocker.** §7 freezes this schema. `n_decoded` appears **nowhere in this plan**, so it was dropped silently, not by direction. No successor exists — a scaffold's rendered primitive count is a per-view quantity nothing else records — and no test guards it: `test_quality_report_keeps_its_schema` (`test_rendering.py:371`) scores a **vanilla** stub only. Fix round 1. |
> | `splats.ply` bytes move for scaffold + `pose_opt` | **CONFIRMED — blocker.** §7 freezes the ply byte-for-byte. `train()` passes one `corrected` tensor to `write_outputs`, which feeds it to both the checkpoint (right) and `model.export_gaussians` (wrong); the parent called `bake_anchor_gaussians(anchor_field, cam_to_world, ...)` with the **raw** poses (`outputs.py:246`). `Gaussians.export_gaussians` ignores its camera arguments so vanilla is unaffected; `Scaffold.export_gaussians` uses them for anchor visibility and the baked mean view direction, so scaffold bytes move. The parity harness missed it because both its scaffold configs set `pose_opt: False`. Fix round 1. |
>
> **The `corrected` tensor is correct and stays.** It is how `ckpt.pt` gains folded pose deltas
> instead of a separate `pose_adjust` — the §7-sanctioned change — and Task 18's mesh adapter
> reads it back. The fix threads the **raw** poses to the ply export only, and pins the asymmetry
> with a test. Neither blocker changes this task's prescribed structure.

**Files:**
- Modify: `collab_splats/splats/trainer.py`
- Modify: `collab_splats/splats/rendering.py` (from Task 9)
- Delete: `collab_splats/splats/outputs.py` (from Task 9)
- Modify: `tests/splats/test_trainer.py`
- Modify: `tests/splats/test_rendering.py` (from Task 9)
- Delete: `tests/splats/test_outputs.py` (from Task 9)

> **This list was incomplete — corrected 2026-09-06 by measuring `git show --name-status
> aaadb0e1`.** The commit touches **14** files; the six above are the ones this plan named. The
> other eight are consequences of retiring `outputs.py` and of the model interface landing, and
> they are expected, not scope creep — but a reviewer working from the list above would flag every
> one of them. They are:
>
> | File | ± | Why it moves (read off the diff, not inferred) |
> | --- | --- | --- |
> | `collab_splats/splats/pgsr.py` | +52 | `render_neighbor` and its `numpy` import land here. This is **Task 14's precondition, already satisfied** — Task 14 must not re-add it. |
> | `collab_splats/splats/losses.py` | +48 | **gains** two helpers lifted out of the trainer: `rescale_depth_units(losses, scale)` and `neighbor_selection(spec)`. |
> | `collab_splats/splats/__init__.py` | 14 | `__all__` goes from `["GSPLAT_COMMIT", "SplatsConfig", "train"]` to add `Gaussians`, `Scaffold` and `load_checkpoint`. It never re-exported `outputs`. |
> | `collab_splats/splats/gaussian.py` | 18 | **comment text only.** `SH_C0`'s note becomes "the sole definition" now that `rendering.SH_DC_NORMALIZER` is gone; `make_strategy`'s "temporary duplicate of `trainer.py`" note becomes "the one factory"; the `activate_vanilla` duplicate note is deleted. No code. |
> | `collab_splats/splats/scaffold.py` | 5 | **docstring text only** — two references to `init_gaussians_from_points` and `splats.zarr` retargeted at `Gaussians` and `rendering.render_views`. No code. |
> | `tests/splats/test_gaussian.py` | +48 | one test: `test_render_forwards_every_positional_in_its_own_slot`. |
> | `tests/splats/test_scaffold.py` | 7 | swaps `trainer.denormalize_anchors` for `utils.denormalize_cameras` at one call site. |
> | `tests/test_cu121_migration.py` | 4 | its importable-module list swaps `collab_splats.splats.outputs` for `gaussian`, `scaffold` and `pgsr`. |
>
> Four of those eight rows were **wrong on first writing** — this plan asserted that `gaussian.py`
> and `scaffold.py` gained model members, that `losses.py` *lost* plumbing, and that `__init__.py`
> dropped `outputs` re-exports. All four were corrected by reading the diff. Treat any "why it
> moves" claim in this document as a hypothesis until you have run `git diff` yourself.

> **MERGED SCOPE — recorded 2026-09-06, measured at `c90eea22`.** One commit, in this order.
> Combined file set is roughly 2,100 lines; that is the price of the merge and it was weighed
> against the two alternatives (a thin `outputs.py` shim, and widening Task 9 to own the trainer
> seam) — both measured and rejected in Task 9's resolution block.
>
> 1. **Rewrite `train()` onto the model interface.** `model = Gaussians(...) | Scaffold(...)`,
>    `model.render` / `pre_backward` / `post_backward` / `checkpoint` / `export_gaussians` /
>    `n_primitives`. This is the step that first binds `model` in `train()` — Task 9's writer body
>    calls `model.export_gaussians(...)`, `model.checkpoint()` and `model.n_primitives`, none of
>    which exist before it runs. That is why Task 9's own Step 6 admitted its six new tests
>    "cannot run yet".
> 2. **Fold `outputs.py` into `rendering.py`** as Task 9 specifies (three functions, no
>    `anchor_field=` parameter, `splats.zarr` retired, `ckpt.pt` gains cameras).
> 3. **Rewire `trainer.py:43` and `:622`** — the import and the call site. **This seam is the
>    reason the two tasks cannot be separated: it belonged to neither task before.** Task 9's Files
>    list never named `trainer.py`; this task's never named `outputs.py`.
> 4. **Delete `outputs.py` and `tests/splats/test_outputs.py`.** Not before step 3 — deleting first
>    breaks `import collab_splats` and takes the whole `tests/splats` suite down with it.
> 5. **Only now run Task 9's six new tests.** They need `model` bound in `train()` (step 1) and the
>    writer moved (step 2).
>
> **Why not either order separately.** Task 9 first breaks the import at step 4 and cannot run its
> own tests at step 5. Task 10 first must keep handing `anchor_field=` to a still-branching writer,
> which *moves* the branches into the new `train()` instead of deleting them and invents an interim
> contract Task 9 would immediately delete.
>
> **`outputs.py` holds five of the branches this task exists to delete**, so they are not two
> disjoint branch sets — they are one:
> ```
> outputs.py:244  ply_baked = anchor_field is not None
> outputs.py:273  if anchor_field is not None:      # checkpoint["mlps"], checkpoint["voxel_size"]
> outputs.py:288  n_anchors=(0 if anchor_field is None else len(anchor_field.params["anchors"]))
> outputs.py:299  n_gaussians = int(len(gaussians["anchors" if anchor_field is not None else "means"]))
> outputs.py:318  unit = "gaussians" if anchor_field is None else "anchors"
> ```
>
> **`bake_anchor_gaussians` is already gone from the model side.** Task 8 (`a7358c4c`) moved it onto
> `Scaffold.export_gaussians` — AST-verified identical statement for statement modulo the receiver
> rename, `with torch.no_grad():` becoming the `@torch.no_grad()` decorator at `scaffold.py:655`,
> and `SH_DC_NORMALIZER` becoming `SH_C0` (0 ULP, C12-5). `outputs.py:145` is dead weight this task
> deletes; do not port it a second time.

> **CORRECTION PASS — measured at `235c954e` (2026-09-06), carried over from the Task 6 and
> Task 7 reviews. These add to this task's scope; they do not rewrite its steps.**
>
> **C10-1, C10-2 and C10-3 are now CLOSED by Task 7's fix round (`c414eff7`), which landed the
> guards in `tests/splats/test_gaussian.py` before this task deletes the trainer's copies —
> exactly the ordering C10-1 demanded. Verify they are still present before deleting, then treat
> the three blocks below as history, not as work.** What `c414eff7` added, by test name:
>
> | Correction | Guard now in `test_gaussian.py` |
> |---|---|
> | C10-1 (defaults unpinned) | `test_make_strategy_uses_splatfactos_tuning_literals_by_default` (`prune_opa` 0.1, `prune_scale3d` 0.5, `refine_scale2d_stop_iter` 4000), `test_gaussians_use_upstreams_adam_epsilon_by_default` (1e-15), `test_gaussians_decay_the_means_lr_by_one_hundredth_over_the_run` (`lr_decay` 0.01), `test_gaussians_initial_scale_uses_the_three_nearest_neighbors_by_default` (`knn` 4) |
> | C10-2 (keyword-only unguarded) | `test_make_strategy_tuning_literals_are_keyword_only` and `test_gaussians_init_tuning_literals_are_keyword_only` — both assert `pytest.raises(TypeError, match="positional")` on a positional call, which is the property the old test's name claimed but never tested |
> | C10-8 (splatfacto parity constants lose their only assertions) | CLOSED — `test_make_strategy_uses_splatfactos_tuning_literals_by_default` at `test_gaussian.py:193` (the three asserts at `:199-201`) asserts all three (`prune_opa` 0.1, `prune_scale3d` 0.5, `refine_scale2d_stop_iter` 4000). Confirm it is still there, then delete `trainer.make_strategy`. |
> | C10-3 (model methods uncovered) | `test_render_warms_up_the_sh_degree_and_forwards_the_strategys_absgrad` covers the SH warm-up and `absgrad`; `test_post_backward_dispatches_mcmc_with_the_means_lr_and_default_without` covers the strategy dispatch; `means_scheduler` is covered by the `lr_decay` test above |
>
> **C10-3 is now FULLY closed by Task 7's second fix round (`31a2495b`).** It was partly closed at
> `c414eff7` — `pre_backward` and `strategy_state` had zero direct coverage there, and making
> `pre_backward` a no-op still passed. `31a2495b` added the coverage as
> **two** tests, not one. **Corrected 2026-09-06 by measurement — the single test name this plan
> previously gave here exists at neither SHA, and its description of what that test asserts was
> also wrong.** The real coverage is
> `test_gaussians_seed_the_default_strategy_state_with_the_scene_scale` (`test_gaussian.py:380`),
> which asserts `strategy_state["scene_scale"] == approx(2.0)` on the `DefaultStrategy` branch and
> `"scene_scale" not in mcmc.strategy_state` on the MCMC branch, plus
> `test_pre_backward_dispatches_to_the_default_strategy_and_no_ops_under_mcmc`
> (`test_gaussian.py:391`), which monkeypatches `step_pre_backward` and pins the step argument.
> **There is no `retain_grad` assertion here and there should not be** — gsplat does the
> `retain_grad` inside `step_pre_backward`, so dispatch is all `Gaussians.pre_backward` owns.
> (`Scaffold.pre_backward` is the method that calls `retain_grad` directly; different contract.) The scene-scale guard is
> the load-bearing one — gsplat's `DefaultStrategy` prunes on `prune_scale3d * scene_scale`, so a
> wrong value prunes a different Gaussian set with no exception and no red test, which is exactly
> the Ground Rules §7 silent-numerics failure mode. **Verify both tests are present before deleting
> the trainer's call sites, then treat C10-3 as history.**
>
> The same round also killed four other survivors that bear on this task: `adam_eps` is now pinned
> exactly rather than through `pytest.approx` (at an expected `1e-15` the `approx` default `abs=1e-12`
> floor dominates entirely, so the old assertion read "eps is below about 1e-12" and `0.0` satisfied
> it); `absgrad` is now proven to be *read from* the strategy rather than merely forwarded as a
> constant, via a second `_model(primitive="2dgs", ...)` whose `DefaultStrategy.absgrad` is flipped —
> the default `_model` builds an **MCMCStrategy** (`SplatsConfig.primitive` defaults to `"3dgs"`), and
> `render` computes `absgrad = isinstance(self.strategy, DefaultStrategy) and self.strategy.absgrad`,
> so flipping `absgrad` on the MCMC instance is inert; the `post_backward` oracle lambdas now capture
> `args[3]` (the step), so forwarding a hardcoded `step=0` no longer survives; and `render_plane` is
> pinned alongside `render_normals`.
>
> **C10-1 (blocker) — the §9 defaults lifted into `gaussian.py` are pinned by nothing, and this
> task deletes the copies that currently keep them honest.**
>
> Task 7's spec reviewer mutation-probed the 16 tests in `tests/splats/test_gaussian.py` against
> `gaussian.py`. Mutating `prune_opa` 0.1 -> 0.9, `adam_eps` 1e-15 -> 1e-8, `lr_decay`
> 0.01 -> 0.5 and `knn` 4 -> 8 each leaves **16/16 passing**.
> `test_make_strategy_picks_default_with_splatfacto_arguments_for_2dgs` asserts `absgrad`,
> `key_for_gradient`, `grow_grad2d` and `pause_refine_after_reset` — never the three pruning
> defaults.
>
> That is harmless only while `trainer.py`'s own `make_strategy` copy still carries the real
> literals, because a drift would then show up as a diff between two live copies. **This task
> deletes that copy.** After it lands, a drifted default in `gaussian.py` is a silent production
> numerics change with nothing between it and a training run — and `train()`'s numerics are a
> frozen surface (Ground Rules §7).
>
> Add default-value assertions to `tests/splats/test_gaussian.py` **before** deleting the
> trainer's copy, and assert them against the literals measured off `trainer.py`, not off
> `gaussian.py` — otherwise the test pins whatever drift already happened.
>
> **C10-2 — `test_make_strategy_tuning_literals_are_keyword_arguments` does not test its own
> name.** Deleting the bare `*` from `make_strategy`'s signature (making the three params
> positional-or-keyword) leaves all 16 tests green: the test passes them by keyword, which works
> either way. Ground Rules §9's *keyword-only* property is currently unguarded anywhere in the
> tree. If §9 is meant to hold across the plan, the guard has to come from a signature-
> introspection test (`inspect.signature(...).parameters[...].kind is KEYWORD_ONLY`) or a lint
> rule. This task is the natural place, since it is where the trainer stops carrying a second
> copy of the same signature.
>
> **C10-3 — `render`, `pre_backward`, `post_backward`, `means_scheduler` and `strategy_state`
> have zero coverage.** Making `pre_backward` a no-op, or making `render` ignore `step` and
> always use the full `sh_degree`, both leave 16/16 green. The SH-degree warm-up at
> `gaussian.py:248` is the one behavior `render` adds over `rendering.py:238 render_view`, and
> **this task deletes that warm-up logic from the trainer's call site** — so it goes from
> double-covered-by-accident to uncovered. Test it here.
>
> **C10-4 — the last schedule-shaping logic still lives in `trainer.py` and hand-rolls the loss
> spec vocabulary.** The `normalize_scene` block (pre-Task-6 numbers `trainer.py:377-384`;
> re-grep) copies `cfg.losses` and divides `distortion["weight"]` and `distortion["end_weight"]`
> by the normalization scale. That is a rule about how a loss spec is shaped, expressed by
> enumerating by hand the exact two keys `losses.py`'s `loss_weight` now owns. If a third
> scheduling key is ever added, `loss_weight` will honor it and this block will silently skip
> it. Task 6's premise is that `losses.py` owns every loss rule, so this belongs beside
> `loss_weight` — e.g. `def rescale_depth_units(losses: dict[str, dict], scale: float) -> dict[str, dict]`.
> It was deferred out of Task 6 because Task 8 held the writer claim on `trainer.py`.
>
> **Whoever moves it MUST keep the defensive copy.** `cfg.losses` may be the caller's own yaml
> dict rather than `default_losses()`'s return, and `asdict(cfg)` lands in `ckpt.pt` and the zarr
> attrs. Mutating in place would corrupt both.
>
> **C10-5 — two pgsr spec defaults live outside `losses.py`.** `num_multi_view` (5) and
> `max_points` (20000) are read at the trainer call site (pre-Task-6 `trainer.py:435-436`;
> re-grep) while `geo`, `ncc`, `pixel_noise_threshold`, `num_sample` and `patch_size` all default
> inside `losses.py`. `losses.py`'s docstring acknowledges the split deliberately, so it is
> documented rather than accidental — but one loss's spec defaults are spread across two files,
> and this task is rewriting the call site anyway.
>
> **C10-6 — `collab_splats/splats/__init__.py`'s `__all__` is still
> `["GSPLAT_COMMIT", "SplatsConfig", "train"]`.** `gaussian.py` is deliberately not exported
> (Task 7 modified nothing). Decide here whether the model classes join the public surface or
> stay internal, and make `__init__.py` say so — do not leave it undecided.

> **SECOND CORRECTION PASS — measured after Task 8 landed (`a7358c4c`, `8c4f837c`).**
>
> **C10-7 — CONFIRMED against the tree at `c90eea22`. Do NOT re-add a standalone `accumulate` call
> to the rewritten loop.** Task 8 moved
> `AnchorStrategy.accumulate(...)` inside `Scaffold.post_backward`, where it reads its three extra
> arguments (`decode_index`, `decoded_opacities`, `visible_ids`) back out of the `info` dict that
> `render()` returns. The trainer's existing standalone call is still present and
> still correct for *today's* loop, which calls `decode` / `render_gaussians` directly rather than
> `model.render`. **This task deletes that call as it switches to `model.render` + `model.post_backward`.**
> Keeping both double-counts anchor statistics, and no test would fail.
>
> **Line-number correction: the call is at `trainer.py:578`, not the `:584` this block first
> recorded.** Measured at `c90eea22`:
> `strategy.accumulate(strategy_state, info, decode_index, decoded["opacities"], decoded["visible_ids"])`.
> `:585`/`:586` are `anchor_field.mlp_optimizer.step()` / `.zero_grad(set_to_none=True)` — **those two
> must survive** (see the Task 8 status block; they are the only thing training the MLP heads).
> Deleting by line number rather than by content would therefore have deleted exactly the wrong pair.
>
> **C10-8. Deleting `trainer.make_strategy` also deletes the only assertions on three splatfacto
> parity constants — Task 7's fix round moved them, so verify the move landed before you delete.**
> `tests/splats/test_trainer.py:16` imports `make_strategy` at module level, so removing the function
> forces `test_make_strategy_2dgs_splatfacto_args` out, and `test_trainer.py:92-94` go with it:
> `prune_opa == 0.1`, `prune_scale3d == 0.5`, `refine_scale2d_stop_iter == 4000`. This task's step
> text schedules no replacement. Task 7's code-quality review caught it and its fix round adds the
> equivalent assertions to `tests/splats/test_gaussian.py`, which is the right home now that
> `gaussian.py` owns `make_strategy`. **Confirm those assertions exist in `test_gaussian.py` before
> deleting the trainer copy** — mutation-measured, all three literals currently survive a change with
> the suite green, and the next gate after this task is a PSNR parity run, where drift reads as an
> unexplained dPSNR rather than a red test.
>
> **C10-9. `Scaffold.render` is re-entrant and must stay that way.** It carries per-view state in the
> returned `info` dict and writes nothing to `self` outside construction; `8c4f837c` pins this with
> `test_scaffold_render_carries_per_view_state_in_info_not_on_self`, which asserts `set(vars(field))`
> is unchanged across a main render plus a discarded neighbor render. When wiring the loop, keep the
> main view's `info` and hand *that* to `post_backward` — never the neighbor's.
>
> **C10-10 (BLOCKER, measured at `1bdadecd`). This task's Step 4 imports `render_neighbor` from
> `pgsr.py`, and that function does not exist yet — Task 14 creates it. As written, Step 4 turns the
> whole `tests/splats` suite into a collection error.**
>
> Step 4's import block contains
> `from collab_splats.splats.pgsr import render_neighbor, select_near_views`, and Step 3's loop body
> calls `render_neighbor(...)`. Measured: `pgsr.py` at `1bdadecd` defines `pixel_rays`,
> `plane_depth`, `image_gradient_weight`, `erode`, `flat_region_weight`, `patch_offsets`,
> `patch_warp`, `lncc`, `unproject`, `project`, `sample_at_pixels`, `to_gray`, `select_near_views`,
> `pixel_grid`, `_normalize_pixels`, `forward_backward_noise`, `patch_ncc` — **no `render_neighbor`**.
> Step 4's own note ("keep the line and let the PGSR branch be exercised only by Task 14's tests") is
> wrong: an unresolvable module-level import is not a dormant branch, it is an `ImportError` at import
> time. This is the same defect class as the Task 9 structural defect resolved above.
>
> Today the trainer inlines the neighbor render, inside the very
> `if anchor_field is not None: ... else: ...` branch this task exists to delete
> (`trainer.py:520-545` at `c90eea22`) — so the inline code cannot simply be kept either.
>
> **Resolution: this task also adds `render_neighbor` to `pgsr.py`**, using Task 14's function body
> verbatim (including `import numpy as np`). Add `collab_splats/splats/pgsr.py` to this task's Files
> list. Task 14 then only tidies `pgsr.py` and adds the tests, which is what its title already says.
> There is no writer conflict: Tasks 10 through 16 run serially.
>
> **Do not thread `step` into it** — see the C14-1 verdict under Task 14. It was measured and refuted.

> **C10-11 (measured at `da141f16`, from the Task 7 code-quality re-review). One assertion in
> `test_trainer.py` has no home once `trainer.make_strategy` is deleted, and it is not the one the
> plan worried about.** This task deletes `trainer.make_strategy`, and with it the whole of
> `tests/splats/test_trainer.py::test_make_strategy_2dgs_splatfacto_args`. The reviewer traced each
> assertion in that test to see where it lands:
>
> | assertion (`test_trainer.py:92-99`) | lands where |
> |---|---|
> | `prune_opa == 0.1` | covered — `test_gaussian.py:199` (C10-8) |
> | `prune_scale3d == 0.5` | covered — same |
> | `refine_scale2d_stop_iter == 4000` | covered — same |
> | `pause_refine_after_reset == 400` | covered — `test_gaussian.py:161-169` |
> | `absgrad is False` | covered — same |
> | `key_for_gradient == "gradient_2dgs"` | covered — same |
> | `grow_grad2d == approx(2e-4)` | **NOT covered** |
>
> All six covered rows were confirmed by mutation (each mutant dies) after physically deleting the
> three C10-8 assertions from `test_trainer.py`, so the coverage is real and not incidental.
> `grow_grad2d` is different in kind: it is an assertion about the **`SplatsConfig` default**, which
> this task does not delete, not about `make_strategy`. **Move it to a config test rather than
> dropping it with the rest of the block.** `2e-4` is load-bearing — the shipping value was 8e-4
> (4× gsplat's default), which choked 2dgs densification to PSNR 17.95, and it was corrected to 2e-4
> in `dc7d8666`.
>
> **C10-12 (same source, STRENGTHENED by Task 7's third fix round). `_capture` in
> `test_gaussian.py` pins `render_gaussians`' call shape, and it now pins the two arguments that
> matter most to this task.**
>
> The helper around `test_gaussian.py:254` declares all eight of `render_gaussians`' arguments
> positionally. This task folds `outputs.py` into `rendering.py`; if that changes the signature or
> the call shape, this test breaks with a `TypeError` rather than a meaningful failure. It fails
> loudly rather than silently, so it needs no defensive rewrite — but whoever changes the signature
> owns updating it, and a `TypeError` in `test_gaussian.py` during this task is that, not a defect.
>
> **What the third fix round added, and why you must not drop it.** The Task 7 re-review found two
> mutants that survived the entire 235-test suite, both in `Gaussians.render`:
>
> | mutant | effect if shipped |
> |---|---|
> | `render` passes `dict(self.params)` instead of `self.activate()` | the rasterizer gets log-scales, logit-opacities and un-concatenated SH bands — **every render in the project is corrupt**, suite green |
> | `render` passes the literal `"3dgs"` instead of `self.primitive` | **every 2dgs run silently rasterizes as 3dgs** |
>
> Both survived because the round-2 helper at `test_gaussian.py:290` was `_capture(*args, **kwargs)`
> doing `captured.update(kwargs)` — it discarded all eight positionals, and six of the eight were
> unpinned across both helpers. The fix captures `primitive`, `decoded`, `sh_degree` and `absgrad`
> and asserts `captured["primitive"]` on both the 3dgs and the 2dgs model plus
> `torch.allclose(captured["decoded"]["opacities"], torch.sigmoid(model.params["opacities"]))`.
>
> **This task rewrites the call site those assertions guard.** On a 3dgs/2dgs refactor, the
> `primitive` passthrough is the single highest-value argument in the call. If folding `outputs.py`
> in changes the signature, port these assertions to the new shape — do not delete them to make a
> `TypeError` go away.
>
> **One more thing measured there, so nobody chases it as a coverage hole:** dropping the
> `isinstance` guard in `Gaussians.pre_backward` is an **equivalent mutant**.
> `MCMCStrategy.step_pre_backward` resolves to `Strategy.step_pre_backward`, whose body is `pass`,
> so the guard is documentation, not correctness — verified empirically:
> `MCMCStrategy.step_pre_backward is Strategy.step_pre_backward` is `True`. The comment claiming
> "MCMC has no pre-backward hook" was factually wrong and is corrected in the same round.
>
> **Correction to this correction:** an earlier version of this block put that wrong comment at
> `gaussian.py:320`. It was never there — `gaussian.py:320` is the closing paren of `post_backward`'s
> else branch, and `gaussian.py` contains no such text at all. The wrong comment lived at
> **`tests/splats/test_gaussian.py:320`**. The fix round did both halves: it rewrote the false
> test-file comment, and added one explanatory line to `gaussian.py` directly above the
> `pre_backward` guard. Recorded because the misattribution was committed to this plan once already,
> and because it is the same class of staleness as C7-1's and C7-4's wrong line numbers —
> **re-grep, never trust a line number in this document.**

> **C10-13 (from the Task 8 code-quality review, measured at `8c4f837c`). Three more trainer sites
> this task owns, and one claim from that review that must NOT be acted on.**
>
> | Site | What this task does with it |
> |---|---|
> | `trainer.py:571` — inline `info[strategy.key_for_gradient].retain_grad()` | This *is* `pre_backward`'s body. It goes when the call is replaced by `model.pre_backward(info)`; do not leave both. |
> | `trainer.py:475` and `:520` — inline `anchor_field.decode(...)` + `render_gaussians` | Replace with `model.render(...)`. The reviewer verified the swap is **bitwise-equivalent**: `Scaffold.render` reproduces the inline path exactly and adds exactly the three `info` keys (`decode_index`, `decoded_opacities`, `visible_ids`), nothing else. |
> | The neighbor render path | Mutant M20 — rasterizing the neighbor's decode with the **main** view's `camera_id` — survives every test today and becomes *reachable* the moment `render` is wired into the neighbor path. Pin the neighbor camera index when you wire it. |
>
> The same review's "items belonging to later tasks" also says `render_neighbor` must thread `step`
> through rather than defaulting it to `None`. **That is C14-1, and it was measured and REFUTED**
> (see under Task 14): `plane_depth`, `plane_normal`, `plane_distance`, `depth` and `alpha` are
> bit-identical across `sh_degree` 0/2/3 while `rgb` moves (the positive control), and a
> `plane_depth`-only loss gives exactly zero `sh0`/`shN` gradient. `step=None` is correct. Do not
> reverse a measured verdict on a review's say-so.
>
> **C10-14 (same source). `Scaffold` re-entrancy is verified, not assumed — but the guard test was
> weak, and this task is the one that starts depending on it.** The Task 8 reviewer confirmed by
> measurement that a second render from a shifted pose leaves the main view's `info["decode_index"]`,
> `info["decoded_opacities"]`, `info["visible_ids"]` and `render["opacities"]` / `["log_scales"]`
> intact with `vars(model)` unchanged, and that `post_backward` with vs without an interleaved
> neighbor render leaves `denom`, `opacity_accum` and `anchor_denom` bit-identical (`grad_accum`
> differs by 1.907e-06, but a no-neighbor vs no-neighbor control at the same seed differs by
> 3.815e-06 — the signal is below gsplat's own atomicAdd noise floor). **The invariant holds.** What
> did not hold was the test: at its fixture both views saw all 163 anchors, so its two `torch.equal`
> assertions compared values that would match even if `render` were fully stateful. That is being
> fixed in Task 8's own fix round. If this task's interleaved main/neighbor rendering ever produces
> a mismatch, suspect a newly-introduced cache on `self`, not the invariant.
>
> **C10-15 (from the Task 7 code-quality re-review round 4, measured on a pinned tree at
> `8a4fc746`). Four `render_gaussians` arguments are still unpinned, and this task is the one that
> makes them load-bearing.** Task 7's fix round 3 raised the pinned positionals from 2 of 8 to 4 of
> 8 (`primitive`, `decoded`, `sh_degree`, `absgrad`). The reviewer then measured the remaining four
> as live mutants surviving the entire 247-test suite:
>
> | mutant | result | why it matters here |
> |---|---|---|
> | `width` / `height` **transposed** | SURVIVED | Every non-square view renders wrong, suite green. The realistic one. |
> | `cam_to_world` / `intrinsics` **swapped** | SURVIVED | — |
> | `decoded["means"]` zeroed | SURVIVED | Unpinned **anywhere** in the package. |
> | `decoded["quats"]` zeroed | SURVIVED | Unpinned **anywhere** in the package. |
>
> `test_gaussians_activate_raw_parameters_for_the_rasterizer` asserts the *shapes* of `scales`,
> `opacities` and `colors` but never that `means` and `quats` are handed through untouched. It also
> asserts `opacities` against a fixture whose `params["opacities"]` is a **constant vector**
> (`logit(0.1)` for every primitive, 1 unique value), so it pins the value 0.1, not `sigmoid` as a
> function: `torch.full_like(params["opacities"], 0.1)` substituted for the activation SURVIVES the
> full suite. (`scales`/`exp` is genuinely pinned — knn init gives 32 unique values over
> `[-0.94, 0.0039]`.) This is the same fixture-degeneracy class as C12-8's `offsets` and `scaling`
> traps: **perturb `params["opacities"].data` before asserting.**
>
> This task replaces `trainer.py:475`/`:520`'s inline `render_gaussians` call with `model.render(...)`
> per C10-13, which removes the last independent reader of those arguments. Close the four gaps
> before the swap, or a transposed `width`/`height` ships with nothing between it and a training run
> — the same shape of risk N2 flagged for the §9 defaults. The reviewer also noted the lower-
> duplication form for the render test, which kills the same mutants without copying the activation
> math out of the activate test:
>
> ```python
> expected = model.activate()
> assert set(captured["decoded"]) == set(expected)
> assert all(torch.equal(captured["decoded"][k], expected[k]) for k in expected)
> ```

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_trainer.py`:

```python
def test_trainer_has_no_representation_branches():
    """
    The point of the refactor: the trainer must not ask which model it is training.
    """
    source = Path(trainer.__file__).read_text()
    body = source.split("def train(", 1)[1]

    assert "anchor_field" not in body
    assert 'representation == "scaffold"' not in body
    assert "AnchorStrategy" not in body


def test_train_builds_a_gaussians_model_by_default(tmp_path):
    scene = make_scene(n_views=3, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "cap_max": 500, "losses": {}})

    train(cfg, scene["images"], scene["world_to_cam"], scene["intrinsics"], scene["points"], scene["colors"], tmp_path)

    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert set(ckpt["splats"]) == {"means", "scales", "quats", "opacities", "sh0", "shN"}


def test_train_builds_a_scaffold_model_when_asked(tmp_path):
    scene = make_scene(n_views=3, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict(
        {"representation": "scaffold", "scaffold": {}, "max_steps": 3, "losses": {}}
    )

    train(cfg, scene["images"], scene["world_to_cam"], scene["intrinsics"], scene["points"], scene["colors"], tmp_path)

    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert "anchors" in ckpt["splats"]
    assert "mlps" in ckpt
    assert "voxel_size" in ckpt


def test_train_refuses_too_few_seed_points(tmp_path):
    scene = make_scene(n_views=3, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "losses": {}})

    with pytest.raises(ValueError, match="need >= 100 seed points"):
        train(
            cfg,
            scene["images"],
            scene["world_to_cam"],
            scene["intrinsics"],
            scene["points"][:50],
            scene["colors"][:50],
            tmp_path,
        )


def test_train_refuses_mismatched_per_view_arrays(tmp_path):
    scene = make_scene(n_views=3, height=32, width=32, n_points=200)
    cfg = SplatsConfig.from_dict({"max_steps": 3, "losses": {}})

    with pytest.raises(ValueError, match="frames mismatch"):
        train(
            cfg,
            scene["images"],
            scene["world_to_cam"][:2],
            scene["intrinsics"],
            scene["points"],
            scene["colors"],
            tmp_path,
        )
```

Extend the imports:

```python
from pathlib import Path

import pytest
import torch

from collab_splats.splats import trainer
from collab_splats.splats.trainer import SplatsConfig, train
from tests.splats.synthetic import make_scene
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_trainer.py -q
```

Expected: `test_trainer_has_no_representation_branches` fails on `assert "anchor_field" not in body`.

- [ ] **Step 3: Write the new `train()`**

Replace the whole function. This is the complete body:

```python
def train(
    cfg: SplatsConfig,
    images: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsics: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
    depth_targets: np.ndarray | None = None,
    *,
    min_points: int = 100,
    lr_decay: float = 0.01,
) -> None:
    """
    Train one splat model and write splats.ply, ckpt.pt and splats_quality_report.json to out_dir.

    Args:
        cfg: the run's SplatsConfig; `representation` picks the model class and `primitive`
            picks the rasterizer.
        images: (n_views, H, W, 3) uint8 frames.
        world_to_cam: (n_views, 4, 4) COLMAP-convention poses.
        intrinsics: (n_views, 3, 3) camera matrices at frame resolution.
        points: (P, 3) float32 seed points in the same world frame.
        colors: (P, 3) uint8 seed colors.
        out_dir: output directory; created if absent.
        depth_targets: optional (n_views, h, w) float32 depth at any resolution, 0 = no target.
        min_points: fewest seed points that can produce a model.
        lr_decay: total multiplicative decay of the position lr over the run.

    Returns:
        None — everything is written to `out_dir`.
    """
    device = "cuda"
    n_views, height, width = images.shape[:3]
    out_dir = Path(out_dir)

    # Refuse inputs that cannot train: too few seed points, or mismatched per-view arrays
    if len(points) < min_points:
        raise ValueError(f"splats: need >= {min_points} seed points, got {len(points)}")
    per_view_counts = {n_views, len(world_to_cam), len(intrinsics)}
    if depth_targets is not None:
        per_view_counts.add(len(depth_targets))
    if len(per_view_counts) != 1:
        raise ValueError(
            f"splats: frames mismatch — images {n_views}, world_to_cam {len(world_to_cam)}, "
            f"intrinsics {len(intrinsics)}, depth_targets "
            f"{None if depth_targets is None else len(depth_targets)}"
        )

    # Cameras on the GPU; frames stay uint8 on the CPU and move one view at a time.
    # normalize_scene trains in splatfacto's unit-cube frame (scene_scale fixed to 1.0);
    # otherwise the world frame is kept and scene_scale carries the extent.
    cam_to_world = torch.from_numpy(np.linalg.inv(world_to_cam)).float().to(device)
    intrinsics = torch.from_numpy(np.asarray(intrinsics)).float().to(device)
    world_extent = compute_scene_scale(cam_to_world)
    center, normalize_factor = None, None
    if cfg.normalize_scene:
        cam_to_world, points, center, normalize_factor = scene_normalization(cam_to_world, points)
        scene_scale = 1.0
    else:
        scene_scale = world_extent

    # The model owns its parameters, optimizers, schedulers and densification strategy;
    # pose and appearance refinement own theirs. Neither construction branches on
    # representation past this line.
    model_class = Scaffold if cfg.representation == "scaffold" else Gaussians
    model = model_class(cfg, points, colors, scene_scale, n_views, device, lr_decay=lr_decay)
    refine = CameraOpt.from_config(
        cfg, n_views, world_extent, scene_scale, lr_decay ** (1.0 / cfg.max_steps), device
    )
    logger.info(
        "splats: training %s/%s, %d views, %d primitives at start",
        cfg.primitive,
        cfg.representation,
        n_views,
        model.n_primitives,
    )

    # PGSR's losses are defined only against the 3dgs kernel (GS-SR pairs scaffold-pgsr with
    # the vanilla rasterizer; there is no 2dgs-pgsr upstream). Its neighbor views are scored
    # once over the seed cloud, not per step.
    pgsr_mv_spec = cfg.losses.get("pgsr_multiview")
    if (cfg.losses.get("pgsr_normal") is not None or pgsr_mv_spec is not None) and cfg.primitive != "3dgs":
        raise ValueError(f"pgsr losses need primitive: 3dgs, got {cfg.primitive!r}")
    near_ids: list[list[int]] = []
    if pgsr_mv_spec is not None:
        near_ids = select_near_views(
            torch.linalg.inv(cam_to_world),
            intrinsics,
            torch.from_numpy(np.ascontiguousarray(points)).float().to(device),
            height,
            width,
            num_views=int(pgsr_mv_spec.get("num_multi_view", 5)),
            max_points=int(pgsr_mv_spec.get("max_points", 20000)),
        )

    # Shuffle-and-pop view order, coarse-to-fine resolution, and the depth targets the
    # optional depth loss reads. `view_order` is an infinite generator: it reshuffles per epoch.
    views = view_order(n_views)
    depth_tensor = None if depth_targets is None else torch.from_numpy(depth_targets).float()
    background = torch.zeros(3, device=device)
    started = time.time()
    final_losses = {}

    for step in progress(range(cfg.max_steps), desc="splats train"):
        view = next(views)
        factor = downscale_factor(cfg, step)
        view_image, view_intrinsics, view_height, view_width = downscale_view(
            images[view], intrinsics[view], factor
        )
        target = prepare_target(view_image, device)

        # Pose-corrected camera, then the render, then the appearance affine over a random
        # background — every representation goes through the same three calls
        camera_id = torch.tensor([view], device=device)
        view_cam_to_world = refine.camera(cam_to_world[view : view + 1], camera_id)
        render, info = model.render(
            view_cam_to_world,
            view_intrinsics[None],
            view_width,
            view_height,
            camera_id,
            step=step,
            render_normals=loss_active(cfg.losses, "normal_consistency", step),
            render_plane=loss_active(cfg.losses, "pgsr_normal", step)
            or loss_active(cfg.losses, "pgsr_multiview", step),
        )
        render["rgb"] = refine.color(render["rgb"], camera_id)
        background = torch.rand(3, device=device)
        render["rgb"] = render["rgb"] + background * (1.0 - render["alpha"])

        # PGSR compares against one co-visible neighbor, rendered through the same pose
        # correction at this step's resolution. Upstream does NOT detach it: the geometric term
        # pulls both views' plane depths together (GS-SR gssr/scene/pgsr_scene.py).
        if loss_active(cfg.losses, "pgsr_multiview", step) and near_ids[view]:
            near = near_ids[view][random.randrange(len(near_ids[view]))]
            near_image, near_intrinsics, _, _ = downscale_view(images[near], intrinsics[near], factor)
            near_camera_id = torch.tensor([near], device=device)
            render["world_to_cam"] = torch.linalg.inv(view_cam_to_world)
            render["intrinsics"] = view_intrinsics[None]
            render["pgsr_neighbor"] = render_neighbor(
                model,
                near_image,
                refine.camera(cam_to_world[near : near + 1], near_camera_id),
                near_intrinsics[None],
                near_camera_id,
            )

        model.pre_backward(step, info)
        loss, losses = compute_losses(cfg, render, target, step, depth_tensor, view)
        loss.backward()

        # Upstream's order: parameters, then schedulers, then densification. Reversing the last
        # two silently skips refinement (gsplat @ d2f5c0f, examples/simple_trainer.py).
        for optimizer in model.optimizers + refine.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in model.schedulers + refine.schedulers:
            scheduler.step()
        model.post_backward(step, info)

        final_losses = {name: float(value) for name, value in losses.items()}
        if step % cfg.log_every == 0:
            logger.info(
                "splats: step %d/%d loss %.4f, %d %s",
                step,
                cfg.max_steps,
                float(loss),
                model.n_primitives,
                type(model).__name__,
            )

    # Put the model and the cameras back in the world frame before anything is written
    seconds = time.time() - started
    if cfg.normalize_scene:
        denormalize_cameras(cam_to_world, center, normalize_factor)
        model.denormalize(center, normalize_factor)
        refine.denormalize(normalize_factor)

    # The stored poses are the ones that were rendered: pose deltas fold in here and are
    # not saved separately
    with torch.no_grad():
        corrected = torch.cat(
            [
                refine.camera(cam_to_world[view : view + 1], torch.tensor([view], device=device))
                for view in range(n_views)
            ]
        )
    write_outputs(
        cfg, model, refine, images, list(range(n_views)), corrected, intrinsics, out_dir, seconds, final_losses
    )
```

- [ ] **Step 4: Fix the imports**

`trainer.py`'s import block becomes:

```python
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from collab_splats.splats.cameras import CameraOpt
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.losses import (
    LOSS_SPEC_KEYS,
    OPTIONAL_LOSSES,
    compute_losses,
    default_losses,
    loss_active,
    validate_schedule,
)
from collab_splats.splats.pgsr import render_neighbor, select_near_views
from collab_splats.splats.rendering import write_outputs
from collab_splats.splats.scaffold import Scaffold
from collab_splats.splats.utils import (
    compute_scene_scale,
    denormalize_cameras,
    downscale_factor,
    downscale_view,
    prepare_target,
    scene_normalization,
    view_order,
)

logger = logging.getLogger(__name__)
```

**Import direction:** `trainer.py` sits at the top — it imports everything and nothing imports it, except tests. `render_neighbor` is added to `pgsr.py` in Task 14; until then, keep the line and let the PGSR branch be exercised only by Task 14's tests.

- [ ] **Step 5: Delete the helpers `train()` no longer needs**

From `trainer.py`, delete:
- `init_gaussians_from_points` — Task 7's `Gaussians.__init__`.
- `make_strategy` — moved to `gaussian.py` in Task 7.
- `make_pose_refiner`, `make_appearance_module` — Task 5's `CameraOpt.from_config`.
- `ViewSampler` — Task 4's `view_order`.
- `denormalize_outputs` — split across `utils.denormalize_cameras`, `model.denormalize` and `refine.denormalize`.

Confirm nothing else calls them:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "init_gaussians_from_points\|make_pose_refiner\|make_appearance_module\|ViewSampler\|denormalize_outputs" \
    collab_splats tests evals docs || echo "NONE LEFT"
```

Expected: `NONE LEFT`.

- [ ] **Step 6: Update the module docstring**

```python
"""
Gaussian-splat training on upstream gsplat.

Two independent axes: `primitive` picks the rasterizer (3dgs or 2dgs), `representation` picks
the model (vanilla Gaussians or Scaffold-GS anchors). Both models answer the same interface,
so `train` never asks which it has.

In: frames, COLMAP poses, intrinsics and a seed point cloud. Out: splats.ply, ckpt.pt and
splats_quality_report.json.
"""
```

- [ ] **Step 7: Run the tests to verify they pass**

The whole splats suite is meaningful now — this is the first point at which the refactored trainer runs end to end.

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats -q > /tmp/claude-0/-workspace-collab-splats/2c4f47d0-1708-41c5-89da-14aa041ea396/scratchpad/task10.txt 2>&1
echo "PYTEST_RC=$?"; tail -20 /tmp/claude-0/-workspace-collab-splats/2c4f47d0-1708-41c5-89da-14aa041ea396/scratchpad/task10.txt
```

Expected: `PYTEST_RC=0` and every test in `tests/splats` passing, including the six from Task 9 that could not run before this task existed.

- [ ] **Step 8: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/trainer.py tests/splats/test_trainer.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/trainer.py tests/splats/test_trainer.py && \
  git commit --only collab_splats/splats/trainer.py tests/splats/test_trainer.py \
    -m "refactor(splats): rewrite train() with no representation branches"
```

---

## Task 11: Parity check after the interface phase

Tasks 3-10 moved every line of the training path. This is where you find out whether the numbers moved with them. Do not start Task 12 until this passes.

**Files:**
- Modify: `/tmp/claude-0/-workspace-collab-splats/scratchpad/parity_after_phase3.json` (scratchpad, not committed)

- [x] **Step 1: Re-run the three configs**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -u /tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py after_phase3 \
  > /tmp/claude-0/-workspace-collab-splats/scratchpad/after_phase3.log 2>&1
echo "RC=$?"; tail -5 /tmp/claude-0/-workspace-collab-splats/scratchpad/after_phase3.log
```

Expected: `RC=0`, and the log's last lines list three fingerprints. Runtime is roughly 6 minutes on the A40 (three 300-step runs). If it exceeds the Bash timeout, re-run with `run_in_background: true` rather than a longer timeout.

- [x] **Step 2: Compare against the baseline**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py --compare before_v3 after_phase3
```

Expected, for each of `3dgs_vanilla`, `2dgs_vanilla`, `3dgs_scaffold`:

```
PASS 3dgs_vanilla: dPSNR=0.00e+00 means_ok=True n_ok=True
PASS 2dgs_vanilla: dPSNR=0.00e+00 means_ok=True n_ok=True
PASS 3dgs_scaffold: dPSNR=0.00e+00 means_ok=True n_ok=True
PARITY OK
```

The gate is `dPSNR <= 1e-3` dB and `np.allclose(means_stats, rtol=1e-4)`, not exact equality — cuda reductions are not associative and the loop's tensor shapes changed.

- [x] **Step 3: If it fails, bisect before you patch**  *(not needed — Step 2 passed on the first run.)*

A failure here means one of Tasks 3-10 changed behavior. Work backwards, cheapest first:

1. **Only scaffold fails** → Task 8. The likely cause is the `LambdaLR` conversion. Print the lr in force at steps 0, 1, 150 and 299 for `offsets` and each MLP group, and compare against `expon_lr` at the same steps (recover it from `git show HEAD~N:collab_splats/splats/scaffold.py`).
2. **Only 2dgs fails** → Task 7's `make_strategy`. `DefaultStrategy` is sensitive to `packed`, `absgrad` and `key_for_gradient`; diff the constructed strategy's `__dict__` against the old one.
3. **All three fail by the same sign** → Task 3 or Task 4. Either the view order changed (compare the first 12 indices against the literals in Task 4) or `prepare_target` changed (compare a target tensor's mean and dtype).
4. **All three fail with `n_gaussians` differing** → densification ran a different number of times. Check that `post_backward` is called after the schedulers, not before.

Fix the cause, re-run Step 1, and only then continue.

- [x] **Step 4: Record the numbers and commit the plan checkbox**

Append the three fingerprints to the plan's parity table (Task 2 created it), then:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  git add -f docs/superpowers/plans/2026-09-06-splats-cleanup.md && \
  git commit --only docs/superpowers/plans/2026-09-06-splats-cleanup.md \
    -m "docs(plans): splats cleanup — phase 3 parity confirmed"
```

---

> **DONE 2026-09-06 — 7/7 PASS at `348d879b`, carried forward to `1ae66a7f`.** The harness runs
> seven configs, not the three this task was written against (`3dgs_vanilla`, `2dgs_vanilla`,
> `3dgs_scaffold`, `3dgs_pgsr`, `3dgs_appearance`, `3dgs_norm_nopose`, `scaffold_norm_nopose`),
> baseline `before_v3`, `STEPS = 50`. Worst deviation **dPSNR 1.18e-07** (`3dgs_scaffold`); three
> configs at exactly 0.00e+00; `means_ok` and `n_ok` true everywhere; `PARITY_RC=0`.
>
> **Carried forward, not re-run, through fix rounds 4 and 5.** Both were test-only. The carry-forward
> is licensed by one command, which must print nothing:
> `git diff --name-only 348d879b HEAD -- collab_splats/ tests/splats/synthetic.py`
> (`synthetic.py` is the only member of `tests/` the harness imports). Verified empty at `1ae66a7f`.
> **This expires the moment Task 12 or Task 14 lands** — both touch `collab_splats/`. After that the
> honest statement is "parity re-established by Task 13", never "carried forward".
>
> **What this gate cannot see.** Both scaffold configs leave `pose_opt` off (`3dgs_scaffold` takes
> the config default, `scaffold_norm_nopose` sets it `False`), so `corrected == cam_to_world` in
> every config and the harness is structurally blind to pose-dependent `splats.ply` defects — which
> is exactly why it passed at `aaadb0e1` while the ply was wrong. `PARITY OK` means "nothing else
> moved", never "the ply is right". The ply is pinned by
> `test_rendering.py::test_the_scaffold_ply_bakes_against_the_uncorrected_training_poses` and
> `..._holds_the_values_exported_at_the_frame_size`, not here.

## Task 12: Simplify `scaffold.py`

With the interface in place, the scaffold module's remaining complexity is its own: an abstract base class with one implementation, a `verbose` flag, a helper nobody calls, a constant that is always the same number, and a strategy that carries its state in a dict it hands back and forth.

**Files:**
- Modify: `collab_splats/splats/scaffold.py`
- Modify: `tests/splats/test_scaffold.py`

> **POST-IMPLEMENTATION AMENDMENTS — measured at `398293f2`, after Task 12 landed as `378a9e91`.
> Two of these say the plan's own steps were wrong; one says a correction in this very block is
> still owed and its line numbers have drifted.**
>
> **A1. `_frustum_anchors` is LIVE. Step 1's `assert "_frustum_anchors" not in source` must be
> deleted, and Step 7's "three dead pieces" is two.** Measured: the call is at `scaffold.py:408`,
> inside `visible_anchors`' non-CUDA branch, and the definition at `:422`. Every CPU test in
> `tests/splats` goes through it. The two genuinely dead pieces Step 7 names are `verbose` and
> `VIEW_DIM`. An implementer following Step 1 literally would have written a test that fails on
> correct code, then deleted a live method to make it pass.
>
> **A2. Step 4 says "allocate the three tensors"; `AnchorStrategy.__init__` allocates four.**
> Measured by AST at `398293f2`: `offset_gradient_accum`, `offset_denom`, `opacity_accum`,
> `anchor_denom`. (Its other `self` assignments are `cfg`, `primitive`, `voxel_size`, `device`,
> `key_for_gradient`.) Confirmed alongside: the class's base list is now empty — it no longer
> subclasses gsplat's `Strategy` — `prune` carries the keyword-only `scale_cap`, `accumulate` takes
> `(step, info)`, `refine` takes `(scaffold, step)`, and `strategy_state` appears **zero** times in
> `scaffold.py`.
>
> **A3. Step 4 omits the `AnchorStrategy(...)` call-site rewrites the new signature forces.**
> Measured at `398293f2`: three in `tests/splats/test_scaffold.py`. (The implementer estimated ~10
> before writing them; three is what shipped. Take the shipped number, not the estimate.)
>
> **A4. `SCALE_CAP` was imported by the tests and Step 7 never says so.** It is now `prune`'s
> keyword-only `scale_cap=0.05`, and `tests/splats/test_scaffold.py:1377` asserts
> `"SCALE_CAP" not in source`. Whoever wrote Step 7 checked production callers and not test ones.
>
> **A5 (reported by the implementer, NOT re-measured here — both targets are now deleted, so it is
> no longer checkable).** Step 7's line numbers were stale by roughly four lines each: `VIEW_DIM`
> defined `:131` used `:156` against the plan's `:127`/`:152`, and `SCALE_CAP` defined `:136` used
> `:1046` against `:132`/`:722`. Recorded for the pattern, not as a live instruction.
>
> **A6 (reported by the implementer, found by its own probe).** The plan's test set left `refine`'s
> window bounds unguarded: mutating `step <= self.cfg.update_from` to `step <` **survives** unless a
> test asserts `step == update_from` **and** `step == update_until` explicitly. Both are multiples of
> `refine_every` in the fixture, so both are reachable. This is the same defect class the branch has
> now closed seven times — a boundary guarded by one interior case is one case over an aggregate of
> two endpoints.
>
> **A7 (reported by the implementer).** `test_anchor_growth_still_adds_anchors` as written in the
> plan **grows nothing** — measured 163 anchors before and 163 after. `offsets` initializes to zero,
> so every growth candidate lands in an already-occupied voxel, and upstream thins candidates at
> random on top of that. It needs `monkeypatch.setattr(torch, "rand_like", torch.ones_like)` plus a
> displaced offset (`model.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])`), and its
> `assert model.n_primitives >= before` must be `>`. A `>=` assertion on a growth test passes when
> nothing grows: the same shape as every other vacuous guard on this branch.
>
> **A9 — C12-1 IS NOW CLOSED, landed as `025ca70f` ("refactor(splats): retire the dead absgrad
> expression in gaussian.py", 2 files, 41 insertions / 11 deletions). Three things came back with it,
> two of which are defects in A8 and in this block, not in the code. All independently re-verified by
> the controller before being written here.**
>
> **A9-1. The trap A8's dispatch brief warned about does not exist, and its word "silently" was
> wrong.** The brief claimed that hardcoding `absgrad=False` at the `render` call would silently
> diverge from a future `make_strategy(absgrad=True)`. Measured in gsplat **1.5.3**: the backward
> writes `.absgrad` **onto `means2d` only** — `cuda/_wrapper.py:2118-2119` and `:3123-3124`, both
> under `if absgrad:` — while `DefaultStrategy` reads `info[self.key_for_gradient].absgrad`
> (`strategy/default.py:244-245`), and for 2dgs `key_for_gradient` is `"gradient_2dgs"`, which
> `rendering.py:2515` aliases to the **`densify`** tensor, not `means2d`. So `absgrad=True` raises
> `AttributeError` at the first refine step **whether or not `render` forwards it** — the implementer
> measured the identical four-failure set and the identical `default.py:245` AttributeError against
> the *pre-change* derived expression. The failure is loud, and this commit does not change it.
>
> **A9-2 (the real defect). C12-1's remedy deletes an assertion a LATER commit deliberately added,
> and A8 re-derived C12-1's line numbers without noticing.** C12-1 was measured at `235c954e`
> (07:50). At `31a2495b` (09:15) Task 7's second fix round added
> `default.strategy.absgrad = True; assert captured["absgrad"] is True` to
> `test_render_warms_up_the_sh_degree_and_forwards_the_strategys_absgrad`, precisely to prove
> `absgrad` is *read from* the strategy rather than forwarded as a constant — a property this plan
> records two tasks earlier, at line 4593. Retiring the derived expression makes that assertion
> unwritable. A8 checked four line numbers and re-confirmed the remedy "otherwise unchanged"; it did
> not check whether anything had come to depend on the property being retired. **Re-measuring a
> correction's coordinates is not the same as re-measuring its premise.**
>
> The implementer's resolution, which the controller endorses: replace the flip-assertion with the
> coupling `assert captured["absgrad"] is default.strategy.absgrad` alongside
> `assert default.strategy.absgrad is False`, and rename the host test to
> `test_render_warms_up_the_sh_degree` since its old name became false. The retired half is carried
> by a new guard, `test_render_never_asks_for_absolute_gradients`, whose sole-killer status was
> measured: deleting it lets the `render` literal drift `False`->`True` under **346 passed, RC=0**.
>
> **A9-3. Line 4569 of this plan still names the deleted test.** The C10-3 row reads
> "`test_render_warms_up_the_sh_degree_and_forwards_the_strategys_absgrad` covers the SH warm-up and
> `absgrad`". That name no longer exists in `tests/splats/test_gaussian.py`. Read it as
> `test_render_warms_up_the_sh_degree` (SH warm-up) plus
> `test_render_never_asks_for_absolute_gradients` (absgrad); C10-3 stays closed.
>
> **Numbers, at `025ca70f`:** `tests/splats` **347 / 0 / 0, RC=0** (346 before, +1 the new guard);
> scoped wide gate **439 / 0 / 0, RC=0** (438 before). Mutation kills, pinned tree, blobs md5-verified,
> restored between probes: the `render` literal flipped `False`->`True` gives **1 failed, zero
> collateral**, the intended test named. `make_strategy`'s 2dgs `absgrad=False`->`True` gives 4 failed
> — 1 intended, 3 pre-existing and reproduced identically against the pre-change code, so not
> attributable to this commit.
>
> **A8. C12-1 IS STILL OPEN and its line numbers have drifted.** It is the one item in this block no
> commit has closed — Task 12's implementer correctly refused it as a third file outside its
> two-file ownership. Re-measured at `398293f2`: the dead expression is
> `absgrad = isinstance(self.strategy, DefaultStrategy) and self.strategy.absgrad` at
> **`gaussian.py:268`** (C12-1 says `:249`), consumed at `:277`; `make_strategy` hardcodes
> `absgrad=False` at **`:83`** (C12-1 says `:76`); the comment explaining why lives at
> **`:63-66`** (C12-1 says `:56-59`). The remedy C12-1 states is otherwise unchanged. **This needs
> an owner** — it is a one-file change and nothing else in the plan touches `gaussian.py`.

> **CORRECTION PASS C12-1..C12-6 — from Task 7's code-quality review, measured at `235c954e`.**
> This task is a simplification pass, and a simplification pass is exactly the thing that deletes
> the five items below. Line numbers are stale — re-grep before acting.
>
> **C12-1. One thing in `gaussian.py` genuinely IS dead and should go.** `gaussian.py:249`,
> `absgrad = isinstance(self.strategy, DefaultStrategy) and self.strategy.absgrad`, is `False` on
> every reachable path — measured `3dgs -> False`, `2dgs -> False`, `from_checkpoint -> False`.
> `make_strategy` is the sole producer and hardcodes `absgrad=False` at `:76`. It is a faithful port
> of an equally dead expression at `trainer.py:500`, and CLAUDE.md forbids dead branches kept for
> hypothetical future use. Replace with a literal `absgrad=False` plus a pointer to the `:56-59`
> comment that explains why.
>
> **The remaining four look deletable and are not. Do not simplify them away.**
>
> **C12-2. The `pause_refine_after_reset` cap at `gaussian.py:63-73` is a bug fix over splatfacto,
> not a port.** Upstream's bare `n_views + 100` silently disables densification forever once
> `n_views >= reset_every - 100`. The cap plus its `logger.warning` is the only mutant-killing
> literal assertion in the file, and `test_gaussian.py:163-170` asserts the log text too.
>
> **C12-3. `export_gaussians` returns the live parameters, not copies**, and `test_gaussian.py:126`
> pins identity with `is`. A well-meaning defensive `.clone()` breaks the ply writer's view of
> post-`denormalize` values — and `splats.ply` bytes are frozen by Ground Rule §7.
>
> **C12-4. The four "unused" parameters on `export_gaussians`, and `camera_id` on `render`, are
> deliberate** — each is annotated as `Scaffold`-required at the site. A dead-parameter sweep would
> delete precisely the interface this plan exists to build.
>
> **C12-5. `gaussian.py:24 SH_C0` and `rendering.py:25 SH_DC_NORMALIZER` are bit-identical** —
> verified `3fd20dd750429b6d` both ways (big-endian IEEE-754, `struct.pack('>d', ...)`),
> **0 ULP**. Existing `splats.ply` and checkpoint `sh0` values
> are safe under Task 9's rename. Do not "reconcile" one form to the other on the assumption they
> might differ; changing either form changes frozen bytes.
>
> **C12-6. The Google `Args:` / `Returns:` style in the rewritten files is convergence, not
> drift.** Measured: `utils.py` 7/6, `cameras.py` 4/4, `losses.py` 2/2 use it, while the
> not-yet-rewritten `rendering.py`, `pgsr.py`, `scaffold.py` and `outputs.py` still use `- ` bullets.
> Do not "restore consistency" by reverting the rewritten files to bullets.

> **C12-7 (BLOCKING for this task's Step 1. First measured at `08050ada`, re-verified at
> `5bf13f1f`). `_scaffold_model()` NO LONGER EXISTS.** Task 8's fix round deleted it.
> `grep -rn _scaffold_model tests/ collab_splats/` returns **nothing**. This task's Step 1 code
> block still calls it four times and Task 8's own section still shows its definition — both are
> stale text, not instructions. **Do not trust the line numbers an earlier version of this block
> quoted; they have already shifted once. `grep -n '_scaffold_model' <this file>` instead.** Same
> staleness class as C7-1, C7-4 and the C10-12 misattribution.
>
> `tests/splats/test_scaffold.py` has **one** builder, measured at `5bf13f1f`:
>
> ```python
> def _scaffold_config(run=None, **scaffold): ...
> def _field(n_offsets=4, appearance_dim=0, n_views=3, device="cpu",
>            scene_scale=1.0, run=None, **scaffold): ...
> ```
>
> `**scaffold` goes into the `"scaffold"` block and `run=` into the top level, and **both levels
> validate**: a typo in `**scaffold` raises `ValueError: splats.scaffold: unknown keys [...]`, a
> typo in `run=` raises `ValueError: splats: unknown keys [...]`. So `_field(lr_max_steps=200)`
> configures what its name says. The old `_scaffold_model(**overrides)` was deleted because its
> `**overrides` splatted into the **top level** of `SplatsConfig.from_dict` rather than into the
> `"scaffold"` block — `_scaffold_model(n_offsets=8)` looked like it configured the scaffold and
> silently did not. Two competing builders in one file were also the direct cause of the degenerate
> re-entrancy fixture (see C12-8). **Use `_field`; do not resurrect the deleted helper.**
>
> The other helpers in that file at `5bf13f1f`, so you do not rebuild one that exists:
> `_seed_points`, `_cam`, `_cam_pair`, `_strategy_and_state`, `_window`, `_backward_info`,
> `_reachable_state`. Local variables are named `model`, never `field` — the whole `field` token
> was purged in the fix round (`grep '\bfield\b'` over both `scaffold.py` and `test_scaffold.py`
> returns zero hits). The helper `_field` itself keeps its name; renaming it is a **Task 16**
> question, not yours.
>
> **C12-8 (three fixture facts measured at `08050ada` — this task rewrites the code they guard, so a
> test you break here will not tell you it broke).**
>
> **The shape of all three: `Scaffold`'s parameters initialize to values that make different code
> paths indistinguishable.** A test that builds a fresh model and asserts on its output cannot
> separate them. Two of the three had to be found twice — once by a reviewer, once by a re-reviewer
> — because arming one of them does not arm the others.
>
> **1. `scaling` is `full((n, 6), log_voxel)` at init**, so the offset extent `[:, 0:3]` and the
> gaussian extent `[:, 3:6]` are **bit-identical** until something perturbs them. A test that bakes
> an export straight from a fresh model cannot distinguish the two slices at all — swapping them is
> unobservable, and `export_gaussians` reading the wrong one would ship gaussians at the wrong size
> with a green suite. `test_scaffold.py` now perturbs `model.params["scaling"][:, 3:] += 0.75`
> specifically to arm that assertion. **If you touch the scaling layout, keep the perturbation and
> keep it on the same model instance the assertion reads.**
>
> **2. The re-entrancy fixture is degenerate at almost every camera offset.** At `_field(n_offsets=2)`
> on CUDA the main view sees **163/163** anchors, and `dx=1.5` (the previously shipped value),
> `dz=2.0` and `dz=2.5` *all* also see 163 with `visible_ids` equal to the main view's — so the two
> `torch.equal` assertions compared values that would match even if `render` were fully stateful.
> `dx=3.0` gives **103/163** and is what the fixed test uses. The test now asserts non-degeneracy
> **before** the equality assertions, which is the load-bearing ordering: it makes a future
> degeneracy fail loudly instead of going quiet. **Preserve that ordering.**
>
> **3. `offsets` initialize to EXACTLY ZERO** (`scaffold.py:320`,
> `torch.zeros(n_anchors, n_offsets, 3)`). In
>
> ```python
> means = (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep]
> ```
>
> `means == anchors` **whatever multiplies `offsets`**. Any assertion on decoded or baked `means`
> from a fresh model is vacuous with respect to the offset-extent factor. Three mutants survived the
> whole suite on this: taking `scaling[:, 3:6]` instead of `[:, :3]` in `export_gaussians`' `means`
> (the exact mirror of the M11 slice swap, on the line above it), **the same slice swap in `decode`'s
> own copy of that arithmetic**, and dropping the offset term from the export entirely. The test file
> now perturbs `model.params["offsets"] += 0.3` alongside the `scaling` perturbation.
>
> **The `decode` one is a pre-existing production gap, and this task rewrites `decode`.** `means` is
> on the frozen `splats.ply` surface (Ground Rules §7): a wrong slice ships gaussians at the wrong
> size with a green suite. **Keep both perturbations, on the same model instance the assertion
> reads.**
>
> **A related limit worth knowing before you restructure — CORRECTED at `5bf13f1f`.**
> `test_scaffold.py`'s `_reachable_state` helper snapshots values reachable from the model, and it
> is what proves `render` is re-entrant. It **now reaches `self.strategy`**: the fix round made it
> walk `vars(model.strategy)` under a `strategy.` prefix, and mutant `N11`
> (`self.strategy.last_decode_index = decode_index` inside `render`) dies through that walk —
> it survived at `08050ada` and is KILLED at `5bf13f1f`.
>
> **The reason this block previously gave for the blind spot was WRONG, and the correction matters
> because the same false premise would mislead you again.** It said `AnchorStrategy` defines no
> `__repr__` so the default `<... object at 0x...>` made a cache there invisible. Measured:
> gsplat's `Strategy` base **is a dataclass**, so `AnchorStrategy` inherits a generated `__repr__`
> and `repr(model.strategy)` is `AnchorStrategy()` — address-free. The old helper missed strategy
> state because it did not *walk* the strategy at all, not because the repr was opaque. The
> conclusion (the snapshot is stable across calls) was right; the mechanism was not. Same staleness
> class as C7-1, C7-4 and the C10-12 misattribution: **re-measure, do not inherit a stated cause.**
>
> **The walk's measured ceiling at `5bf13f1f`** — 44 keys, stable across two calls on an untouched
> model. It CATCHES: a new attribute on `model`, a new attribute on `strategy`, a new key in
> `strategy_state`, an in-place `params` mutation, and changes to `strategy.cfg`'s declared fields,
> `strategy.primitive`, `strategy.key_for_gradient`. It does NOT catch (each measured with a
> planted mutant that survived): a dynamically-added attribute on `strategy.cfg` (one level deeper
> than the walk), an attribute on `model.mlps`, or a module-level dict in `scaffold.py`. If you move
> state into any of those three, no test in the file will tell you.
>
> **One line of this is still owed and it is yours:** `_reachable_state`'s own docstring in
> `tests/splats/test_scaffold.py` repeats the false `<... object at 0x...>` mechanism. Correct it to
> say the walk descends into `vars(model.strategy)`; do not leave a docstring asserting something
> that was measured false.
>
> **C12-9 (BLOCKING for how you verify this task, measured at `5bf13f1f`). The only guard on
> `decode`'s means arithmetic is an export↔decode PARITY assertion, and parity is structurally
> blind to the mutation you are most likely to make here.**
>
> This task rewrites `decode`. The re-reviewer established exactly where its guard lives, by
> co-mutation:
>
> | mutant | result |
> |---|---|
> | wrong scaling slice in `decode` ALONE (`N13`) | **KILLED** |
> | wrong scaling slice in `export_gaussians` ALONE (`N6`) | **KILLED** |
> | the same wrong slice in **BOTH** (`N13both`) | **SURVIVES**, 68/68 green |
> | offset term dropped in both (`N14both`) | **SURVIVES** |
> | offset sign negated in both (`J6signBoth`) | **SURVIVES** |
>
> Only three value assertions in the file touch decoded `means`, and two of them cannot see a slice
> swap: `displacement <= offset_extent * 1.001 + 1e-6` is a **bound** (and was vacuous at
> `offsets == 0`), and `world["means"] == trained["means"] / scale` is decode-vs-decode. The one
> that fires is `baked["means"] == decoded["means"]` — pure parity.
>
> **So: rewrite `decode` and the export path consistently-wrong and this suite stays green, on the
> frozen `splats.ply` surface.** A second parity test would not help; the fix is an **independent
> absolute-value guard on `decode`** — assert the decoded means against values computed in the
> test, not against the export. Add one before you touch the arithmetic. Do not treat "68 passed"
> as evidence that `decode` still decodes.
>
> **C12-10 (measured at `5bf13f1f`, with a verified one-line fix). Offset SLOT identity is
> unpinned — `+= 0.3` armed the magnitude, not the slot.**
>
> C12-8's fix adds a **uniform** `model.params["offsets"] += 0.3`, so with `n_offsets=2` slot 0 and
> slot 1 hold identical values and `offsets.roll(1, dims=1)` is a no-op on them. Measured surviving
> at **both** `08050ada` and `5bf13f1f` (pre-existing, not introduced by the fix round):
> `J6slotRollExport`, `J6slotRollDecode`, `J6slotRollBoth`.
>
> Fix, measured in a pinned probe tree — baseline 68 passed, then `J6slotRollExport` **KILLED** and
> `J6slotRollDecode` **KILLED**:
>
> ```python
> model.params["offsets"] += (
>     torch.arange(1, model.cfg.n_offsets + 1).float().view(1, -1, 1) * 0.3
> )
> ```
>
> `J6slotRollBoth` and `J6signBoth` still survive it — that is C12-9's parity blind spot, which no
> fixture can close. **Take the ramp when you rewrite the fixture; it costs one line and it is the
> same defect class as the two `+= ` perturbations already there.**
>
> Already pinned and passing, so do not re-derive them: `exp()` on the offset extent, the
> `reshape(-1, 3)` ordering, and the `[keep]` visibility mask each die in export, in decode, and in
> both.
>
> ---
>
> **C12-11 (RETRACTION of C12-9's premise, measured at `378a9e91`). C12-9 opens "This task rewrites
> `decode`, so that blindness is directly in its path." That premise is FALSE.** `decode` and
> `export_gaussians` are **byte-identical across `378a9e91`** — the commit does not touch either
> function. What the task actually rewrites is `accumulate`, which only *reads*
> `info["decode_index"]` and `info["decoded_opacities"]`; it never recomputes the means.
>
> The parity blind spot C12-9 describes is real and the co-mutation measurements behind it stand.
> But it is a **pre-existing** hole in the decode/export guards, not a hole this task opened. The
> consequence for review: `test_decode_means_are_the_anchor_plus_its_offset_extent`, added by this
> task's implementer, is a guard on code **this task did not modify**. That is a defensible thing to
> ship — the hole was measured and it is now closed — but it must be scored as *extra*, not as
> *required by the task*, and a spec reviewer must not treat its absence in an earlier round as
> non-compliance.
>
> **C12-12 (scope decision, deliberate, measured at `378a9e91`). `_frustum_anchors` STAYS.** The
> design spec wishes it deleted (`docs/superpowers/specs/2026-09-05-splats-cleanup-design.md:149`
> and `:316`) and Step 1's code block still carries
> `assert "_frustum_anchors" not in source`. **That assertion must be deleted, not satisfied.** The
> function is live: it is called at `scaffold.py:408`, inside `visible_anchors`' non-CUDA branch,
> and defined at `:422`. Step 7's own removal gate returns hits for it. Deleting it would make the
> non-CUDA path unreachable and force most of `tests/splats/test_scaffold.py` to become CUDA-only,
> which is a strictly worse test surface than the dead-code saving is worth. Step 7's "three dead
> pieces" is therefore **two** — `verbose` and `VIEW_DIM`.

> **SPEC RE-REVIEW OF THE FIX ROUND — verdict ISSUES FOUND (1 blocking, 3 observations), measured
> at `7feed636`. The headline is good: all four of the fix round's findings are genuinely closed
> and the round stayed in scope. It shipped one new hole alongside them.**
>
> **C12-13. The fix round's production diff is exactly one deleted line, re-derived three ways.**
> `7feed636` removes `self.primitive = primitive` from `AnchorStrategy.__init__` and changes nothing
> else under `collab_splats/`. The reviewer confirmed the attribute is genuinely unreachable by AST
> walk, by grep across the whole repo, and by a **dynamic `__getattr__` tripwire** installed on the
> class for the duration of the 448-case wide gate: it fired exactly **once**, from the new guard
> test's own `hasattr`. That is the strongest form of "nothing reads it" available short of running
> production traffic, and it is worth reusing — a static absence proof cannot see an attribute
> reached through `getattr(obj, name)` with a computed name.
>
> **C12-14. The gate at each tree in the chain, all pinned, all RC=0, skips 0.**
>
> | tree | `tests/splats` | scoped wide gate |
> |---|---|---|
> | `225abd6f` (pre-Task-12 baseline) | 346 | — |
> | `025ca70f` (Task 12's actual parent) | 347 | 439 |
> | `7feed636` (fix round 1) | 356 | 448 |
>
> **The fix round's commit body advertises "346 → 356".** That cites `225abd6f`, which is not this
> commit's parent. Against the real parent `025ca70f` the delta is **+9, not +10**. The tests are all
> there; only the arithmetic in the message is off by one. Recorded so that a later reader
> reconciling counts does not go looking for a tenth test that was never written.
>
> **C12-15 (BLOCKING). The new CUDA accumulator test fails when run alone.**
>
> ```
> pytest tests/splats/test_scaffold.py::test_post_backward_accumulates_a_cuda_render_inside_the_window
> → RC=1,  tests/splats/test_scaffold.py:1160
>   assert float(strategy.offset_denom.sum()) > 0.0
>   AssertionError: assert 0.0 > 0.0
> ```
>
> `_field` builds `ScaffoldMLPs` from the **unseeded global torch RNG**. Whether any decoded Gaussian
> survives projection with `radii > 0` — and therefore whether `accumulate`'s gradient half reaches
> `index_add_` at all — is a function of that RNG state. Across 12 explicit seeds:
> `offset_denom.sum() > 0` holds in **7 / 12** (fails at seeds 2, 3, 5, 8, 9); `anchor_denom.sum() > 0`
> holds 12/12, always 163. Only the `offset_denom` half is fragile.
>
> Fair attribution: the file already had unseeded-RNG order coupling — over 10 shuffled collection
> orderings, `025ca70f` fails **3/10** (all `test_export_gaussians_writes_the_decoded_values_in_the_
> ply_s_raw_forms`) and `7feed636` fails **1/10** (the new test). So the fix round added an instance
> of a pre-existing class rather than inventing the class. But its instance is **the only one of the
> file's 14 accumulate/`post_backward`/`@cuda` tests that trips `pytest <nodeid>`** — 13 pass alone,
> this one does not — and the node-id run is the probe a developer reaches for when debugging.
>
> **Severity is false RED, never false green.** Under the four-accumulators-on-CPU mutant, run alone
> in the flaking RNG state, the mutant still dies with
> `RuntimeError: Expected all tensors to be on the same device ... cpu and cuda:0`, because
> `opacity_accum` / `anchor_denom` take an unfiltered index and always raise.
>
> **C12-16. `-p no:randomly` is a NO-OP on this branch — `pytest-randomly` is not installed.**
> Verified by the controller with `importlib.util.find_spec`. Every brief on this branch writes the
> flag and none of them has ever suppressed anything: default runs are file-order. **A green suite
> here is not evidence of order-independence**, which is exactly how C12-15 hid. Order-dependence
> must be probed deliberately — isolate-run each test, or shuffle the collection explicitly.
>
> **C12-17 (unseeded torch RNG is branch-wide, not local to this file).** Two independent agents
> found the same class in two places: `ScaffoldMLPs` via `_field` here, and `Gaussians.__init__` in
> `test_pgsr.py`, where it made a "max abs diff 1.446" threshold non-reproducible — three runs of the
> identical fixture gave 13.441 / 5.110 / 1.730. **Do not write a value threshold against an unseeded
> fixture anywhere in `tests/splats`.**
>
> **C12-18. The `*args` behavioral test's stated necessity is refuted; keep the test, fix the words.**
> The behavior verified as claimed: `def prune(self, scaffold, *args, scale_cap=0.05)` reports
> `KEYWORD_ONLY`, and `prune(model, 0.5)` returns with `scale_cap=0.05` and `args=(0.5,)`, raising
> nothing. **But the pre-existing `test_anchor_strategy_calls_do_not_take_a_state_argument` already
> kills that mutant** — measured at `225abd6f`, **1 failed / 345**, because `args` enters
> `list(signature(strategy.prune).parameters)`. `*args` was never an open escape for the suite. The
> docstring is also wrong that "the arity in `match` pins that": `pytest.raises(TypeError)` alone
> excludes the `*args` form, since nothing raises at all; the `match=` string pins **CPython's exact
> wording**, which is version-coupled brittleness rather than the discriminator claimed. The test is
> a legitimate second axis and stays — only its justification needs correcting.
>
> **C12-19 (BLOCKING FOR TASK 15). `test_the_retired_constants_and_the_verbosity_flag_are_absent_by_
> name` fires on a pure COMMENT.** Planting
> `# VIEW_DIM was retired; its literal 3 now lives on the head that reads it` — no binding, just
> prose — gives **1 failed / 355**. This is the two-sided shape of instance 9: a substring guard
> misses every constant it was not told the name of, **and** fires on prose containing no constant.
>
> Its retention is nonetheless load-bearing: `VIEW_DIM = 3` planted **inside**
> `AnchorStrategy.__init__` gives 1 failed / 355 killed by this test alone, invisible to the AST
> binding guard. So the mechanism must be fixed, not the test deleted — a **token** check over
> `tokenize.NAME` tokens skips comments and string literals by construction while keeping the
> in-function-body reach an AST *binding* walk cannot express.
>
> This must close **before Task 15 dispatches**, because Task 12's own commit message says VIEW_DIM's
> "explanation move[s] to the one head that reads it" and Task 15 is a docstring-and-comment sweep
> over this exact file. The natural way to write that explanation breaks the test.
>
> Related, so the retention argument is not double-counted: `verbose` planted as a `prune`
> **parameter** fails both this test and the names test (2 failed / 354); `verbose` as a `Scaffold`
> **attribute** is still caught only here.
>
> **C12-20. `_module_level_bindings`' comment overclaims.** All four parametrized spellings are
> load-bearing, and the four misses (`(X := v)` walrus, module-level `for X in ...` target,
> `with ... as X`, `import m as X`) are not realistic ways to reintroduce a tunable — the reviewer
> does not call this a hole in the guard's stated property. But the helper's comment says *"An `if` /
> `try` / `with` at module level still runs at import, so its bodies count too"*, and it walks the
> `with` **body**, not its `as` target. A comment that reads as coverage is this defect class's
> permanent tell; correct the comment whether or not the walk widens.

> **CORRECTION PASS C12-21..C12-25 — from Task 12's fix round 2 (`3ecd780a`), every item measured by
> the implementer in a pinned tree and reported back against the controller's brief. Three of these
> falsify claims the controller wrote; one is a live defect that outlived the round.**
>
> **C12-21 (BLOCKING, live defect, now Task 12 fix round 3).
> `test_export_gaussians_writes_the_decoded_values_in_the_ply_s_raw_forms` is FLAKY, and it is not
> order-dependent.** Measured at `3ecd780a`: **3 of 20** isolated node-id runs fail (and 2 of 10 in a
> separate batch) on
> `assert torch.allclose(torch.sigmoid(baked["opacities"]), decoded["opacities"], atol=2e-4)`, with
> `sigmoid(-9.21) = 1e-4` against `decoded = -0.0632`. Mechanism, read off `scaffold.py`:
> `mlp_opacity` ends in `Tanh` (`:156`), so neural opacity is signed, and both `decode` (`:497-501`)
> and `export_gaussians` (`:716-721`) carry the same all-closed fallback that force-keeps
> `neural_opacity.argmax()` **even when it is ≤ 0**. From there the paths legitimately diverge:
> `decode:513` returns the opacity **raw** (negative), `export_gaussians:735` **clamps** it into a
> valid probability first — `torch.logit(opacities.clamp(1e-4, 1 - 1e-4))`. A ply cannot carry a
> negative opacity, so **both sides are correct and the test is wrong** to assert exact parity across
> that branch. **Seeding alone would hide it**, which is why fix round 3 is briefed to diagnose before
> seeding.
>
> **C12-22 (BLOCKING for every test on the shared scaffold fixture).** When that fallback fires,
> `keep` selects exactly **one** slot, so `decode` returns **one** gaussian. Every test built on
> `_field(n_offsets=2)` that asserts a property *across* decoded rows is then asserting it over a
> single row — near-vacuous, and green. This is the vacuous-test class again, in the form where the
> aggregate silently collapses to N=1. The remedy is the non-vacuity guard the file already uses at
> `:1216` (`assert bool(model.visible_anchors(...).all())`).
>
> **C12-23. "Fails alone" was reported as a state and is actually a coin flip.** The controller's
> brief said the CUDA accumulator test fails under `pytest <nodeid>`. The implementer's first
> isolated run **passed**; over 10 runs it failed 2. `torch` seeds its default generator from entropy
> per process, so an isolated run samples a random seed. The direction and the seed table (7/12 for
> `offset_denom`, zero at 2/3/5/8/9; `anchor_denom` 163 at 12/12) are exactly right — the framing was
> not. **Never describe an unseeded test's outcome as a deterministic state**, and a single isolate
> run cannot establish "passes alone".
>
> **C12-24. "`verbose` as a `Scaffold` attribute is caught only by the by-name test" is FALSE.**
> Measured 2 failed / 365: `test_scaffold_checkpoint_round_trips` also catches it, because
> `from_checkpoint` bypasses `__init__` and the test asserts `set(vars(restored)) == set(vars(model))`.
> The retention argument for the by-name guard survives on the `VIEW_DIM`-inside-`__init__` case,
> which that test does hold alone (1 failed / 366).
>
> **C12-25. A trap in probing a token-based guard.** "Prove it does not fire on a comment" **cannot**
> be probed by making `_identifiers` scan `tokenize.COMMENT`: a COMMENT token's string is the whole
> comment line, and set membership is exact, so that mutant **survives**. The probe has to split
> comment text into words (or drop to a substring form) to bite. Equally, a docstring and a string
> literal are one token kind, so those two prose cases are **not separable from each other** — the
> test says so in its own comment rather than implying coverage it lacks.
>
> **Recorded as still true after this round, do not re-measure:** the 360 / 452 baselines at
> `2155f673`; `scaffold.py` and `test_scaffold.py` unchanged since `7feed636`; the `*args` mutant
> already dying at the names test; the comment false positive at 1 failed / 359; CUDA present with 0
> skips; `pytest-randomly` absent.

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_scaffold.py`:

```python
def test_anchor_strategy_owns_its_state():
    model = _scaffold_model()
    strategy = model.strategy

    # State is attributes on the strategy, not a dict passed through every call
    assert isinstance(strategy.opacity_accum, torch.Tensor)
    assert isinstance(strategy.offset_gradient_accum, torch.Tensor)
    assert isinstance(strategy.offset_denom, torch.Tensor)
    assert len(strategy.opacity_accum) == model.n_primitives
    assert not hasattr(model, "strategy_state")


def test_anchor_strategy_accumulate_takes_step_and_info():
    model = _scaffold_model()
    strategy = model.strategy
    signature = inspect.signature(strategy.accumulate)

    assert list(signature.parameters) == ["step", "info"]


def test_anchor_strategy_refine_takes_the_model_and_the_step():
    model = _scaffold_model()
    signature = inspect.signature(model.strategy.refine)

    assert list(signature.parameters) == ["scaffold", "step"]


def test_scaffold_module_has_no_abstract_base_and_no_verbose():
    source = Path(scaffold.__file__).read_text()

    assert "class Strategy" not in source
    assert "verbose" not in source
    assert "_frustum_anchors" not in source
    assert "VIEW_DIM" not in source


def test_anchor_growth_still_adds_anchors():
    """
    The simplification must not change what densification does — only where its state lives.
    """
    model = _scaffold_model()
    strategy = model.strategy
    before = model.n_primitives

    # Force every slot over the growth threshold, then refine on a step inside the window.
    # offset_denom must clear refine_every * success_threshold * 0.5 or `grow` reads the slot
    # as unseen and ignores its gradient.
    cfg = strategy.cfg
    step = (cfg.update_from // cfg.refine_every + 1) * cfg.refine_every
    strategy.offset_gradient_accum += 1.0
    strategy.offset_denom += float(cfg.refine_every)
    strategy.refine(model, step)

    assert model.n_primitives >= before
```

Extend the imports:

```python
import inspect
from pathlib import Path

from collab_splats.splats import scaffold
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -q -k "strategy or module_has_no"
```

Expected: failures on `strategy.opacity_accum` (AttributeError) and on the `class Strategy` assertion.

- [ ] **Step 3: Delete the abstract base class**

`AnchorStrategy` inherits `Strategy` from gsplat purely for the `check_sanity` hook, which this codebase never calls. Change the declaration and drop the `super().__init__()`:

```python
class AnchorStrategy:
    """
    Scaffold-GS anchor densification: grow into under-covered voxels, prune anchors that shut.

    - Accumulates a per-offset screen-space gradient and a per-anchor opacity, both averaged
      over the steps where the offset was actually rendered.
    - Growth adds one anchor per occupied voxel of the growth candidates that no anchor covers.
    - Pruning drops anchors whose mean decoded opacity stayed under the threshold.

    Ported from Scaffold-GS @ main, scene/gaussian_model.py (`training_statis`, `adjust_anchor`).
    """
```

Delete the `from gsplat.strategy import Strategy` import.

- [ ] **Step 4: Move the strategy state onto the strategy**

Delete `initialize_state`. In `AnchorStrategy.__init__`, allocate the three tensors directly:

```python
    def __init__(self, cfg, primitive: str, voxel_size: float, n_anchors: int, device: str):
        """
        Args:
            cfg: the run's ScaffoldConfig; thresholds and cadence are read here and kept.
            primitive: "3dgs" or "2dgs"; picks which gsplat info key carries the gradient.
            voxel_size: the anchor grid pitch, in world units. Growth quantises to it.
            n_anchors: anchors at initialization; sizes the accumulators.
            device: torch device string.
        """
        self.cfg = cfg
        self.voxel_size = voxel_size
        self.device = device
        self.key_for_gradient = "gradient_2dgs" if primitive == "2dgs" else "means2d"

        # One slot per offset for the gradient, one per anchor for opacity. Both are running
        # sums with their own denominator: an offset that never renders must not read as zero.
        n_offsets = n_anchors * cfg.n_offsets
        self.offset_gradient_accum = torch.zeros(n_offsets, device=device)
        self.offset_denom = torch.zeros(n_offsets, device=device)
        self.opacity_accum = torch.zeros(n_anchors, device=device)
        self.anchor_denom = torch.zeros(n_anchors, device=device)
```

The four dict keys map onto the four attributes one for one — `state["grad_accum"]` →
`self.offset_gradient_accum`, `state["denom"]` → `self.offset_denom`, `state["opacity_accum"]` →
`self.opacity_accum`, `state["anchor_denom"]` → `self.anchor_denom`. The two renames are
deliberate: `grad_accum`/`denom` do not say what they are per (a slot, not an anchor), and the
class now has both kinds of denominator side by side. Grow and prune reassign the attributes
where they reassigned dict entries (`self.offset_denom = torch.cat([...])`), so the resize
still happens in the same block that resizes `params["offsets"]`.

`self.device` is new and load-bearing: the accumulators used to inherit their device from
whatever `initialize_state` allocated, and every `.to(device)` in `accumulate` read it back off
`state["grad_accum"]`. They are now allocated on `device` directly.

In `Scaffold.__init__` (Task 8, Step 8), the construction collapses to one line and `strategy_state` disappears:

```python
        # Anchor densification: grows into under-covered cells and prunes anchors whose offsets shut
        self.strategy = AnchorStrategy(
            self.cfg, self.primitive, self.voxel_size, len(self.params["anchors"]), device
        )
```

- [ ] **Step 5: Collapse the two post-backward entry points**

`should_accumulate` + `accumulate(state, info, decode_index, decoded_opacities, visible_ids)` + `step_post_backward(field, state, step)` become two methods that read what they need from `info`:

```python
    def accumulate(self, step: int, info: dict) -> None:
        """
        Fold this step's screen-space gradients and decoded opacities into the running sums.

        - The window guard was `should_accumulate`: both bounds exclusive, counting starts
          before growing does so the first refine reads a full window (GS-SR densify(),
          gssr/gaussian/scaffold_gaussian.py:710).
        - Gradients are renormalized to [-1, 1] screen space exactly as gsplat's
          DefaultStrategy does (strategy/default.py:243-249), which is what makes Scaffold's
          published grad_threshold directly usable here.
        - Only Gaussians the projection kept (radii > 0) carry gradient evidence; the opacity
          half is unfiltered, because an anchor is visited whether or not its Gaussians landed.

        Args:
            step: current training step; accumulation runs only inside the statistics window.
            info: the render's gsplat info dict, carrying `decode_index`, `decoded_opacities`,
                `visible_ids` and the retained gradient under `key_for_gradient`.

        Returns:
            None.
        """
        if not self.cfg.start_stat < step < self.cfg.update_until:
            return

        # A step whose gradient never reached the tensor (nothing rendered) is skipped, not counted
        grads = info[self.key_for_gradient].grad
        if grads is None:
            return
        grads = grads.detach().clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        grad_norm = grads.reshape(-1, 2).norm(dim=-1)

        index = info["decode_index"].to(self.device)

        # radii is [C, N, 2] (or [N, 2] for one camera); a Gaussian counts if any axis rendered
        rendered = info["radii"].reshape(len(grad_norm), -1).amax(dim=-1) > 0
        grad_index = index[rendered.to(self.device)]
        self.offset_gradient_accum.index_add_(0, grad_index, grad_norm[rendered].to(self.device))
        self.offset_denom.index_add_(
            0, grad_index, torch.ones_like(grad_index, dtype=self.offset_denom.dtype)
        )

        # Negative opacities were dropped at decode and upstream clamps them to zero before
        # summing, so summing the surviving slots gives the same per-anchor numerator
        anchor_index = torch.div(index, self.cfg.n_offsets, rounding_mode="floor")
        self.opacity_accum.index_add_(0, anchor_index, info["decoded_opacities"].detach().to(self.device))
        visible = info["visible_ids"].to(self.device)
        self.anchor_denom.index_add_(0, visible, torch.ones_like(visible, dtype=self.anchor_denom.dtype))

    def refine(self, scaffold, step: int) -> None:
        """
        Grow then prune anchors on the refine cadence.

        - Both window bounds are exclusive upstream: densify() refines on
          `densify_from_iter < step < densify_until_iter` and frees its accumulators at the
          upper bound (GS-SR gssr/gaussian/scaffold_gaussian.py:707-717).
        - Each half resets only the slots / anchors whose statistics it consumed, so a rarely
          visible one keeps building history instead of being wiped every window.

        Args:
            scaffold: the Scaffold whose params and param_optimizers are grown and pruned.
            step: current training step; refinement runs every `refine_every` steps inside
                the window.

        Returns:
            None.
        """
        if step <= self.cfg.update_from or step >= self.cfg.update_until:
            return
        if step % self.cfg.refine_every != 0:
            return

        self.grow(scaffold)
        self.prune(scaffold)
```

`grow` and `prune` lose their `state` parameter the same way — `grow(self, scaffold)` and
`prune(self, scaffold)`, reading `self.offset_gradient_accum` where they read `state["grad_accum"]`.
Their bodies are otherwise untouched; this task moves state, it does not change densification.

`Scaffold.post_backward` becomes:

```python
    def post_backward(self, step: int, info: dict) -> None:
        """
        Accumulate anchor statistics, then grow and prune.

        Reading the retained gradient here rather than before the optimizer step is safe:
        `zero_grad(set_to_none=True)` clears the *parameters'* gradients, and the tensor read
        here is a retained non-leaf whose `.grad` no optimizer touches.

        Args:
            step: current training step.
            info: the gsplat info dict this step's render returned.

        Returns:
            None.
        """
        self.strategy.accumulate(step, info)
        self.strategy.refine(self, step)
```

- [ ] **Step 6: Turn `SCALE_CAP` into a `prune` keyword**

The last module-level tunable in this file. It is used once, at `scaffold.py:722`, inside
`prune`. Give it to the function that uses it and delete the constant at `scaffold.py:132`:

```python
    def prune(self, scaffold, *, scale_cap: float = 0.05) -> int:
        """
        Drop anchors whose mean decoded opacity stayed below min_opacity across the window.

        - An anchor must have been visited for most of the window before its mean opacity is
          evidence (upstream anchor_demon > check_interval * success_threshold); one that was
          never decoded is kept, since no evidence is not evidence of transparency.
        - Never prunes the field empty: gsplat's projection kernel raises SIGFPE on an empty input.

        Args:
            scaffold: the Scaffold whose params and param_optimizers are pruned.
            scale_cap: upper bound on the raw gaussian-extent channels of `scaling`, applied on
                every refine. Upstream's value; raising it lets offsets grow past the anchor cell.

        Returns:
            The number of anchors removed.
        """
```

and at the clamp:

```python
        with torch.no_grad():
            scaffold.params["scaling"][:, 3:].clamp_(max=scale_cap)
```

`refine` calls it with no keyword, so the default is what runs — this is a signature change,
not a behavior change.

- [ ] **Step 7: Delete the three dead pieces**

- `verbose` — the flag and every `if self.verbose:` block. Logging is `logger.debug`, gated by log level like the rest of the codebase.
- `_frustum_anchors` — confirm it is dead before deleting:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "_frustum_anchors" collab_splats tests evals || echo "DEAD - safe to delete"
```

- `VIEW_DIM` — the constant is always `3` (a unit direction has three components) and it has
  exactly one use, at `scaffold.py:152`. Inline it there, with the reason in the comment, and
  delete the module-level definition at `scaffold.py:127`:

```python
        # Every head sees the anchor feature concatenated with the unit view direction (3 components)
        base_dim = cfg.feat_dim + 3
```

The three `torch.nn.Sequential` heads below it already read `base_dim` and do not change.

- [ ] **Step 8: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats -q
```

Expected: all pass. `tests/splats/test_scaffold.py` is the file that matters, but the whole suite runs because `post_backward`'s signature changed.

- [ ] **Step 9: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/scaffold.py tests/splats/test_scaffold.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/scaffold.py tests/splats/test_scaffold.py && \
  git commit --only collab_splats/splats/scaffold.py tests/splats/test_scaffold.py \
    -m "refactor(splats): simplify AnchorStrategy state and delete scaffold dead code"
```

---

## Task 13: Parity check after the simplification phase

Same drill as Task 11. Task 12 touched densification, which is the part of the system most able to change the numbers without changing the code's apparent meaning.

> **RUN 2026-09-06 — PASS 7/7 at `f7623794`. Covers BOTH Task 12 and Task 14 in one run.**
>
> - Tree: `/workspace/collab-splats/.worktrees/clean-splats`, proof line
>   `collab_splats.__file__ = .worktrees/clean-splats/collab_splats/__init__.py`
> - Production tree at run time = `99f309cb` (Task 14, `pgsr.py`) + `378a9e91` (Task 12,
>   `scaffold.py`). `f7623794` itself is docs-only.
> - Baseline `before_v3`, label `after_task12_14`. Log `scratchpad/parity_after_task12_14.log`,
>   JSON `scratchpad/parity_after_task12_14.json`. `PARITY_RUN_RC=0`, `PARITY_CMP_RC=0`.
>
> ```
> PASS 2dgs_vanilla:        dPSNR=0.00e+00 means_ok=True n_ok=True
> PASS 3dgs_appearance:     dPSNR=5.91e-08 means_ok=True n_ok=True
> PASS 3dgs_norm_nopose:    dPSNR=0.00e+00 means_ok=True n_ok=True
> PASS 3dgs_pgsr:           dPSNR=2.60e-08 means_ok=True n_ok=True
> PASS 3dgs_scaffold:       dPSNR=1.32e-07 means_ok=True n_ok=True
> PASS 3dgs_vanilla:        dPSNR=3.10e-07 means_ok=True n_ok=True
> PASS scaffold_norm_nopose:dPSNR=0.00e+00 means_ok=True n_ok=True
> PARITY OK
> ```
>
> This run was made **speculatively, while Task 12's and Task 14's reviews were still open**, on the
> reasoning that both commits had landed and a re-run costs only wall-clock. That is also its
> validity condition: **any review fix touching `collab_splats/` or `tests/splats/synthetic.py`
> voids it and needs a re-run before Task 13 is checked off.** Task 14's spec review has since
> returned ISSUES FOUND, and its Finding 1 remedy is test-only while Finding 2 adds two tests — so
> if both fixes stay inside `tests/`, this result carries. Verify, do not assume:
>
> ```bash
> git diff --name-only f7623794 HEAD -- collab_splats/ tests/splats/synthetic.py
> ```
>
> must print nothing.
>
> **Still blind to the same thing as every previous run.** Both scaffold configs leave `pose_opt`
> off, so no run of this harness can see a pose-dependent ply defect. `PARITY OK` means "nothing
> else moved", never "the ply is right".

> **RUN 2026-09-06 (v2) — PASS 7/7 at `2d5c01d7`. THIS RUN SUPERSEDES THE ONE ABOVE.**
>
> The `f7623794` block's own validity condition fired. Two commits landed after it that touch
> `collab_splats/`, so its `git diff --name-only f7623794 HEAD -- collab_splats/` no longer prints
> nothing and its numbers are **void**:
>
> - `025ca70f` — `rendering.py`, retires the dead `absgrad` expression (one line, plus a comment).
> - `7feed636` — Task 12's fix round; the production half is the single deleted line
>   `self.primitive = primitive` in `AnchorStrategy.__init__`.
>
> Both are "provably dead" claims, which is exactly the class of claim this harness exists to
> check — a dead expression that was not dead moves the numbers. It did not.
>
> - Tree: `/workspace/collab-splats/.worktrees/clean-splats`, proof line
>   `collab_splats.__file__ = .worktrees/clean-splats/collab_splats/__init__.py`
> - HEAD `2d5c01d7` at start **and** at finish. `collab_splats` tree hash `05d09a4d…` and the
>   `tests/splats/synthetic.py` blob `0f6ca092…` identical at both ends; `git status --porcelain`
>   empty; nothing staged, nothing committed by the run.
> - Baseline `before_v3`, label `after_task12_14_v2`. `PARITY_RUN_RC=0`, `PARITY_CMP_RC=0`.
>
> ```
> PASS 2dgs_vanilla:        dPSNR=0.00e+00 means_ok=True n_ok=True
> PASS 3dgs_appearance:     dPSNR=0.00e+00 means_ok=True n_ok=True
> PASS 3dgs_norm_nopose:    dPSNR=3.39e-08 means_ok=True n_ok=True
> PASS 3dgs_pgsr:           dPSNR=1.55e-07 means_ok=True n_ok=True
> PASS 3dgs_scaffold:       dPSNR=1.05e-07 means_ok=True n_ok=True
> PASS 3dgs_vanilla:        dPSNR=0.00e+00 means_ok=True n_ok=True
> PASS scaffold_norm_nopose:dPSNR=0.00e+00 means_ok=True n_ok=True
> PARITY OK
> ```
>
> **Extra measurement, not asked for and worth keeping.** The agent also ran
> `--compare after_task12_14 after_task12_14_v2`, which cancels the earlier refactor out and
> isolates `025ca70f` + `7feed636` **alone**: RC=0, max drift 3.10e-07. That is inside the harness's
> own ≤4e-07 noise floor at 50 steps and four orders below the 1e-3 gate. Both scaffold configs came
> in at 0.00e+00 / 2.36e-07 with `n_gaussians` exact (163 / 166) — the configs that actually exercise
> the deleted `AnchorStrategy` line.
>
> **Three corrections to the dispatch brief, returned by the agent and verified here.**
>
> 1. **The run takes ~3 minutes, not "tens of minutes."** The brief said the latter. Budget the
>    re-runs accordingly — Task 19's final parity is cheap, and a speculative re-run after any
>    `collab_splats/` change is cheaper than reasoning about whether it was needed.
> 2. **`means_ok` compares only the FIRST FIVE GAUSSIANS.** `splats_parity.py:92` records
>    `"means_head": means[:5].tolist()` and `:143` is
>    `np.allclose(a[name]["means_head"], b[name]["means_head"], rtol=1e-4, atol=1e-6)`. The JSON also
>    records `ssim`, `means_mean` and `means_std` — **`compare` never asserts any of them**
>    (`:145`, `good = d_psnr <= 1e-3 and means_ok and n_ok`). So the per-config gate is exactly three
>    things: `d_psnr <= 1e-3`, five means, and exact `n_gaussians`. A defect confined to gaussians
>    6..N with no PSNR signature is invisible to `PARITY OK`. Note this beside the standing
>    `pose_opt` blindness; it is a second, independent limit on what a PASS means.
> 3. **Do not run Step 4's `git add -f`.** The plan's Step 4 prescribes it, but the plan file is
>    already tracked — `add -f` on an already-tracked path stages it into the **shared** index, which
>    every concurrent session's `git commit` would then sweep up. `git commit --only <path>` is
>    sufficient and is what this task uses. The same stale instruction sits in Task 16 Step 10; T16-1
>    records it there.
>
> **Blindness unchanged.** Both scaffold configs still leave `pose_opt` off. `PARITY OK` means
> "nothing else moved", never "the ply is right".

**Files:**
- Modify: `/tmp/claude-0/-workspace-collab-splats/scratchpad/parity_after_phase4.json` (scratchpad, not committed)

- [ ] **Step 1: Re-run the three configs**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -u /tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py after_phase4 \
  > /tmp/claude-0/-workspace-collab-splats/scratchpad/after_phase4.log 2>&1
echo "RC=$?"; tail -5 /tmp/claude-0/-workspace-collab-splats/scratchpad/after_phase4.log
```

- [ ] **Step 2: Compare against the baseline**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py --compare before_v3 after_phase4
```

Expected: three `PASS` lines then `PARITY OK`, and exit code 0.

- [ ] **Step 3: If scaffold fails, the cause is in Task 12**

The vanilla configs never touch `AnchorStrategy`, so a scaffold-only failure localises immediately:

1. **`n_gaussians` (anchors) differs** → grow or prune ran a different number of times. The window guards moved from `should_accumulate` into `accumulate` and `refine`; check both boundaries are `>=` / `<=` exactly as before (recover the old guards with `git show HEAD~1:collab_splats/splats/scaffold.py`).
2. **PSNR differs but anchor count matches** → the accumulators are being read at a different scale. The dict-to-attribute move must preserve the denominators; print `offset_denom.sum()` at step 150 before and after.
3. **Growth crashes on a shape mismatch** → the accumulator resize is not tracking the grow. Every attribute sized `n_anchors * n_offsets` must be resized in the same block that resizes `params["offsets"]`.

- [ ] **Step 4: Record and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  git add -f docs/superpowers/plans/2026-09-06-splats-cleanup.md && \
  git commit --only docs/superpowers/plans/2026-09-06-splats-cleanup.md \
    -m "docs(plans): splats cleanup — phase 4 parity confirmed"
```

---

## Task 14: Tidy `pgsr.py` and give it `render_neighbor`

> **CORRECTED 2026-09-06 by the Task 14 implementer, five errors, all verified by the controller
> against `9714e202` before being written back. Landed as `99f309cb`. The shipped code deviates
> from Steps 1, 3, 4 and 5 deliberately and correctly — read these before judging spec compliance.**
>
> 1. **Step 3 says "the four module-level constants". There are FIVE:** `THETA0`, `SIGMA_BELOW`,
>    `SIGMA_ABOVE`, `MIN_RAY_COSINE`, `MIN_DEPTH` (`pgsr.py:30-44` at `9714e202`). Verified.
>    *(Controller note: a naive `grep -nE "^[A-Z_]+ *= *"` returns four and hides `THETA0` — the
>    character class excludes the digit. Use `[A-Z_0-9]+`. That grep is how the plan got it wrong.)*
>
> 2. **`MIN_DEPTH` has TWO call sites, not one:** `project` (`:231`) and `forward_backward_noise`'s
>    ray rescale (`:401`). Step 3 relocates it only to `project`, which would leave a bare `1e-6`
>    behind and silently split one knob in two. The shipped fix gives `forward_backward_noise` the
>    same keyword-only `min_depth: float = 1e-6` and threads it through both `project` calls and its
>    own clamp, so one value governs the whole round trip. Verified.
>
> 3. **Step 4's `sample_at_pixels` code block is from a different version of the file** and would
>    break `forward_backward_noise`. It shows `sample_at_pixels(image, pixels)` over `(1,H,W,C)` with
>    `align_corners=False` and `pixels[...,0]/width*2.0 - 1.0`. The real function is
>    `sample_at_pixels(image, pixels, height, width)` over `(1,C,H,W)` with `align_corners=True` and
>    the half-pixel `2*(p-0.5)/(W-1) - 1` convention, which is how `forward_backward_noise` calls it.
>    Verified at `pgsr.py:236,245`. **Do not apply that block.**
>
> 4. **Step 4's "inline `_normalize_pixels`" direction is BACKWARDS — it would ADD duplication.**
>    The helper has two call sites (`patch_ncc` `:455`, `:481`) plus a third private copy inside
>    `sample_at_pixels`; inlining gives three copies where there were two. Verified. The shipped fix
>    dedupes the other way: `sample_at_pixels` normalizes *through* `_normalize_pixels`, which moved
>    up beside it to avoid a cross-section forward reference. **Consequence: Step 1's
>    `assert "_normalize_pixels" not in source` is STRUCK** — it now asserts the opposite of the
>    desired state. It was replaced by a monkeypatch spy asserting the routing. The
>    `pixel_rays` -> `pixel_grid` half of Step 4 is correct and was applied as written.
>
> 5. **Step 5's code block carries a bug the shipped code does not have.** It writes
>    `to_gray(torch.from_numpy(image)...float()[None] / 255.0)`. `to_gray` takes `(H,W,3)` and
>    returns `(1,1,H,W)`, so the extra `[None]` yields `(1,1,1,h,w)`, which `patch_ncc` cannot
>    consume. Shipped `render_neighbor` (`pgsr.py:574`) has no `[None]` and is correct.
>    **Strike the `[None]`.**
>
> ~~**Step 1's two `render_neighbor` tests were NOT added and remain owed as a coverage item.** They
> are unrunnable as written on three counts: `device="cpu"`; `(1, 32, 32, 1)`; the 32x32 square
> fixture...~~
>
> **CORRECTION 6 — 2026-09-06, by Task 14's spec reviewer, verified by the controller. The
> paragraph above was wrong on all three counts and is retracted.** It described the Step 1 block as
> it read *before* `2cbc7520` ("fix degenerate fixtures and the missing CUDA guard in T14/T16/T17",
> 12:40:52), which is an ancestor of `9714e202` (17:53:23) by 5h13m — `git merge-base --is-ancestor
> 2cbc7520 9714e202` exits 0. At `9714e202` the block already reads `device="cuda"` with a `@cuda`
> marker, a **24x40** fixture commented "deliberately not square", and
> `assert neighbor["gray"].shape == (1, 1, 24, 40)`. The reviewer ran both tests **verbatim** from
> the plan in a pinned `99f309cb` tree: **2 passed** (CUDA available, NVIDIA A40).
>
> **This is a controller error, and worth naming as a process fact: a correction block is itself
> a claim, and gets no more trust than the claim it corrects.** The five corrections around it were
> each verified against `git show 9714e202:collab_splats/splats/pgsr.py`; this sixth paragraph was
> written from the plan text as remembered rather than as read at that SHA, and stale text is
> indistinguishable from current text unless you go and look.
>
> **So the two tests are owed as WORK, not as a coverage note — and they have teeth.** Three real
> defects in `render_neighbor` each survive the **entire** shipped `tests/splats` suite (308 passed)
> and are each killed by the plan's own Step 1 tests:
>
> | mutant in `render_neighbor` | Step 1's tests | shipped `tests/splats` |
> |---|---|---|
> | `model.render(..., height, width, ...)` swapped | FAILED | **308 passed** |
> | `render["plane_depth"].detach()` | FAILED | **308 passed** |
> | returns `cam_to_world` instead of `world_to_cam` | FAILED | **308 passed** |
>
> The old paragraph's fallback — "exercised through `tests/splats/test_trainer.py`'s
> `record_neighbor` monkeypatch, so it is not uncovered" — is hollow. That spy delegates to the real
> function but asserts only `set(seen) == {2}`; nothing constrains the returned dict. And
> `make_scene` defaults to **64x64 — square** (`tests/splats/synthetic.py:10`), so the width/height
> swap is invisible there by exactly the degeneracy this plan named two corrections ago. The lines
> execute; the behavior is untested. **Write them.**
>
> **The `min_depth` thread guard is the SIXTH instance of this branch's defect class — measured.**
> `test_forward_backward_noise_floors_the_whole_round_trip_with_one_min_depth` is the guard for
> correction #2 (one knob must govern the whole round trip, not split in two). It does not establish
> that. Un-threading **any single one** of the three sites survives the whole file, controller-verified
> in a pinned `99f309cb` tree with `__pycache__` cleared between runs:
>
> | mutant | `tests/splats/test_pgsr.py` |
> |---|---|
> | `:432` first `project` loses `min_depth=min_depth` | **21 passed** |
> | `:445` rescale clamp back to a literal `1e-6` | **21 passed** |
> | `:451` second `project` loses `min_depth=min_depth` | **21 passed** |
> | all three at once | 1 failed — the intended test |
>
> The test's own comment claims it rewrites "**both** `project` calls **and** the ray rescale", but
> the assertion (`floored[floored_valid].max() > 1.0`) is satisfied by any one site still honoring
> the argument. **The exact regression correction #2 exists to prevent lands green.** Same rule as
> the five before it: *when a guard aggregates N things, the test needs N cases, not one case over
> the aggregate* — here N = 3 call sites. The file already uses the right pattern twice (the
> `pixel_grid` and `_normalize_pixels` monkeypatch spies): spy on `pgsr.project`, assert **both**
> calls received `min_depth=<value>`, and probe the clamp separately.
>
> **Open scope question — ANSWERED with measurement, 2026-09-06.** `select_near_views` still
> hardcodes `depths.clamp(min=1e-6)` twice at `pgsr.py:356-357` while `project`'s identical floor is
> now the keyword `min_depth`. Two facts settle how to treat it:
>
> 1. **The no-tunables guard is not satisfiable by inlining the tunables it names.** Inlining
>    `project`'s floor while keeping the signature, or dropping the keyword entirely, is killed by
>    the `KEYWORD_ONLY` test and the behavioral test (controller-measured: 2 failed / 19 passed,
>    both intended, zero collateral). The suite covers that property — just not via the guard whose
>    name claims it.
> 2. **But `select_near_views`' literal has ZERO coverage on either side.** Changing it from `1e-6`
>    to `1e-2` gives **21 passed**. It is genuinely pre-existing (`9714e202:301-302`), so not a
>    regression and not Task 14's to fix — but nothing tests it today, so "leave it inline" is an
>    untested choice rather than a defended one. The same shape sits unmentioned at `:441`
>    (`points_near_cam[:, 2] > 0.1`).
>
> **Disposition: leave both literals alone in Task 14** — they are out of its scope and changing
> them is behavior risk with no test on either side. **Record them for Task 15's sweep**, which
> already owns the "explain every magic number" pass.
>
> **One more, non-blocking:** the no-tunables guard is a raw substring test over the file text, so a
> name merely *containing* one of the five trips it — renaming the `select_near_views` literal to a
> module constant `VISIBILITY_MIN_DEPTH` fails the guard on the `MIN_DEPTH` substring. Accidental
> strictness, and it would equally fire on a docstring mention. This is the shape Step 3 prescribed,
> so it is not a deviation; the stronger form is an AST check that every module-level `Name`
> assignment is in `{"logger"}` (controller-measured: control 21 passed, and it kills a tunable
> replanted under a *new* name, which the substring form does not — 1 failed, intended test, zero
> collateral). **Task 14's code-quality reviewer decides whether that swap is worth making.**
>
> ---
>
> **CORRECTIONS 7-9 — 2026-09-06, from Task 14's spec RE-review of the fix round `70644950`
> (verdict SPEC COMPLIANT, both round-1 findings closed, six mutants killed with clean attribution).
> These are plan defects the re-review surfaced; none of them is a defect in the shipped code.**
>
> **7. Step 1's code block still literally carries the assertion correction #4 strikes.** Correction
>    #4 above says `assert "_normalize_pixels" not in source` is STRUCK because the shipped dedupe
>    routes `sample_at_pixels` *through* the helper — the assertion now asserts the opposite of the
>    desired state. But the strike lives only in prose: the Step 1 code block still contains the line
>    verbatim, inside `test_module_constants_are_gone`. **A literal executor who skips the correction
>    block writes a test that fails on correct code.** Same shape as amendment A1 recorded for Task
>    12. The line must be deleted from the block, not merely annotated. The other three assertions in
>    that test (`THETA0`, `MIN_RAY_COSINE`, `MIN_DEPTH`) stand.
>
> 10. **NINTH instance of the vacuous-test class, and this one is two-sided. Measured, not
>     inferred.** `test_the_module_keeps_no_tunables_as_module_level_constants`
>     (`tests/splats/test_pgsr.py:344-350`, introduced by `99f309cb` itself) is the five-name
>     substring check this Step-1 block prescribes at plan lines **6416-6422**. It reads
>     `Path(pgsr.__file__).read_text()` and asserts five *spellings* are absent from the raw source.
>     The property it names — "the module keeps no tunables as module-level constants" — is about
>     module-level **bindings**. Substring ≠ binding, and the two axes fail in opposite directions:
>
>     | plant, in a pinned tree at `5b92ac2b` | `tests/splats` |
>     |---|---|
>     | module-level `PATCH_SIGMA = 0.5` (a real tunable, new name) | **356 passed, RC=0** — survives |
>     | a *comment* reading `... once the module constant THETA0, is now select_near_views' theta0 keyword` | **1 failed** — `test_the_module_keeps_no_tunables_as_module_level_constants` |
>
>     So the guard misses every constant it was not told the name of, and fires on prose that
>     contains no constant at all. **The second half is a live hazard for Task 15**, whose whole job
>     is a docstring and comment sweep over these nine files: a sentence explaining where `THETA0`
>     went fails this test on correct code. `collab_splats/splats/pgsr.py` has exactly **one**
>     module-level binding today (`logger` at `:25`, by AST).
>
>     The remedy already exists on this branch and is not `test_pgsr.py`'s: **`tests/splats/test_trainer.py:330-444`** is the
>     exemplar — an AST helper, a guard, and *two* meta-tests proving the guard sees a plant and
>     that its name set has not been narrowed. `7feed636` follows it for `scaffold.py`
>     (`test_the_scaffold_module_binds_no_module_level_constants`, module-level bindings `== {"logger"}`
>     by AST, plus a four-spelling parametrized meta-test). Port that shape: assert the AST
>     module-level binding set equals `{"logger"}`, and keep any name-based substring assertion only
>     for the properties an AST binding walk genuinely cannot see, in a test whose name says so.
>
>     **Provenance: the implementer was compliant.** The substring shape is what plan lines 6416-6422
>     literally prescribe, and correction 1 above only widened it from four names to five. Like
>     instance 8, the hole is this plan's, not `99f309cb`'s. Adjudication belongs to Task 14's
>     code-quality reviewer; if it declines, it must carry into Task 15 **before** that task writes
>     any prose naming a retired constant.
>
>     **This is the fourth confirmation of the standing lesson.** `99f309cb` closed the
>     five-name-substring instance in `pgsr.py`'s *production* code by moving the tunables to keyword
>     arguments — and shipped the same defect shape in the test it wrote to guard the move.
>
> **8. The CUDA-guard cross-reference is wrong on two of its three line numbers, at the plan's own
>    SHA.** The Step 1 prose cites `test_rendering.py:25` / `test_scaffold.py:22` /
>    `test_trainer.py:20` as the guard's siblings. Measured with an AST-free line scan for
>    `skipif`+`cuda`:
>
>    | file | at `9714e202` (the plan's SHA) | at HEAD |
>    |---|---|---|
>    | `test_rendering.py` | **29** (plan says 25) | 29 |
>    | `test_scaffold.py` | 22 ✓ | 25 |
>    | `test_trainer.py` | **26** (plan says 20) | 26 |
>
>    So two were already wrong when written and the third has since drifted. The guard line itself is
>    identical in all three files — `cuda = pytest.mark.skipif(not torch.cuda.is_available(),
>    reason="gsplat needs CUDA")`. **Grep by that text, never by these numbers.** This is the same
>    class as the `:441` → `:440` drift recorded for Task 15: a line number in a plan is stale the
>    moment any file above it changes, and cites in a *correction* block are no more trustworthy than
>    cites in the step it corrects.
>
> **9. The vacuous-test class has an EIGHTH instance, and it is inherited from THIS Step 1 block.**
>    `test_render_neighbor_returns_the_four_keys_the_multiview_loss_reads` names four keys and
>    guards two of them by **shape only**. Measured in a pinned `70644950` tree, `__pycache__`
>    cleared between runs:
>
>    | mutant in `render_neighbor` | `tests/splats` | why it lives |
>    |---|---|---|
>    | `"plane_depth": render["depth"]` | **346 passed, 0 failed** | same shape `(1,24,40,1)`, max abs diff **1.446**, both `requires_grad` |
>    | `gray` built without `/255.0` | **346 passed, 0 failed** | same shape `(1,1,24,40)`, max abs diff **244.6** |
>    | `"intrinsics": intrinsics.clone()` (positive control) | 1 failed | killed by the fix round's `is intrinsics` assertion |
>
>    plane_depth-versus-depth is the exact distinction the pgsr module exists for, and the test's
>    name reads as coverage — the tell, every time. The aggregation axis is **key x property**: four
>    keys, and for each, identity/value versus shape. Two cells are empty. **Because the test came
>    from this block essentially as written, the implementer was compliant and the defect is the
>    plan's.** Adjudication belongs to Task 14's code-quality reviewer; if it declines, it carries
>    into Task 15's sweep.
>
> **And the standing lesson, now three times on this branch: a review that closes instance N ships
> instance N+1 in its own remedy.** `70644950` closed instance 6 (the `min_depth` thread) and
> introduced instance 8 in the same commit.

**Task 14 outcome — fix round 3, `eb5bc9a8`.**

Round 3 addressed the vacuous-test hole in `_module_level_bindings`'s meta-test
(`tests/splats/test_pgsr.py`): the guard walks four node shapes, and the test previously exercised
them through a single fixture whose failure could not be attributed to any one arm. The remedy gives
the guard **a case per node shape** (+18/-6, one file, zero `collab_splats/` paths).

> **Blast radius is not the acceptance criterion — a discriminating case per arm is.**
> An earlier draft of this block claimed arm 1 (`ast.Assign`) admits "no unique single-arm killer"
> and proposed a weakened rule. The round-3 review disproved that. Two different properties were
> being conflated:
>
> - *Does mutating arm 1 fail exactly one test?* No — **4 of 25**, and that is **irreducible**.
>   Arms 3 (`node.targets`) and 4 (`ast.walk(target)`) are reachable only through `ast.Assign`, so
>   removing the parent necessarily disables its children's inputs. The grammar makes it structural:
>   `AnnAssign` has exactly one target that can never be a `Tuple` (`a, b: int = 1` is a
>   SyntaxError), and `pgsr.py`'s only module-level binding, `logger = logging.getLogger(__name__)`,
>   is itself an `Assign`, so the base test is unavoidable collateral too.
> - *Does arm 1 have a case that fires only for arm 1?* **Yes** — `("THETA0 = 5.0", "THETA0")`
>   passes under the arm-2, arm-3 and arm-4 mutations and fails only when `ast.Assign` is dropped.
>
> So the branch's N-cases rule **is** satisfied for arm 1, unweakened. What is unachievable is
> exclusive *attribution* of a mutation to a single test, which is a different and unnecessary bar.
> No structural exception to the rule is needed or granted.
>
> Two further corrections from round 3, both accepted after independent check:
> - Extending the isinstance tuple with `ast.Import, ast.ImportFrom` (suggested in the dispatch
>   brief) raises `AttributeError: 'Import' object has no attribute 'target'` — those nodes carry
>   `names`, not `target`. The extension is wrong and was not made.
> - The test counts in the brief were stale: at `dae59f89` `tests/splats` is **367**, not 360, and
>   the wide gate is **462**, not 452. At `eb5bc9a8` they are **368** and **463**.

**Task 14 — fix round 3 did not close the finding; round 4 follows.**

The re-review of `eb5bc9a8` returned **NOT APPROVED**. `node.targets` is a list of N targets and the
round-3 remedy added exactly one case, `("logger = THETA0 = 5.0", "THETA0")`, which plants the
tunable in the **second** position. That kills head-truncation and nothing else; the mirror direction
is a live escape. Measured on pinned trees at `eb5bc9a8`:

| mutation | result |
|---|---|
| `node.targets` → `node.targets[:1]` | 1 failed / 367 — killed |
| `node.targets` → `node.targets[-1:]` | **RC=0, 368 passed — SURVIVES** |
| that mutation + `THETA0 = logger = logging.getLogger(__name__)` planted in `pgsr.py` | **RC=0, 368 passed** — a module-level tunable ships green |

This is the branch's own defect class reappearing inside the remedy sent to close it — the **eighth**
confirmed instance of "the review that closes instance N ships instance N+1 in its own remedy," and
the first where it recurred on the same finding. A remedy that adds one case to an N-way aggregate
reproduces the defect at a smaller scale; the fix must enumerate the aggregate's ends, not sample one.

Three minor findings accompanied it:
- The new comment's claim that a live module-level `THETA0` ships green holds only for a *contrived*
  plant. The natural reintroduction `THETA0 = MIN_DEPTH = 5.0` is caught by the base equality test
  (1 failed / 366); only a chained plant whose surviving target is the allow-listed `logger` escapes
  (367 passed). An unconditional coverage claim in a comment is this defect class's own tell.
- `_module_level_bindings`'s docstring omits the chained-targets arm — the arm this work made
  load-bearing — and claims `sorted`, which nothing asserts: `return sorted(` → `return list(`
  survives the full suite (RC=0, 368 passed).
- The new comment uses ASCII `--` where the file uses `—`.

**Task 14 — fix round 4 landed as `a1fd1e46`** (`tests/splats/test_pgsr.py` only, +19/−4). It adds the
mirror case `("THETA0 = logger = 5.0", "THETA0")` beside round 3's, so both ends of `node.targets` are
pinned; each direction now has a killer that names itself in the param id, and the live-plant escape
(`targets[-1:]` plus `THETA0 = logger = logging.getLogger(__name__)` in `pgsr.py`) fails instead of
shipping green. It also pinned the previously-unasserted `sorted` with a new test whose source order
is reversed from sorted order, enumerated all four arms in the helper's docstring, and narrowed the
overclaiming comment. `tests/splats` 368 → 370, wide gate 463 → 465, 0 skips, RC=0 in pinned trees.
Spec re-review in flight.

Two corrections it made to the round-4 brief, both accepted:

- The brief's claim that `targets[:1]` plus a `logger`-surviving plant "escapes, RC=0/367" is
  **impossible by construction** — on the round-3 tree `[:1]` already fails its own case, so no plant
  can make that mutant green. Measured RC=1. The neighboring row undercounted too (2 failed / 366,
  not 1), though the underlying claim holds: the natural plant `THETA0 = MIN_DEPTH = 5.0` with **no**
  mutation gives exactly one failure, the base test.
- Control drift was mis-attributed to a branch move. Committed HEAD was still 368 and `test_pgsr.py`
  byte-identical to `eb5bc9a8`; the worktree's 369 was a peer's uncommitted `test_scaffold.py` test.
  **A count taken in the shared worktree is not a control.**

It left one residual open for adjudication: the tuple-target case asserts only one of the two names it
binds. It argues no single-arm mutation reaches that asymmetry today, because `ast.walk(target)` is
generic (`[-1:]` yields the `Name`'s `Store` child, not a `Name`, and kills 7 tests), and that it
becomes live only if anyone replaces the walk with explicit `target.elts` handling. Routed to the
re-review to decide on measurement.

**Task 14 — round 4's re-review returned NOT APPROVED; round 5 follows.** The blocking finding round 4
was sent to close **is** closed and was verified end to end: both ends of `node.targets` now have a
uniquely-named killer, and the live-plant escape fails instead of shipping green. But round 4 shipped
instance N+1 inside its own remedy — the **ninth** confirmed occurrence on this branch, and the second
consecutive round on this same task.

The new `test_the_module_binding_check_returns_its_names_sorted` has exactly one input,
`"logger = 1\nTHETA0 = 5.0\n"` expecting `["THETA0", "logger"]`. Two bindings, arranged so source
order is the exact reverse of sorted order — which the test's own comment says out loud. At N=2 in
reverse-sorted order, "sort it" and "reverse the input" produce the identical list, so the case pins
`sorted` against only one of the two reachable wrong orderings. Measured at `a1fd1e46`:

| mutation of the helper's return | result |
|---|---|
| `sorted(…)` → `list(…)` | 1 failed, `…returns_its_names_sorted`, unique |
| `sorted(…, reverse=True)` | 1 failed, same test, unique |
| **`sorted(…)` → `list(…)[::-1]`** | **RC=0, 370 passed — SURVIVES** |

A helper returning source order *reversed* ships the suite green. That bites on round 4's own stated
rationale for pinning `sorted` — that the base test "becomes order-dependent the moment `pgsr.py`
gains a second binding" — because under the surviving mutant the helper is order-dependent in exactly
that way and nothing notices. Remedy measured by the re-review: a three-binding input whose sorted
order is neither the source order nor its reverse (`"ALPHA = 0\nlogger = 1\nBETA = 5.0\n"` →
`["ALPHA", "BETA", "logger"]`) kills `[::-1]` uniquely with nothing else disturbed.

One minor finding travels with it: round 4's rewritten comment claims *"the surviving target still
binds `logger` a second time, so the list differs either way"*. Over all four (case × truncation)
combinations the surviving target is `logger` in only 2 of 4; in the other 2 it is `THETA0` and the
`!=` passes for the opposite reason. The conclusion holds in all four, only the mechanism is wrong —
a round-3 sentence generalised to "either truncation" without re-checking.

> **Residual tuple-target caveat — ADJUDICATED: do not add the mirror case today, and record a
> precondition instead.** Measured both sides. Against today's generic `ast.walk(target)`, the mirror
> case `("THETA0, MIN_DEPTH = 5.0, 1e-6", "THETA0")` is satisfiable (371 passed) but has **no unique
> killer** under any single-arm mutation — it merely fails alongside the existing `MIN_DEPTH` case.
> Against a hypothetical `target.elts` refactor, `target.elts[-1:]` **does** escape (RC=0, 370
> passed) and the mirror case then kills it uniquely.
>
> Reason: the N-cases rule is about aggregates whose arms a single-arm mutation can isolate.
> `ast.walk` is a generic BFS whose ends are the `Tuple` node and a `Store` ctx, not the tuple's
> names, so there is no arm to discriminate; `[:1]` collapses to the already-covered target itself and
> `[-1:]` yields no `Name` at all and fails 7 tests. **Adding a case that kills nothing uniquely is
> the opposite failure mode but still a spec failure.**
>
> **Precondition, binding on any future change:** the commit that replaces `ast.walk(target)` with
> positional `target.elts` handling must add
> `("THETA0, MIN_DEPTH = 5.0, 1e-6", "THETA0")` in the same commit. The docstring already claims
> *"Every `Name` under a target"* — coverage the tests only half demonstrate today.

Three corrections the re-review made to the round-4 brief, all accepted:

- **The wide-gate numbers were stale by exactly 6.** The brief said "463 → 465 in pinned clean
  trees"; measured at `a1fd1e46^`/`a1fd1e46` it is **469 → 471**, 0 skipped, RC=0 both. Cause:
  `efc4d7b1` (two commits earlier) adds 6 tests to `tests/mesh/test_splats_adapter.py`, so round 4
  measured its wide gate on a tree predating it. The delta (+2) was right; the absolutes and the
  "in pinned clean trees" claim were not.
- **The live-plant row understated its precondition.** `targets[-1:]` plus
  `THETA0 = logger = logging.getLogger(__name__)` escapes only when the plant **replaces** `pgsr.py`'s
  existing `logger = logging.getLogger(__name__)`. **Appended**, it gives RC=1 — the base test sees
  `['logger','logger'] != ['logger']`. A reader reproducing the row verbatim would conclude the escape
  does not exist.
- **The `third_party` skip-downgrade rule does not explain anything about this gate.** A no-symlink
  tree at `a1fd1e46` gives an identical wide gate (471 / 0 skipped / RC=0): every skip guard in these
  three paths is CUDA-gated and CUDA is available. Keep symlinking, but **the protection is the
  skip-count-is-0 assertion, not the symlinks**, and 0 skips is not evidence the symlinks took.


Three small changes. The module constants become keyword arguments like everywhere else, one duplicated pixel-normalization helper is inlined, and the ~50-line neighbor-render block Task 10 lifted out of the trainer lands here as a function — this module already owns everything else PGSR needs.

**Files:**
- Modify: `collab_splats/splats/pgsr.py`
- Modify: `tests/splats/test_pgsr.py`

> **C14-1 — REFUTED BY MEASUREMENT (2026-09-06, at `1bdadecd`). NOT a blocker. `step=None` is
> correct; do not thread `step` through `render_neighbor`.** The original finding, and the evidence
> that overturns it, are both kept below — the call-site facts were right, the conclusion was not.
>
> **What was right.** `trainer.py:543` does pass `min(step // cfg.sh_degree_interval, cfg.sh_degree)`
> to today's neighbor render, and `gaussian.py:248` does read
> `sh_degree = self.sh_degree if step is None else min(step // self.sh_degree_interval, self.sh_degree)`.
> So `step=None` genuinely renders the neighbor at a different SH degree.
>
> **Why it does not matter.** `render_neighbor` returns only `plane_depth`, `gray`, `world_to_cam`
> and `intrinsics`. `gray` comes from the neighbor image, not the render. And `plane_depth` does not
> depend on the SH degree — measured on CUDA against one `Gaussians` model with `shN` perturbed to
> non-zero, rendering the same view at `sh_degree` 0, 2 and 3:
>
> | key | sh2 vs sh0 | sh3 vs sh0 |
> |---|---|---|
> | `rgb` | differs, max 2.235e-01 | differs, max 3.448e-01 |
> | `plane_depth` | **bit-identical** | **bit-identical** |
> | `plane_normal` | **bit-identical** | **bit-identical** |
> | `plane_distance` | **bit-identical** | **bit-identical** |
> | `depth` | **bit-identical** | **bit-identical** |
> | `alpha` | **bit-identical** | **bit-identical** |
>
> `rgb` moving is the positive control: it proves the degree really varied and the probe was not
> measuring a no-op.
>
> **The backward pass agrees.** `render_neighbor`'s render is deliberately not detached, so the
> gradient matters. Back-propagating a `plane_depth`-only loss through one model at degree 0 and at
> degree 3 gives **exactly zero** `sh0` and `shN` gradient at both degrees, and geometry gradients
> that differ by ~1e-4 — the same ~1e-4 a **sh0-vs-sh0 control** produces, i.e. gsplat's backward
> atomicAdd nondeterminism, not an SH effect.
>
> **Two traps this probe hit, worth repeating for anyone re-measuring it.** The first two attempts
> reported everything bit-identical *including* `rgb`, which reads as "SH degree changes nothing".
> Both were false greens: (1) `Gaussians.__init__` inits `shN` to zeros, so a fresh model renders
> identically at every degree — perturb `shN` first; (2) a camera at `c2w[2,3] = +3.0` with identity
> rotation looks **away** from a cloud at the origin and renders nothing at all (`alpha.sum() == 0`) —
> use `-3.0` and assert `alpha.sum() > 0` before trusting any comparison. Separately,
> `Gaussians.__init__` is nondeterministic, so gradients must be compared on **one** model instance
> across two backward passes; rebuilding per degree makes a same-degree control show the full
> "difference".
>
> This also matches this task's own "A note on `step=None`" below, which was right all along.
>
> **Consequence for Task 10:** its C10-10 adds `render_neighbor` to `pgsr.py` early, using this
> task's body verbatim. That is safe — no `step` parameter, no signature change here.
>
> ---
>
> **Original finding, superseded — kept for the record.**
>
> **C14-1 (BLOCKER for this task — it changes frozen numerics). `render_neighbor` as written in
> this task passes `step=None`, and that is wrong for the vanilla path.**
>
> Found by the Task 8 implementer while writing `Scaffold.render`, and independently flagged by the
> Task 7 code-quality reviewer as an interface-documentation defect that had "already propagated into
> the plan".
>
> `step=None` is harmless for `Scaffold` — its `render` ignores `step` outright, because the SH
> schedule is vanilla-only. It is **not** harmless for `Gaussians`, whose `step` sets the
> coarse-to-fine SH degree at `gaussian.py:248`. Today's trainer passes
> `sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)` to the neighbor render as well, so
> passing `None` would render the neighbor view at **full SH degree from step 0**.
>
> `train()`'s numerics are frozen by Ground Rule §7, and the parity gates (Tasks 11, 13, 19) run
> against `before_v3`. A silent SH-degree change on the PGSR neighbor path is exactly the kind of
> defect those gates surface as an unexplained dPSNR with no failing test to point at. Pass the real
> `step` through.
>
> Root cause worth fixing at the same time: `gaussian.py:240-241` documents `step=None` as "what a
> finished model or a checkpoint wants", which invites reading it as a training-time "don't care".
> Task 15 tightens that wording to "export/inference only — training-time renders must pass `step`";
> this task must not wait for it.

> **CODE-QUALITY REVIEW 2026-09-06 — CHANGES REQUESTED, measured at `70644950` (fix round 1).**
> Three blocking findings, an eleventh member of the vacuous-test class found inside `99f309cb`'s own
> docstring, two corrections to the controller's fix-round-1 brief, and one open scope question
> **closed by measurement**. Gate at `70644950`: **346 passed / 0 failed / 0 skipped / RC=0** in a
> pinned tree. Style and conventions clean.
>
> **Q14-1 (BLOCKING) — instance 9 confirmed independently, at a different SHA than the controller's.**
> The reviewer reproduced both halves of the module-constants guard's failure at `70644950`
> (controller measured at `5b92ac2b`): a module-level `DEPTH_EPS = 1e-6` used as `project`'s default
> **survives at 346 passed**, `PATCH_SIGMA = 0.5` **survives at 346 passed**, and a *comment* naming
> `THETA0` **fails** the guard on code containing no constant. The remedy is the AST binding-set
> shape from `tests/splats/test_trainer.py:330-444`, with a **parametrized** meta-test over
> `("THETA0 = 5.0", "MIN_DEPTH: float = 1e-6")` — control 348 passed, kills both plants, silent on
> the comment.
>
> **The substring loop must be DELETED, not kept beside the AST check.** Measured: with both present,
> the Task-15 false positive survives intact; the comment plant only reaches 348 passed once the loop
> is gone. The case the loop appears to add is subsumed — a module-level reintroduction under any old
> name is an `Assign`/`AnnAssign` the walk catches.
>
> **Q14-2 (BLOCKING) — instance 8 confirmed; `plane_depth` and `gray` are guarded by shape only.**
> All three rows of the key × property matrix reproduce at `70644950`:
> `"plane_depth": render["depth"]` **survives at 346**, `gray` without `/255.0` **survives at 346**,
> `"intrinsics": intrinsics.clone()` is killed. `render["depth"]` is the alpha-weighted depth and
> `render["plane_depth"]` the ray-plane depth (`rendering.py:235`) — **the one distinction this module
> exists to make, able to ship wrong with a same-shape signature.** Adjudicated *fix now, not in Task
> 15/16*. The remedy is an identity assertion against a `model.render` spy.
>
> Severity split worth keeping: the `gray` half is milder. Its consumer `lncc` is invariant to a
> per-patch gain — `lncc(ref, near)` vs `lncc(ref, near * 255.0)` differ by **6.4e-11** with an
> identical keep mask. A real inconsistency with `to_gray(target["rgb"][0])` at `losses.py:447`, but
> numerically inert for the loss.
>
> **Q14-3 (BLOCKING) — INSTANCE 10, found in this review and not in the brief that commissioned it.**
> `99f309cb` rewrote `_normalize_pixels`' docstring from *"Both the reference and the neighbor patch
> sampling go through here"* to *"`patch_ncc`'s reference and neighbor patches **and**
> `sample_at_pixels`' lookups **all** go through here, so the **three** conventions cannot drift"* —
> and guarded exactly one of the three. Behavior-preserving mutants (the identical formula inlined):
>
> | mutant | `tests/splats` |
> |---|---|
> | `sample_at_pixels` inlines its own copy | 1 failed — the intended test |
> | `patch_ncc` inlines **both** its copies | **346 passed, RC=0** — survives |
>
> Same tell as every other instance: the test name (`..._normalizes_through_the_helper_patch_ncc_uses`)
> reads as coverage of the pairing. **The three-way claim is true** — the mutant table shows all three
> sites really do route through the helper — so the defect is the missing guard, not the docstring.
> Guard it; do not revert the docstring, which would edit `pgsr.py` and void Task 13's live parity.
>
> **Q14-4 — TWO CORRECTIONS TO THE CONTROLLER'S FIX-ROUND-1 BRIEF.**
>
> 1. **The brief's "max abs diff 1.446" is NOT REPRODUCIBLE, and no number derived from that fixture
>    can be.** `_neighbor_inputs()` seeds only the numpy RNG; `Gaussians.__init__` consumes an
>    **unseeded torch RNG**. Three runs of the identical fixture gave **13.441 / 5.110 / 1.730**.
>    Direction is stable (the tensors are O(1)–O(10) depth units apart); the magnitude is not. This
>    **rules out any value-threshold guard in this file** and forces the identity-based remedy. Treat
>    it as a standing rule for `test_pgsr.py`, not a one-off.
> 2. The brief's claim (b) is stale: at `70644950` the "delete `project`'s `min_depth`, hardcode
>    `1e-6`" mutation gives **3 failed / 343 passed**, not 2 —
>    `test_forward_backward_noise_threads_one_min_depth_into_both_projections` now fires too. The
>    conclusion is strictly stronger: three tests carry the property.
>
> **Q14-5 — the `select_near_views` clamp question is CLOSED. It stays inline, and must NOT become a
> fourth keyword.** This supersedes the "record it for Task 15's sweep" disposition above, which was
> made when the literal had zero coverage on either side. Measured on a ring scene built with points
> at, behind and astride the cameras (`z = -4.0`, `-4.0 - 1e-9`, `-3.9999999`, `-8.0`, `+40.0`):
>
> ```
> shipped floor 1e-6: [[1, 2], [0, 2], [1, 0]]
>   floor 1e-12 / 1e-9 / 1e-4 : IDENTICAL
>   no clamp at all           : IDENTICAL
> ```
>
> The clamp's only consumer is `pixel_u`/`pixel_v`, feeding the boolean `visible`, which is separately
> conjoined with `depths > 0` (`pgsr.py:358`) — already discarding every row where the clamp could
> bite. It is inf-hygiene on a discarded value inside a `@torch.no_grad()` function returning discrete
> indices. `project`'s floor is a different object: it bounds a **derivative** on a gradient-carrying
> path, which is why that one is a knob. **Task 15 must not promote it.** (`:441`'s
> `points_near_cam[:, 2] > 0.1` is untouched by this and still stands as recorded.)
>
> **Q14-6 — two of the controller's stated worries were unfounded, with the measurement for each.**
> `test_plane_depth_and_the_two_projections_take_their_floors_as_keyword_only_arguments` **does have
> three cases** — it loops a 3-entry dict with per-iteration asserts, and breaking exactly one of the
> three turns it red. And the three `..._refuses_..._positionally` tests **do** pin the reason: with
> `select_near_views` made to bind the 8th positional *and* raise `TypeError('unrelated failure')`
> from its body, `test_select_near_views_refuses_its_scoring_shape_positionally` fails — the `match=`
> string is load-bearing. (A naive probe planting only the raise is inconclusive: the arity error
> fires at binding and the body never runs. Note the method, not just the result.)
>
> **Q14-7 — observations for Task 15 and Task 16, not blocking.**
>
> - **None of the five new knobs is reachable from config.** `losses.py:187-190` forwards only
>   `num_views` and `max_points` to `neighbor_selection`, so `theta0` / `sigma_below` / `sigma_above`
>   never leave their defaults at `trainer.py:279`; `rendering.py:235` calls `compute_plane_depth`
>   with three positionals, so `min_cosine` is unreachable; `losses.py:417` calls
>   `forward_backward_noise` with six positionals, so `min_depth` is unreachable. All five are
>   always-default in the shipped pipeline. **Task 15 must either wire them into the `pgsr_multiview`
>   spec or say in the docstrings that they are an API/test-only surface** — "a caller can move one
>   without editing the module" is true today only for a Python caller, not a yaml one.
> - **The "refuses positionally" tests cannot distinguish keyword-only from parameter-deleted.**
>   Deleting `min_cosine` entirely and hardcoding `1e-4` leaves
>   `test_plane_depth_refuses_its_floor_positionally` green — the arity message is identical; the
>   sibling signature test catches it via `KeyError`. Nothing to fix. Recorded so nobody later deletes
>   a signature test believing the positional test covers it.
> - `forward_backward_noise` is the only one of the four functions given a keyword here that did not
>   gain an `Args:` block. Cosmetic; the module already mixes both styles (6 of 18 functions).
>
> **Q14-8 — every quantified comment in the two commits reproduces to the digits written:**
> the ray-rescale test's "11.63-11.83 px" → `[11.6338, 11.8340]`; `test_patch_ncc_prefers_...`'s
> "~2e-5 against ~0.74" → `2.379e-05` / `0.7353`; `test_patch_ncc_is_zero_...`'s "~1e-7" → `1.095e-07`.
> The commit's central claim also holds: one `min_depth` genuinely governs all three clamps, each site
> broken alone killed by exactly one intended test with zero collateral.
>
> **Q14-9 — the standing lesson, fifth confirmation, and the first time it was applied FORWARD.**
> *"A review that closes instance N ships instance N+1 in its own remedy."* This reviewer applied the
> rule to its **own** Q14-1 remedy before proposing it, spotted that `(ast.Assign, ast.AnnAssign)` is
> a two-node tuple — the exact shape of instance 4 — and closed it with the parametrized meta-test:
> deleting `ast.AnnAssign` gives 1 failed on the `MIN_DEPTH: float = 1e-6` case, which a single-case
> meta-test would have missed. **This is the first remedy on this branch that was audited for the
> defect class before it shipped rather than after.** Make it the default.
>
> **One error in the review itself, recorded so it does not propagate:** its report labels `378a9e91`
> as "Task 15's". It is **Task 12's** ("refactor(splats): AnchorStrategy owns its densification
> state"). The tree it measured is correct; only the label is wrong.

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_pgsr.py`. **The module has no CUDA guard yet — add one**, matching
`test_rendering.py:25` / `test_scaffold.py:22` / `test_trainer.py:20`:

```python
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")
```

Every test below that renders is decorated with it. gsplat's rasterization kernels are
registered for CUDA only, so a `device="cpu"` render raises
`NotImplementedError: Could not run 'gsplat::projection_ewa_3dgs_fused_fwd' ... from the 'CPU'
backend` — measured 2026-09-06. The fixtures are 24x40 and never square: a height/width swap is
invisible on a square frame, which is how the swap in `model.render` survived on this branch.

```python
@cuda
def test_render_neighbor_returns_the_four_keys_the_multiview_loss_reads():
    cfg = SplatsConfig.from_dict({"max_steps": 10, "losses": {}})
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, (200, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (200, 3)).astype(np.uint8)
    model = Gaussians(cfg, points, colors, scene_scale=1.0, n_views=2, device="cuda")

    # 24x40, deliberately not square: render_neighbor reads `height, width = image.shape[:2]`
    # and passes them to model.render(..., width, height, ...), so a swap only shows here
    image = rng.integers(0, 255, (24, 40, 3)).astype(np.uint8)
    cam_to_world = torch.eye(4)[None].cuda()
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[[40.0, 0.0, 20.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]]).cuda()

    neighbor = render_neighbor(model, image, cam_to_world, intrinsics, torch.tensor([1]).cuda())

    assert set(neighbor) == {"plane_depth", "gray", "world_to_cam", "intrinsics"}
    assert neighbor["plane_depth"].shape == (1, 24, 40, 1)
    # to_gray returns (1, 1, H, W) to match torchvision's Grayscale, NOT (1, H, W, 1).
    # Measured 2026-09-06; the shape this plan asserted before was wrong in both conventions.
    assert neighbor["gray"].shape == (1, 1, 24, 40)
    # world_to_cam, not cam_to_world: the multi-view loss projects into this view
    assert torch.allclose(neighbor["world_to_cam"], torch.linalg.inv(cam_to_world))


@cuda
def test_render_neighbor_keeps_the_gradient_path():
    """
    Upstream does not detach the neighbor: the geometric term pulls both plane depths together.
    """
    cfg = SplatsConfig.from_dict({"max_steps": 10, "losses": {}})
    rng = np.random.default_rng(0)
    model = Gaussians(
        cfg,
        rng.uniform(-1.0, 1.0, (200, 3)).astype(np.float32),
        rng.integers(0, 255, (200, 3)).astype(np.uint8),
        scene_scale=1.0,
        n_views=2,
        device="cuda",
    )
    image = rng.integers(0, 255, (24, 40, 3)).astype(np.uint8)
    cam_to_world = torch.eye(4)[None].cuda()
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[[40.0, 0.0, 20.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]]).cuda()

    neighbor = render_neighbor(model, image, cam_to_world, intrinsics, torch.tensor([1]).cuda())

    assert neighbor["plane_depth"].requires_grad


def test_select_near_views_takes_its_scoring_shape_as_keywords():
    signature = inspect.signature(select_near_views)

    assert signature.parameters["theta0"].default == 5.0
    assert signature.parameters["sigma_below"].default == 1.0
    assert signature.parameters["sigma_above"].default == 10.0
    assert signature.parameters["theta0"].kind is inspect.Parameter.KEYWORD_ONLY


def test_plane_depth_and_project_take_their_floors_as_keywords():
    assert inspect.signature(plane_depth).parameters["min_cosine"].default == 1e-4
    assert inspect.signature(project).parameters["min_depth"].default == 1e-6


def test_module_constants_are_gone():
    source = Path(pgsr.__file__).read_text()

    assert "THETA0" not in source
    assert "MIN_RAY_COSINE" not in source
    assert "MIN_DEPTH" not in source
    assert "_normalize_pixels" not in source
```

Extend the imports:

```python
import inspect
from pathlib import Path

import numpy as np
import torch

from collab_splats.splats import pgsr
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.pgsr import plane_depth, project, render_neighbor, select_near_views
from collab_splats.splats.trainer import SplatsConfig
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_pgsr.py -q
```

Expected: collection error, `ImportError: cannot import name 'render_neighbor'`.

- [ ] **Step 3: Move the three constant groups into signatures**

`THETA0`, `SIGMA_BELOW`, `SIGMA_ABOVE` become keyword arguments of `select_near_views`, carrying the citation comment that stood above them:

```python
def select_near_views(
    world_to_cam: Tensor,
    intrinsics: Tensor,
    points: Tensor,
    height: int,
    width: int,
    num_views: int = 5,
    max_points: int = 20000,
    *,
    theta0: float = 5.0,
    sigma_below: float = 1.0,
    sigma_above: float = 10.0,
) -> list[list[int]]:
    """
    Per view, the `num_views` best-scoring neighbors by MVSNet covisibility.

    Args:
        world_to_cam: (N, 4, 4) OpenCV poses.
        intrinsics: (N, 3, 3) camera matrices at (height, width).
        points: (P, 3) world points; the shared set the score sums over.
        height: frame height in pixels.
        width: frame width in pixels.
        num_views: neighbors to keep per view.
        max_points: points to score against; the cloud is strided down to this.
        theta0: the ideal angle, in degrees, that two cameras subtend at a shared point.
        sigma_below: score falloff for pairs closer together than `theta0` — sharp, because a
            near-duplicate view carries almost no new information.
        sigma_above: falloff for pairs further apart; ten times gentler, because a wide
            baseline is merely harder to match, not useless.

    Returns:
        One list of neighbor indices per view, best first. Empty for a view with no
        co-visible partner.

    Ported from GS-SR @ main, gssr/utils/mvsnet_utils.py:306-343.
    """
```

`MIN_RAY_COSINE` becomes `plane_depth(..., *, min_cosine: float = 1e-4)` and `MIN_DEPTH` becomes `project(..., *, min_depth: float = 1e-6)`. Each keeps its explanation on the argument, in the docstring:

```python
        min_cosine: floor on the ray-plane cosine. Bounds a derivative rather than fixing a
            value: a plane within this of edge-on has an unbounded depth gradient.
```

```python
        min_depth: floor on the perspective divide, for the same reason — a point this close
            to the image plane projects with an unbounded pixel gradient.
```

Delete the four module-level constants and the comment block above them.

- [ ] **Step 4: Inline `_normalize_pixels`**

Two functions normalize pixel coordinates to `[-1, 1]` for `grid_sample`. Keep the one inside `sample_at_pixels` and delete the helper:

```python
def sample_at_pixels(image: Tensor, pixels: Tensor) -> Tensor:
    """
    Bilinearly sample `image` at floating-point pixel coordinates.

    Args:
        image: (1, H, W, C) tensor to sample from.
        pixels: (1, H, W, 2) pixel coordinates, gsplat convention (pixel i's center at i + 0.5).

    Returns:
        (1, H, W, C) sampled values; out-of-frame coordinates clamp to the border.
    """
    height, width = image.shape[1:3]

    # grid_sample wants [-1, 1] with the same half-pixel center convention
    normalized = torch.stack(
        [pixels[..., 0] / width * 2.0 - 1.0, pixels[..., 1] / height * 2.0 - 1.0], dim=-1
    )
    sampled = F.grid_sample(
        image.permute(0, 3, 1, 2), normalized, mode="bilinear", padding_mode="border", align_corners=False
    )
    return sampled.permute(0, 2, 3, 1)
```

Have `pixel_rays` call `pixel_grid` rather than rebuilding the meshgrid:

```python
def pixel_rays(height: int, width: int, intrinsics: Tensor) -> Tensor:
    """
    Camera-frame ray directions with unit z, one per pixel: `((u - cx)/fx, (v - cy)/fy, 1)`.

    - Not normalized: `plane_depth` divides a plane distance by the ray's z-component, which is
      1 by construction here. Normalizing these would silently rescale every rendered depth.

    Args:
        height: image height in pixels.
        width: image width in pixels.
        intrinsics: (C, 3, 3) camera matrices.

    Returns:
        (C, H, W, 3) camera-frame directions with z == 1.
    """
    # pixel_grid is flat (H*W, 2); reshape back to the image so the per-camera broadcast works
    pixels = pixel_grid(height, width, intrinsics.device, intrinsics.dtype).reshape(height, width, 2)
    grid_u, grid_v = pixels[..., 0], pixels[..., 1]

    # Broadcast the per-camera intrinsics over the pixel grid
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]
    ray_x = (grid_u[None] - cx) / fx
    ray_y = (grid_v[None] - cy) / fy
    return torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)
```

The returned values are unchanged — `pixel_grid` builds the same `+ 0.5` centers in the same
row-major order — so this is a dedupe, not a numerical edit. Prove it before moving on:

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -c "
import torch
from collab_splats.splats.pgsr import pixel_rays
K = torch.tensor([[[300.0, 0, 64.0], [0, 310.0, 48.0], [0, 0, 1.0]]])
rays = pixel_rays(96, 128, K)
u = (torch.arange(128) + 0.5)[None, None]
v = (torch.arange(96) + 0.5)[None, :, None]
assert torch.allclose(rays[..., 0], (u - 64.0) / 300.0)
assert torch.allclose(rays[..., 1], (v - 48.0) / 310.0)
assert (rays[..., 2] == 1.0).all()
print('PIXEL_RAYS UNCHANGED')
"
```

Expected: `PIXEL_RAYS UNCHANGED`.

- [ ] **Step 5: Add `render_neighbor`**

```python
def render_neighbor(
    model, image: np.ndarray, cam_to_world: Tensor, intrinsics: Tensor, camera_id: Tensor
) -> dict[str, Tensor]:
    """
    Render one co-visible neighbor view for the multi-view losses.

    Only the neighbor's plane depth and its grey image are read, so it needs no normals, no
    appearance correction and no background composite. The render is NOT detached: upstream
    lets the geometric term pull both views' plane depths towards each other, and detaching
    would make it a one-sided fit.

    Args:
        model: the `Gaussians` or `Scaffold` being trained.
        image: (h, w, 3) uint8 neighbor frame, already downscaled to this step's resolution.
        cam_to_world: (1, 4, 4) neighbor pose, already pose-corrected.
        intrinsics: (1, 3, 3) neighbor camera matrix at the same resolution.
        camera_id: (1,) long neighbor view index.

    Returns:
        {"plane_depth", "gray", "world_to_cam", "intrinsics"} — what `patch_ncc` and
        `forward_backward_noise` read.

    Ported from GS-SR @ main, gssr/scene/pgsr_scene.py (`get_train_loss_dict`).
    """
    height, width = image.shape[:2]
    render, _ = model.render(
        cam_to_world,
        intrinsics,
        width,
        height,
        camera_id,
        step=None,
        render_normals=False,
        render_plane=True,
    )
    gray = to_gray(torch.from_numpy(image).to(intrinsics.device).float()[None] / 255.0)
    return {
        "plane_depth": render["plane_depth"],
        "gray": gray,
        "world_to_cam": torch.linalg.inv(cam_to_world),
        "intrinsics": intrinsics,
    }
```

Add `import numpy as np` to the module imports.

**A note on `step=None`:** the neighbor renders at the model's current SH degree in the old code (`min(step // cfg.sh_degree_interval, cfg.sh_degree)`), and at full degree here. This is deliberate and harmless — only `plane_depth` is read from the render, and the plane depth does not depend on the SH degree. It is also why this task is safe to land after Task 13's parity gate: none of the three parity configs enables a PGSR loss.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_pgsr.py -q
```

Expected: all pass.

- [ ] **Step 7: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/pgsr.py tests/splats/test_pgsr.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/pgsr.py tests/splats/test_pgsr.py && \
  git commit --only collab_splats/splats/pgsr.py tests/splats/test_pgsr.py \
    -m "refactor(splats): pgsr constants to keyword arguments, add render_neighbor"
```

---

## Task 15: Docstring and comment sweep

The last pass over `collab_splats/splats/`. Every public function gets the codebase's docstring shape, every surviving magic number is either a keyword argument or has a comment saying why it is a constant, and every ported block cites its upstream.

This task changes no behavior. If a test fails here, you have edited code, not comments — revert and try again.

**Files:**
- Modify: every file in `collab_splats/splats/`

> **DEFERRED REVIEW FINDINGS — batch them into this sweep.**
>
> Each per-task code-quality review deferred its "Minor" findings here rather than growing the
> task that surfaced them. Every item below was measured against the tree at the commit named.
> Line numbers are from that commit and will have drifted — re-grep for the symbol.
>
> **All 18 findings were re-audited on 2026-09-06 at `8f4bfc37`** (after Task 10 landed): 6 still
> valid as written, 9 valid but relocated, 1 stale premise, 2 already fixed. The corrections are
> inline below. Two findings (T6-9 and the `cfg`-annotation one) live in files Task 10 rewrote —
> **re-grep those two against the current tip, not against `8f4bfc37`.**
>
> **CORRECTED 2026-09-06 at `0d889a29` — the "re-grep two of them" instruction is now far too narrow.
> FIVE of the six files have been rewritten since that audit.** Tasks 12 and 14 and two fix rounds
> landed in between:
>
> | file | change since `8f4bfc37` | findings anchored here |
> |---|---|---|
> | `losses.py` | **unchanged** | 11 |
> | `cameras.py` | **unchanged** | 1 |
> | `gaussian.py` | 28+/3− | 7 |
> | `trainer.py` | 10+/6− | 4 |
> | `pgsr.py` | 103+/59− | 1 |
> | `scaffold.py` | 149+/108− | 1 |
> | `rendering.py` | 68+/26− | — |
>
> (Commits: `aaadb0e1`, `f07b6e92`, `4306078a`, `348d879b`, `99f309cb`, `378a9e91`, `025ca70f`,
> `7feed636`.)
>
> **So the audit splits cleanly, and only half of it needs redoing:**
>
> - Findings anchored in **`losses.py` and `cameras.py` — the large majority — stand exactly as
>   audited**, line numbers included. Those two files have not been touched since. Do not re-derive
>   them; that is wasted work and every re-derivation is a chance to introduce an error.
> - Findings anchored in **`gaussian.py`, `trainer.py`, `pgsr.py`, `scaffold.py` and `rendering.py`
>   must be re-verified at your dispatch HEAD** — not merely re-grepped for a new line number, but
>   **re-checked for whether they still exist at all**. The `8f4bfc37` audit itself found 2 of 18
>   already fixed and 1 with a stale premise; three tasks have landed since, so expect more of both.
>   A finding that a later task already closed must be reported as closed, not re-closed.
>
> **This entry will go stale again.** The rule, not the table, is what to carry: *a re-audit is
> stamped with the tree it was taken at, and every commit to that tree expires part of it.* Before
> acting on any finding here, check whether its file has moved since the stamp — one
> `git diff --numstat <stamp> HEAD -- <file>` per file, which is cheaper than being wrong.
>
> *From the Task 6 review, measured at `6bd45bb5`:*
>
> - `losses.py:36-41, 59-64` — `default_losses` and `validate_schedule` use `Args:`/`Returns:`
>   where most of the file uses bullets. `Args:`/`Returns:` is common repo-wide (141/120
>   occurrences), so this is locally inconsistent, not wrong. Pick one and make the file
>   consistent.
>
>   **Corrected 2026-09-06.** The original text said "the only two of ten functions"; at
>   `8f4bfc37` it is **four of fifteen** — `rescale_depth_units` (`:153`/`:157`) and
>   `neighbor_selection` (`:181`/`:184`), both added by Task 10, use the same style. **Do not
>   chase the count** — it drifts every time a task touches this file. The rule is what matters:
>   one docstring style per file, applied to whatever is in `losses.py` when you get here.
> - `losses.py:63-64` — `Returns:` / `None. Raises ValueError...` on a `-> None` function.
>   Documenting a `None` return is padding, and the raise behavior is smuggled into the wrong
>   section. Use a `Raises:` section or a bullet.
> - `losses.py:40` — "A fresh `{name: spec}` mapping the caller owns and may mutate." Nothing
>   exercises that affordance; the one mutation site copies defensively anyway, because the
>   schedule flowing through `train()` may be the user's yaml dict rather than this function's
>   return. True, but reads as a general schedule contract that it is not.
> - `losses.py:67, 71, 177` forward-reference `OPTIONAL_LOSSES` / `LOSS_SPEC_KEYS`, defined at
>   423/438. The 350-line jump is mandated for `OPTIONAL_LOSSES` (it must follow the functions it
>   names). `LOSS_SPEC_KEYS` has no such dependency — it is a pure string table whose only consumer
>   is `validate_schedule` — and can sit directly above it at zero cost, halving the forward
>   references.
> - `losses.py:88` vs `:103` — the two primitive guards gate differently. Distortion is
>   weight-gated (`distortion_weight > 0`); `depth_ratio` is not. Measured:
>   `{"distortion": {"weight": 0.0}}` on 3dgs passes (there is a test for it), while
>   `{"normal_consistency": {"weight": 0.0, "depth_ratio": 0.5}}` on 3dgs raises even though the
>   loss is switched off. Pre-existing behavior, moved verbatim — but the extraction put the two
>   guards 15 lines apart in one function, which is where the inconsistency became visible.
>   **Comment it; do not change it.** The error strings are frozen (Ground Rules §7).
> - `losses.py:99` — the block comment describes the check two lines below it, not the one directly
>   under it. "Median depth only exists for 2DGS, so a non-zero blend on 3dgs is a config error"
>   sits above the `[0, 1]` range check at `:101`, which has nothing to do with 2DGS; it describes
>   `:103`. Pre-existing placement.
> - `losses.py:27-29` — the `# Schedule` divider now spans `compute_losses`. The parent's
>   `# Weighted sum` divider was dropped in Task 6 and its three functions folded under
>   `# Schedule`. `loss_weight`/`loss_active` are schedule logic; `compute_losses` is the evaluator
>   over the registry. A second divider before `:144` restores the distinction the file had.
> - ~~`losses.py:92` promises "reject bools and non-numbers (a quoted '0.6' too)" — only the bool
>   is tested.~~ **STRUCK 2026-09-06 — this finding was FALSE WHEN WRITTEN. Do not act on it, and
>   do not re-raise it.**
>
>   The coverage exists and predates the review that reported it missing: commit `f1443802` added
>   `@pytest.mark.parametrize("depth_ratio", [True, False, None, "0.6", [0.6]])`, and `f1443802`
>   is an **ancestor of `6bd45bb5`**, the commit the finding was measured at. The reviewer searched
>   `test_losses.py`; the coverage lives in `tests/splats/test_trainer.py` — at **`:216`** as of
>   `aaadb0e1`, re-measured. Acting on this finding would mean *adding* a duplicate of a test that
>   already exists, and — via T6-9 below — deleting the original.
> - `tests/splats/test_trainer.py` — `test_config_rejects_invalid` and
>   `test_config_rejects_bad_decay` duplicate tests Task 6 added to `test_losses.py`.
>   **Do not simply delete them** — two decay operands (`weight: 0.0` and `end_weight: 0.0`, in
>   `test_config_rejects_bad_decay`'s parametrize list) are covered ONLY there. Move those two into
>   `test_losses.py`, then reduce the trainer side to a single delegation smoke test.
>
>   **Corrected 2026-09-06. Task 10 rewrote this file — the original line refs (`:39-71`, `:60`,
>   `:61`) are dead. Match by symbol name, not by line.**
>
>   **TRAP — read this before you touch the file.** A naive reading of "reduce the trainer side to
>   a smoke test" also deletes the `depth_ratio` parametrize at **`:216`**
>   (`[True, False, None, "0.6", [0.6]]`, re-measured at `aaadb0e1`), which is the ONLY coverage of
>   the string and list paths anywhere in the suite — see the struck finding above. That block is
>   not duplication and must survive this sweep.
> - **`losses.py`'s RaDe-GS citation is factually wrong, and the true source has now been found.
>   Task 15 should REWRITE the claim, not hedge it and not pin a SHA to it.** Two independent
>   investigations agree (the Task 6 fix round, then the Task 6 re-review reproducing it):
>
>   **What upstream does not do.** `BaowenZ/RaDe-GS` at HEAD `d72f2079` — 29 commits across all
>   branches (the re-review's count; the fix round said 27 because `rtk`'s git-log filter drops merge
>   commits) — has **never contained the string "median" in any commit** of `scene/gaussian_model.py`
>   or `utils/graphics_utils.py`. Its only `.max()` refill is `scene/gaussian_model.py:229`,
>   `distance[~valid_points] = distance[valid_points].max()`, inside `compute_3D_filter`: per-Gaussian
>   3D-filter distances for Gaussians no camera sees. Different quantity, different shape, different
>   purpose — and copied verbatim from GOF, which is the likely route by which it got misread.
>   Upstream's real normal-consistency path (`train.py:166-170`) feeds `expected_depth` and
>   `median_depth` straight into `depth_double_to_normal` with no background refill and no alpha
>   scaling.
>
>   **What the claim was actually describing — our own retired code**, at
>   `collab_splats/nerfstudio/models/rade_gs.py:251-257` (deleted by Task 1; readable from a sibling
>   worktree or from git history):
>
>   ```python
>   expected_depths = torch.where(alpha > 0, expected_depths, expected_depths.detach().max())
>   median_depths   = torch.where(alpha > 0, median_depths,   median_depths.detach().max())
>   ```
>
>   **And even there the "before differencing" ordering is false.**
>   `depth_double_to_normal(camera, expected_depths, median_depths)` runs at `:209-211`, roughly 40
>   lines *before* the refill at `:251-257`. The refill is post-hoc output masking and never fed the
>   normal-consistency term at all.
>
>   So the docstring is wrong on the repo, wrong on the file, and wrong on the ordering. Do not pin
>   `d72f2079` — that would attach a precise citation to a claim the SHA disproves, which is why the
>   Task 6 fix round deliberately left `@ main` in place. Write what the code does instead, and note
>   that this is a **known-false claim currently shipping**: deleting the duplicate parenthetical
>   tightened the sentence around it, so it now reads more confidently than before. If Task 15 slips,
>   delete the clause rather than leave it standing. Every sibling citation in this same file is
>   precise (`gssr/scene/scaffold_scene.py:184`, `gssr/scene/scaffold_2dgs_scene.py:25`,
>   `gssr/scene/pgsr_scene.py:52-70`) and CLAUDE.md's rule is repo + commit + file + line, so this one
>   is the outlier twice over.
>
> *Earlier tasks:* the Task 1-5 reviews also deferred Minor findings here, but their text was not
> folded into this plan at the time. The final whole-implementation review (after Task 19) is the
> backstop — have it re-derive anything this list misses.
>
> ~~Also in scope for this sweep, found during Task 7: `collab_splats/splats/__init__.py` still
> carries a provenance comment naming `splats.zarr`, which Task 9 retires.~~
> **STRUCK 2026-09-06 — ALREADY FIXED, and the target no longer exists.** Task 10 rewrote that line
> to `# The gsplat commit pinned in pyproject.toml; asserted against pyproject by the cu121
> migration test`. The string `splats.zarr` appears nowhere in `collab_splats/splats/` (only in a
> stale `__pycache__/outputs.pyc`), so an implementer following this would hunt for a comment that
> is gone. It also credited the retirement to "Task 9", which was absorbed into Task 10 at
> `1bdadecd`.
>
> **From Task 7's code-quality review (measured at `235c954e`; the tip has moved, so re-grep every
> line number below before acting on it).** All are `gaussian.py` / `test_gaussian.py`; the review's
> Blocker and its three Important findings were fixed in a scoped round and are NOT in this list.
>
> - `gaussian.py:271-272` and `:289-290` — `Returns:` / `None.` is padding on two `-> None` methods.
>   `denormalize`'s at `:309-310` earns its line, because it says the mutation is in place; keep that
>   one.
> - `gaussian.py:323-327` — "Unused here, as above" four consecutive times where one line covers it.
>   Same block, and the more useful half: `cam_to_world` is declared `(N, 4, 4)` here but `(1, 4, 4)`
>   at `:234` — one parameter name, two shapes, one class. State "batched, unlike `render`'s single
>   view" explicitly; it is an easy trap for anyone writing against this interface.
> - `gaussian.py:33` and `:103` — `cfg` lost the `SplatsConfig` annotation it carries in
>   `trainer.py`. The circular-import reason is documented at `:132-133` and is correct for the
>   post-Task-10 state, but `if TYPE_CHECKING: from ... import SplatsConfig` plus
>   `cfg: "SplatsConfig"` restores the type with no runtime import.
>
>   **Corrected and extended 2026-09-06 (re-measured at `aaadb0e1`):**
>   - `trainer.py:280` is **dead** — the annotation the finding points at is now `trainer.py:183`.
>   - `gaussian.py`'s two sites are now **`:40`** and **`:110`**.
>   - **NEWLY UNCOVERED, not in the original finding: `Scaffold.__init__` has the same bare `cfg`,
>     at `scaffold.py:270`.** (The `cfg:` at `:283` is the docstring's `Args:` entry, not the
>     annotation — fix the signature.) Apply the same `TYPE_CHECKING` treatment there, so the sweep
>     leaves both model classes consistent rather than only one.
> - `gaussian.py:1-7` — module docstring is a prose paragraph; CLAUDE.md asks for bullets after the
>   summary.
> - `gaussian.py:96-98` — the class docstring's member list omits `activate`, which is public. The
>   omission is **correct** (Scaffold decodes to post-activation RGB and must not have it) but
>   unstated, which made "match this member list exactly" ambiguous for Task 8. One clause fixes it.
> - `tests/splats/test_gaussian.py:26-32` — `_model()` hardcodes `scene_scale=2.0` / `n_views=4`,
>   so callers hand-roll construction instead of using it. Accept them as kwargs.
>
>   **Extended 2026-09-06 (measured at `aaadb0e1`): there is a SECOND helper with the same defect
>   that the original finding missed** — `_model_with()` at `tests/splats/test_gaussian.py:35-41`,
>   which likewise pins `scene_scale=2.0` / `n_views=4` and differs from `_model()` only in
>   forwarding `**kwargs` to `Gaussians.__init__` instead of to `SplatsConfig.from_dict`. Fix both,
>   or better, collapse them into one helper that forwards to each. The hand-rolled call sites are
>   now at **`:62`** and **`:95`**.
> - `gaussian.py:359-373` (`from_checkpoint`) — `cls.__new__(cls)` sets all 11 attributes by hand.
>   They agree with `__init__` exactly today (measured: `MISSING: [] EXTRA: []`), but nothing asserts
>   it. One line in `test_gaussians_checkpoint_round_trips` —
>   `assert set(vars(restored)) == set(vars(model))` — stops a future `__init__` attribute from
>   silently missing the checkpoint path.

> **CORRECTED 2026-09-06 — five measured errors in this task's own steps, taken at `428e95a8`
> before Task 15 was dispatched. Steps 2 and 3 are wrong as written and following them literally
> produces busywork on already-correct files while missing the one real citation defect.**
>
> **1. Step 2's expected output is FALSE.** It says "exactly one line — `gaussian.py:SH_C0`".
> Measured by AST over `collab_splats/splats/*.py`, there are **seven** module-level uppercase
> assignments:
>
> | name | file:line | disposition |
> |---|---|---|
> | `GSPLAT_COMMIT` | `__init__.py:6` | keep — provenance string, asserted against `pyproject.toml` by the cu121 migration test |
> | `SH_C0` | `gaussian.py:28` | keep — mathematical constant, as the step says |
> | `OPTIONAL_LOSSES` | `losses.py:471` | keep — registry; must follow the functions it names |
> | `LOSS_SPEC_KEYS` | `losses.py:486` | keep — pure string table (the deferred finding asks only that it MOVE, above `validate_schedule`) |
> | `PRIMITIVES` | `trainer.py:62` | keep — registry read by Task 10's no-branch meta-test |
> | `MODEL_CLASSES` | `trainer.py:66` | keep — registry, same |
> | `REPRESENTATIONS` | `trainer.py:67` | keep — registry, same |
>
> **None of the six extra names is a tunable.** Ground Rule 9 is about tunables, not about
> uppercase. Do not "move" a registry to a keyword argument — `PRIMITIVES` / `MODEL_CLASSES` /
> `REPRESENTATIONS` are exactly what `test_trainer.py`'s no-branch guard enumerates, and relocating
> them would break the guard this branch spent five fix rounds giving teeth. Step 2's real content
> is its SECOND grep, the in-body literals.
>
> **2. Step 3 says "each of the seven files"; there are NINE.**
> `__init__.py`, `cameras.py`, `gaussian.py`, `losses.py`, `pgsr.py`, `rendering.py`,
> `scaffold.py`, `trainer.py`, `utils.py`.
>
> **3. Step 3's grep pattern gives three FALSE ALARMS.** Run as written it prints **zero** hits for
> `rendering.py`, `scaffold.py` and `__init__.py` — yet a broader provenance scan finds **50**
> upstream references in `scaffold.py` and **17** in `rendering.py`. The pattern requires an `@`, and
> those files spell their citations without one (`city-super/Scaffold-GS scene/gaussian_model.py`,
> `GS-SR gssr/gaussian/scaffold_gaussian.py:710`). **Do not add citations to those files on the
> strength of a zero count** — they are the two most densely cited files in the package.
>
> **4. The real citation defect, which Step 3's grep structurally cannot see.** CLAUDE.md's rule is
> repo **+ commit** + file + line. Measured against that rule:
>
> - `scaffold.py` — ~15 distinct upstream site citations, **not one of which carries a commit**
>   (e.g. `:65`, `:400`, `:465`, `:810`, `:848`, `:857`, `:927`, `:1046`, `:1065`, `:1105`).
> - `pgsr.py:338` and `:559` — `GS-SR @ main`. `main` is a moving ref, not a commit.
> - `losses.py:274` — `BaowenZ/RaDe-GS @ main`, and this one is additionally the known-false claim
>   the deferred-findings block above says to REWRITE rather than pin.
> - `cameras.py:5` is the only citation in the package in the house form (repo + commit + file +
>   lines).
>
> That gap — a whole file's worth of provenance with no commit component — is this step's actual
> work. **Note that `scaffold.py` is being rewritten by Task 12 right now, so re-grep it at your own
> HEAD; every line number above will have moved.**
>
> **5. Step 3's `cameras.py` cross-check is NOT PERFORMABLE OFFLINE, and must not be faked.** The
> step asks to verify that `examples/utils.py` lines 132-153 and 27-63 still describe
> `CameraOptModule` and `rotation_6d_to_matrix` at `d2f5c0f`. The installed gsplat **is** exactly
> that commit — `direct_url.json` reads `d2f5c0f8eb12190469f92cb408cf033943432532` — but the wheel
> **does not ship `examples/`**, and no gsplat source checkout exists on this box (searched
> `/root/.cache/uv`, `/opt`, `/workspace`). So the line numbers cannot be checked without network
> access. **Report it as unverified rather than asserting it verified.**
>
> **PASSES as written:** Step 3's final check. `grep -rn "\.worktrees/" collab_splats/splats/`
> returns nothing — there are no stale worktree citations left.

- [ ] **Step 1: Find the docstrings that do not match the house style**

The style, from `CLAUDE.md`: `"""` on their own lines, summary on the line after the opening quotes, a blank line before the bullets or `Args:`, no restating the function name, no padding.

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/python - <<'PY'
import ast
from pathlib import Path

for path in sorted(Path("collab_splats/splats").glob("*.py")):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_") and not node.name.startswith("__"):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            print(f"{path}:{node.lineno}: MISSING docstring on {node.name}")
            continue
        lines = doc.split("\n")
        if lines[0].strip():
            print(f"{path}:{node.lineno}: summary on the opening line in {node.name}")
        if len(lines) > 2 and lines[2].strip() and not lines[1].strip():
            pass
        elif len(lines) > 2 and lines[2].strip():
            print(f"{path}:{node.lineno}: no blank line after the summary in {node.name}")
        if node.name.replace("_", " ") in lines[1].lower():
            print(f"{path}:{node.lineno}: restates its own name in {node.name}")
PY
```

Fix every line this prints. The check is a guide, not a gate — it cannot see padding or a summary that says nothing.

- [ ] **Step 2: Find the remaining magic numbers**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -n "^[A-Z][A-Z0-9_]* = " collab_splats/splats/*.py
```

Expected: exactly one line — `collab_splats/splats/gaussian.py:SH_C0 = 0.5 / math.sqrt(math.pi)`. `SH_C0` survives because it is a mathematical constant (the zeroth-order spherical-harmonic basis value), not a tunable. Anything else this prints is a literal that Ground Rules §9 says should be a keyword argument; move it.

Then the in-body literals:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -n "1e-[0-9]\|0\.[0-9][0-9]*\b" collab_splats/splats/*.py | grep -v "def \|#\|\"\"\"" | head -40
```

Every hit must be one of: a keyword argument's default (fine), a value inside a comment-explained expression (fine), or a bare literal in the middle of an expression (not fine — either name it in a comment on the line above, or lift it to a keyword argument).

- [ ] **Step 3: Verify every ported block cites its upstream**

The rule from `CLAUDE.md`: vendored or reimplemented code cites repo + commit + file + line at the site.

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -n "gsplat @\|GS-SR @\|Scaffold-GS @\|PGSR @\|RaDe-GS @\|splatfacto" collab_splats/splats/*.py
```

Each of the seven files should print at least one citation. Cross-check the two known-stale ones:
- `losses.py`'s RaDe-GS citation was fixed in Task 6; confirm it no longer says `.worktrees/streaming`.
- `cameras.py`'s gsplat citation points at `examples/utils.py` lines 27-63 and 132-153 — verify those line numbers still describe the upstream `CameraOptModule` (whose body is now the pose half of `CameraOpt`) and `rotation_6d_to_matrix` at commit `d2f5c0f`, and correct them if not.

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "\.worktrees/" collab_splats/splats/ || echo "NO STALE WORKTREE CITATIONS"
```

- [ ] **Step 4: Run the full splats suite**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats -q
```

Expected: identical to Task 14's result. A comment sweep that changes a test result changed code.

- [ ] **Step 5: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/splats/ && \
  /opt/venv/reconstruction/bin/isort collab_splats/splats/ && \
  git commit --only collab_splats/splats/ -m "docs(splats): docstring and citation sweep"
```

Note the scoped `black`/`isort` paths — Ground Rules §4 forbids a repo-wide run.

**Task 15 outcome — swept at `078ab74d`, spec review NOT APPROVED, fix round 1 in flight.**

The sweep landed as `078ab74d` and covers `collab_splats/splats/`. The spec review upheld four
findings, all of them documentation-only, and all dispatched to a fix round that owns
`gaussian.py`, `scaffold.py`, `losses.py` and `utils.py`:

- **F1 (substantive).** The sweep *inserted* a false claim. `gaussian.py`'s class docstring now reads
  ``Exposes the same members as ``Scaffold`` — `params`, `optimizers`, `schedulers`, `activate`,`` and
  `Scaffold` has no `activate` — verified independently: zero occurrences of the name in
  `scaffold.py`, `hasattr` False on `Scaffold` and True on `Gaussians`, and the two mirror docstrings
  now contradict each other. The fix drops the word and adds the clause T7-5 actually asked for
  (*"`activate` is `Gaussians`-only: `Scaffold` decodes to post-activation values"*), and re-checks
  every member list in both docstrings mechanically with `hasattr`.
- **F3.** `AnchorStrategy.refine` cites `gssr/gaussian/scaffold_gaussian.py:707-717`; the second half
  of the claim ("frees its accumulators") is at `:719-723`. **A range that stops short of the claim it
  supports is the same defect as a wrong range.** Corrected to `:707-723`.
- **F4.** `losses.py`'s `loss_weight` has a two-line run-on summary with no blank line after it —
  it still trips Task 15's *own* Step 1 checker.
- **F5.** Two `nerfstudio @ 50e0e3c` citations (in `gaussian.py` and `utils.py`) name a commit but no
  file. They must name `nerfstudio/data/datamanagers/full_images_datamanager.py`; 1.1.5's line numbers
  must not be copied into a `50e0e3c` citation.

> **A documentation commit is not exempt from this branch's defect class.** F1 is the eighth confirmed
> instance of "the remedy for instance N ships instance N+1 in itself" — and the first to appear in a
> pure docs commit. The tell was the same as always: prose that reads as coverage.

Three corrections the review made to the coordinator's own claims, all verified and accepted:

- **Never verify a pinned citation against anything but the pin.** The earlier "all seven cited files
  match `566359be` by blob hash" claim was worthless, and so was the correction that replaced it:
  there is **no GS-SR clone on this machine at all**, and `6a213ab` is upstream `HEAD` of
  `yanxian-ll/GS-SR`, not a local checkout. HEAD is not the pin, 5 of the 9 cited files differ there,
  and the fix round verified the citations **over the network at their pinned commits** — which is
  the only evidence that counts. Network access is available; use it.
- Task 15's **Files** list names no test file at all, and the package has **nine** `.py` files, not
  seven — so Step 3's "each of the seven files should print a citation" is miscounted.
- Task 15's own prose ("if a test fails you have edited code — revert") contradicts its deferred-
  findings block, which mandates code edits. The deferred block governs.

**The `--amend` incident.** The Task 15 implementer folded a last `pgsr.py` docstring hunk in with
`git commit --amend` after the coordinator had already committed on top. The amend rewrote **the
coordinator's** commit: `17bc12e4` (plan-only) became `257b8027` (plan + the implementer's `pgsr.py`
hunk, under the coordinator's `docs(plans):` subject). Nothing was lost — `17bc12e4` survives as a
live object on no branch — and it was deliberately **not** repaired, because rewriting history while
other agents hold dirty state in the shared worktree trades a cosmetic problem for a tree-swap.
`--only <path>` scopes *which paths*, not *which commit*, and chaining `git rev-parse HEAD &&
git commit --amend` gives zero protection: the chain runs both and never lets you read the answer
first. **The fix for a forgotten hunk is a second commit. Ground Rules now name `--amend` explicitly.**

Task 15 also correctly **deferred** the T6-9 `test_trainer.py` dedup — moving tests would have
confounded its own zero-delta gate — but the handoff was never written down and the finding was
orphaned. It is now restated as **T16-22**.

**Fix round 1 landed as `33a8cba7`** — `gaussian.py`, `losses.py`, `scaffold.py`, `utils.py`, +19/−6,
all four findings closed, AST-minus-docstrings reported identical per file at `HEAD~1` vs `HEAD`, and
both gates flat (`tests/splats` 368, wide 469, 0 skips, RC=0) measured in pinned trees rather than the
peer-contaminated worktree. Spec re-review in flight. Three further corrections it made, all
verified:

- **The dispatch brief's own prescribed check was unsound.** It required each member of the shared
  list to resolve via `hasattr` on **both classes**; run literally that fails on `params`,
  `optimizers` and `schedulers`, which are assigned in `__init__` and are invisible to a class-level
  `hasattr`. Replaced with a live-instance probe: 13/13 resolve on both, lists identical. *A check
  that reports false on true statements is worse than no check — it would have manufactured three
  more "false claims".*
- **The brief conflated two different citations.** It said both `nerfstudio @ 50e0e3c` sites cite
  `full_images_datamanager.py`; that holds only for `utils.py`. `gaussian.py:70` cites splatfacto's
  DefaultStrategy args at `nerfstudio/models/splatfacto.py:264-280`. Following the brief literally
  would have shipped a brand-new false citation — F1's exact defect.
- `utils.py` was not in the sweep commit at all (its citation predates `078ab74d`), and `utils.py:161`
  did name a module stem rather than no file.

The fix round also caught its **own** first draft: it wrote that `Scaffold` "has no raw parameters
left to activate", which is false — `Scaffold`'s `scaling` is raw log-space — and reworded to say the
activations happen per view inside `decode`.

**Task 15 — fix round 1's re-review returned NOT APPROVED; round 2 follows.** The re-review verified
the mechanical claims completely — **AST-minus-docstrings identical for all four files**
(`gaussian.py` 14/14, `losses.py` 16/16, `scaffold.py` 30/30, `utils.py` 8/8 docstrings; full AST
differs, stripped AST equal), scope exactly the four paths at +19/−6, `black`/`isort` clean, zero
lines over 120 — and **fetched every citation over the network at its pin**: `:707-723` lands on
GS-SR's `elif step == densify_until_iter: del … empty_cache()` where the old `:707-717` stopped inside
the `adjust_anchor(` call, both nerfstudio sites land, and every *other* `file:line` in the four files
lands too (gsplat `d2f5c0f`, GS-SR `566359be`, Scaffold-GS `59c833b5`, RaDe-GS `d72f2079`). All five
of the fix round's self-corrections are TRUE.

But F5's own edit shipped **instance N+1**, and it is the same shape as F1: *a documentation pass made
a statement newly false by sharpening the citation next to it.*

`utils.py`'s `view_order` now cites nerfstudio's `full_images_datamanager.py:396-399`, the
`pop(0)` that drains and refills the shuffled epoch. Our implementation does
`rng.shuffle(order); yield from reversed(order)` — the **back** of the list. Measured, seed 42, 15
draws:

| n_views | ours | upstream `pop(0)` | equal? | retired in-repo `ViewSampler` (`pop()` from the end) | equal? |
|---|---|---|---|---|---|
| 2 | `[0,1,0,1,1,0,0,1,0,1,0,1,0,1,0]` | `[1,0,1,0,0,1,1,0,1,0,1,0,1,0,1]` | **False** | same as ours | True |
| 5 | `[0,4,2,1,3,1,4,0,2,3,4,0,2,1,3]` | `[3,1,2,4,0,3,2,0,4,1,3,1,2,0,4]` | **False** | same as ours | True |
| 8 | `[1,0,5,2,7,6,4,3,1,5,6,4,0,2,7]` | `[3,4,6,7,2,5,0,1,3,7,2,0,4,6,5]` | **False** | same as ours | True |

The adjacent sentence — *"Each epoch is yielded in reverse shuffle order, which is what the
shuffle-and-pop sampler this replaces produced — the sequences are identical seed for seed"* — was
safe while the citation was vague. With upstream's front-pop now named two lines below it, that
sentence reads as a claim about the just-cited upstream sampler and is **false for every
`n_views > 1`**. The referent it actually means is the in-repo `ViewSampler` deleted from the tree
(`git show 07167784^:collab_splats/splats/trainer.py`), and `tests/splats/test_utils.py:169-171`
already records the correct provenance.

Four soft findings, measured, not blocking: `gaussian.py`'s "there is no view-independent step to
expose" is contradicted by `scaffold.py:502` (`torch.exp(self.params["scaling"][anchor_ids])`, a
view-independent activation of a raw log-space param) — a weaker restatement of the very draft the
fix round says it caught; "`decode` returns rasterizer-ready values" overlooks that decode's dict
carries `log_scales` (deliberately log space) and `visible_ids` (an index tensor); the singled-out
`activate` exception reads as *the only* excluded member when Gaussians-only public members number 5
and Scaffold-only 9; and `AnchorStrategy.refine`'s window description omits upstream's outer
`step > start_stat` gate at line 710 (necessary-but-not-sufficient, not wrong about exclusivity).

Four corrections to the re-review brief, all accepted:

- **The control/treatment numbers the brief labelled "HEAD" were taken two commits back.** 368 / 469
  reproduce at `061fb4b3` (`HEAD~2`); at the true parent `a1fd1e46` they are **370 / 471**, and both
  sides of the A/B are flat there. The gate conclusion survives; the stated base did not.
- **The brief never defined the wide gate**, so 469 was unreproducible from it alone. It is
  `pytest tests/splats tests/mesh tests/test_cu121_migration.py`.
- **"all 26 `file:line` citations" is not reproducible under any convention** — 25 full `file.py:N`
  references, 28 counting the three bare continuations, 29 counting `adjust_anchor:703`.
- The fix round reported every claim it *listed* as TRUE, correctly. The one claim it did not list —
  `view_order`'s "identical seed for seed" — is the one it broke.


---

## Task 16: Reorganise the tests, then update docs and config

The code is done. This task makes `tests/splats/` mirror it, adds the one test that proves the interface actually exists, and updates the four documents that describe the module.

**Files:**
- Create: `tests/splats/test_model_interface.py`
- Modify: `tests/test_cu121_migration.py`
- Modify: `docs/splats.md`, `CLAUDE.md`, `configs/base.yaml`, `docs/superpowers/CHANGELOG.md`

> **CORRECTION PASS T16-1..T16-10 — controller pre-verification of Task 16, measured at
> `248473c4` (HEAD). Ten findings. Two are index-safety defects in Step 10, one is arithmetic in
> Step 7's frozen-surface proof, and one records a measurement that has NOT drifted and must not be
> re-taken.**
>
> **T16-1 (index safety, BLOCKING). Step 10's `git add -f` is unnecessary and is the exact operation
> the Ground Rules forbid.** All six paths Step 10 commits are TRACKED and none is ignored —
> `git ls-files --error-unmatch` exits 0 and `git check-ignore` exits 1 for each of
> `docs/splats.md`, `docs/superpowers/CHANGELOG.md`, `CLAUDE.md`, `configs/base.yaml`,
> `tests/test_cu121_migration.py` and `tests/splats/test_model_interface.py` (the last is new but
> `--only` takes an intent-to-add path, or add it alone). `git commit --only <paths>` commits a
> tracked file's worktree content **without staging anything**; `git add` writes the index shared
> across every worktree and every concurrent session, sweeping other sessions' work into your
> commit. `add -f` is only ever needed for a *new* file under a gitignored directory —
> `docs/superpowers/CHANGELOG.md` is already tracked. **Delete the `git add -f` line.**
>
> **T16-2 (BLOCKING). Step 10's commit path list is incomplete for its own corrections.** C16-2
> renames `_field` in `tests/splats/test_scaffold.py` and requires C12-7 — which lives in this plan
> file — to be edited *in the same commit*. Neither path is in Step 10's list. C16-1 adds a
> keyword-only refusal test per §9 boundary, and those land in the per-module test files, none of
> which is in the list either. Left as written, the rename and the plan edit stay uncommitted, which
> is how a later bare `-a` sweeps them. **Commit the paths you actually touched, in one `--only`
> invocation, and verify with `git show --stat`.**
>
> **T16-3. C16-1's sweep is FIVE tests, not "every signature" — measured.** AST over
> `collab_splats/splats/*.py`: **15** functions carry a keyword-only boundary, **24** keyword-only
> arguments in total. **Ten already have a refusal test**, so the remaining work is exactly five:
>
> | boundary | keyword-only args | guarded by |
> |---|---|---|
> | `cameras.py:55` `CameraOpt.__init__` | `optimize_pose`, `optimize_appearance` | **nothing** |
> | `cameras.py:86` `from_config` | `weight_decay` | **nothing** |
> | `losses.py:193` `compute_losses` | `l1_weight`, `ssim_weight` | **nothing** (C16-1 measured 44 passed) |
> | `pgsr.py:405` `forward_backward_noise` | `min_depth` | **nothing** |
> | `scaffold.py:266` `Scaffold.__init__` | `lr_decay` | **nothing** (C16-1 said "presumed; verify" — confirmed unguarded) |
> | `gaussian.py:39` `make_strategy` | 3 | `test_gaussian.py:204` |
> | `gaussian.py:113` `Gaussians.__init__` | 3 | `test_gaussian.py:222` |
> | `pgsr.py:62` `plane_depth` | 1 | `test_pgsr.py:385` |
> | `pgsr.py:223` `project` | 1 | `test_pgsr.py:394` |
> | `pgsr.py:297` `select_near_views` | 3 | `test_pgsr.py:363` |
> | `rendering.py:300` `write_outputs` | 1 | `test_rendering.py:490` |
> | `scaffold.py:1032` `prune` | 1 | `test_scaffold.py:753` |
> | `trainer.py:183` `train` | 2 | `test_trainer.py:486` |
> | `utils.py:22` `compute_scene_scale` | 1 | `test_utils.py:39` |
> | `utils.py:152` `view_order` | 1 | `test_utils.py:184` |
>
> C16-1 names `compute_losses` and `Scaffold.__init__`; the two `cameras.py` boundaries and
> `forward_backward_noise` are new here. **Re-derive this table before writing — it moves whenever a
> signature does.**
>
> **T16-4. `match="positional"` does not distinguish what C16-1 assumes, and two existing tests pin
> nothing at all.** For a keyword-only parameter, passing it positionally *is* an arity error — there
> is no separate "keyword-only" message. Measured:
> `compute_scene_scale() takes 1 positional argument but 2 were given`. So `match="positional"`
> matches both the intended violation and any unrelated arity mistake in the same call. What gives
> the test teeth is passing **exactly** the right number of leading positional arguments so the extra
> one lands in the keyword-only slot, and pinning the **full count string** —
> `match="takes 3 positional arguments but 4 were given"` — which eight of the ten existing tests do.
> `test_utils.py:39` and `test_utils.py:184` pass **no `match=` at all** and would accept a `TypeError`
> raised for any reason. Follow the eight; fix the two while you are here.
>
> **T16-5. Step 7's "exact set match" arithmetic is wrong; its disposition is right.** `SplatsConfig`
> has **24** dataclass fields, not 23. `configs/base.yaml`'s `splats:` block spans **172-238** and
> declares **24** keys. The sets are **not** equal — symmetric difference 2, not 0: the yaml has
> `enabled` (no dataclass field) and the dataclass has `scaffold` (present in the yaml only as the
> commented excerpt at 180-191, which is deliberate). Every shared literal equals its dataclass
> default **except `losses`**, whose dataclass default is `None` and whose yaml value equals
> `default_losses("3dgs")` — which the correction already states separately and correctly. So the
> frozen surface is intact and Step 7's `IDENTICAL` check still passes (it compares parsed **values**,
> which is why comment-only edits are invisible to it), but "no orphan keys, no missing keys" is
> false as written and would send someone hunting a `scaffold:` key that is commented out on purpose.
>
> **T16-6. `configs/base.yaml:191`'s quoted text has drifted; the defect has not.** The line now
> reads `#   appearance_dim: 0         # Scaffold's per-image embedding; independent of
> appearance_opt`, not the "`0 = off`" the correction quotes. Do not grep for the quoted string.
> `ScaffoldConfig.appearance_dim`'s default is **32**, read off the dataclass rather than the source
> line, so showing `0` as the default is still wrong and still worth fixing.
>
> **T16-7. Nothing will enforce the `CLAUDE.md` size bar in this tree.** Step 6 says "a PreToolUse
> hook enforces it". `/workspace/collab-splats/.claude/hooks/` **does not exist** here, and this
> worktree's `CLAUDE.md` is the pre-migration 151-line version that does not even carry the paragraph
> describing the hook. Measured `wc -c CLAUDE.md` = **41745**, already past the 40 000 bar (see
> T19-6). Two consequences: an oversize write gets **no signal**, and fixing `:29`'s seven false
> clauses cannot bring the file under the bar on its own. Honor the bar, expect no enforcement, and
> do not treat the overage as this plan's regression — it is inherited from other subsystems' prose.
>
> **T16-8. `docs/splats.md` is 101 lines, and this plan says both 101 and 100.** The task-head block
> says 101, Step 5's inline correction says 100; measured **101**. The same paragraph says "three
> `##` sections" and then names four (Quickstart 10, Two axes 37, Outputs 77, Losses 90). Cosmetic —
> but all **six** false statements reproduce at exactly the lines claimed (58, 68, 79, 84, 85, 97),
> and the structural claim that there is no per-file table and no data-flow prose holds, so Step 5's
> "this is an insertion, not a rewrite" framing is correct.
>
> **T16-9. Step 4's stale header names a Python that does not exist on this box.**
> `tests/test_cu121_migration.py:4` reads `/opt/conda/envs/reconstruction/bin/python`; `/opt/conda`
> is absent (the interpreter is `/opt/venv/reconstruction/bin/python`). Step 4's replacement
> docstring drops the whole `Run:` block, so this is fixed incidentally — named here so it is not
> preserved out of caution. **The rest of Step 4's correction reproduces exactly at HEAD:** eight
> splats entries at `:124-131`, `collab_splats.splats.utils` the only one missing, neither `outputs`
> nor `appearance` present, `collab_splats.utils` at `:132` the unrelated top-level package.
>
> **T16-10. The twelve-member interface table has NOT drifted — do not re-measure it.** Re-derived by
> AST at HEAD, every line number in the correction's table reproduces: `params` 356, `optimizers` 360,
> `schedulers` 363, `pre_backward` 355, `post_backward` 368, `denormalize` 387 (all `trainer.py`);
> `render` `trainer.py:314` + `rendering.py:288`; `n_primitives` `trainer.py:265,377` +
> `rendering.py:386,408`; `export_gaussians` 349, `checkpoint` 355, `frame_report` 377,
> `primitive_unit` 403 (all `rendering.py`). Consumed set is **12**; the plan's declared lists hold
> **9**; the three missing are `frame_report`, `primitive_unit` and `n_primitives`, and all three are
> present on both classes (`primitive_unit` annotated at `gaussian.py:111` / `scaffold.py:264`,
> `frame_report` defined at `gaussian.py:289` / `scaffold.py:595`). The AST width-guard the
> correction prescribes is the right remedy and its `consumed <= declared` assertion holds once the
> three are added.

> **C16-2 (deferred here by the Task 8 re-review, measured at `5bf13f1f`). Rename
> `tests/splats/test_scaffold.py`'s `_field` helper, and update C12-7 in the same commit.**
>
> Task 8 renamed `AnchorField` to `Scaffold` and purged the `field` token everywhere else —
> `grep '\bfield\b'` over `scaffold.py` and `test_scaffold.py` both return zero hits. The helper
> `_field` survives only because the underscore makes it miss that word-boundary grep. It is the
> last stale name from the old class.
>
> It was **not** renamed at `5bf13f1f` on purpose: C12-7 pins the name and tells Task 12 "use
> `_field`; do not resurrect the deleted helper", and renaming it would have bought another review
> round on a commit whose whole point was one functional line. You reorganise these tests wholesale,
> so it costs nothing here. **C12-7 must be edited in the same commit** or the plan starts naming a
> helper that no longer exists — exactly the C12-7 failure this correction chain already paid for
> once.

> **C16-1. Ground Rule §9 turned dozens of literals into keyword-only defaults, and NOTHING in
> the package tests the keyword-only-ness.** This is a package-wide gap, found independently by two
> reviews:
>
> - Task 7's review: dropping the bare `*` from `make_strategy`'s signature, and separately from
>   `Gaussians.__init__`, each left the whole `test_gaussian.py` suite green. Task 7's fix round
>   closes those two.
> - Task 6's re-review: dropping the bare `*` from `compute_losses`' signature leaves **44 passed**.
>   The only production caller (`trainer.py:572`) passes no weights, so this one is a silent widening
>   rather than a break — but it is unguarded.
>
> `Scaffold.__init__`'s `*, lr_decay=0.01` is the same shape and presumed equally unguarded; verify.
> While reorganising the tests, sweep every §9 signature in the package and add one
> `pytest.raises(TypeError, match="positional")` per keyword-only boundary. §9 is a *stated* ground
> rule of this plan, so a rule with no test is a rule that silently stops holding.

> **CORRECTED 2026-09-06 — three measured errors in Step 1, taken at `37da9287`. Step 1's file
> does not collect as written, and the interface it declares is missing two of the twelve members
> its two consumers actually read.**
>
> **1. `@cuda` is used and never defined — the file raises `NameError` at collection.** Step 1's
> block imports `numpy`, `pytest` and `torch`, then decorates
> `test_model_render_returns_a_render_dict_and_an_info_dict` with `@cuda`. Nothing binds that name.
> Collection fails, **zero** tests run, and Step 2's "Expected: 26 passed" is unreachable. Add the
> module-level marker the other three test files already use:
>
> ```python
> cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")
> ```
>
> matching `tests/splats/test_rendering.py:29`, `test_scaffold.py:25`, `test_trainer.py:26`. This is
> the **same defect Task 14's Step 1 carried**; it was caught there and is repeated here.
> **CUDA is available on this box (NVIDIA A40), so the guarded test RUNS — it does not skip, and the
> suite's skip count must stay 0.**
>
> **2. `INTERFACE_ATTRIBUTES` + `INTERFACE_METHODS` omit `frame_report` and `primitive_unit`.**
> The declared interface is nine names (three attributes, six methods) plus `n_primitives` tested
> separately. Derived by AST from the two files that actually consume a model, the real surface is
> **twelve**:
>
> | member | read at |
> |---|---|
> | `params` | `trainer.py:356` |
> | `optimizers` | `trainer.py:360` |
> | `schedulers` | `trainer.py:363` |
> | `pre_backward` | `trainer.py:355` |
> | `post_backward` | `trainer.py:368` |
> | `denormalize` | `trainer.py:387` |
> | `render` | `trainer.py:314`, `rendering.py:288` |
> | `n_primitives` | `trainer.py:265,377`, `rendering.py:386,408` |
> | `export_gaussians` | `rendering.py:349` |
> | `checkpoint` | `rendering.py:355` |
> | **`frame_report`** | **`rendering.py:377` — MISSING from the plan's lists** |
> | **`primitive_unit`** | **`rendering.py:403` — MISSING from the plan's lists** |
>
> Both are present on both classes (`Gaussians.primitive_unit` at `gaussian.py:111`,
> `Scaffold.primitive_unit` at `scaffold.py:264`; `frame_report` is a public method on each), so
> adding them costs two list entries and four test cases. **`primitive_unit` is the sanctioned
> representation discriminator** — it is what lets `rendering.py` stay branch-free, which is the
> whole point of Task 10. An interface test that pins eleven of twelve members and drops that one is
> pinning the wrong eleven.
>
> **3. The two `parametrize` lists ARE the aggregate, and nothing guards their width.** This is
> instance 1 of the defect class this branch has now hit six times: *a `parametrize` over a set
> cannot guard the set's own width.* Adding a thirteenth member to `trainer.py`'s or
> `rendering.py`'s call surface without adding it to `INTERFACE_*` goes completely unnoticed —
> which is exactly how finding 2 above came to exist.
>
> The remedy has already been built once on this branch, for the no-branch meta-test: derive the
> expected set by AST from the consumers and assert the declared list covers it.
>
> ```python
> def _members_read_off_the_model(path):
>     """
>     Every attribute the given module reads off a local named ``model``.
>     """
>     tree = ast.parse(Path(path).read_text())
>     return {
>         node.attr
>         for node in ast.walk(tree)
>         if isinstance(node, ast.Attribute)
>         and isinstance(node.value, ast.Name)
>         and node.value.id == "model"
>     }
>
>
> def test_the_declared_interface_covers_everything_its_consumers_read():
>     consumed = _members_read_off_the_model(trainer.__file__) | _members_read_off_the_model(
>         rendering.__file__
>     )
>
>     assert consumed <= set(INTERFACE_ATTRIBUTES) | set(INTERFACE_METHODS)
> ```
>
> Add `frame_report`, `primitive_unit` and `n_primitives` to the declared lists so that assertion
> holds, and keep the per-name `parametrize` tests — the AST test guards the list's *width*, the
> parametrize guards each member's *presence on both classes*. Neither substitutes for the other.
> Step 2's expected count moves with the list; **re-derive it rather than repeating a number from
> this plan.**
>
> **Verified correct as written, at `37da9287` — do not re-measure these:**
> `_model`'s call `model_class(cfg, points, colors, scene_scale=1.0, n_views=4, device=device)`
> matches both `__init__` signatures (all six are positional-or-keyword; the keyword-only tail is
> `knn`/`adam_eps`/`lr_decay` on `Gaussians` and `lr_decay` on `Scaffold`).
> `from_checkpoint(ckpt, "cpu")` matches `(cls, ckpt, device)` on both.
> Step 4's correction still holds: `tests/test_cu121_migration.py:124-131` carries exactly eight
> splats entries, `collab_splats.splats.utils` is the only one missing, neither `outputs` nor
> `appearance` is present, and `collab_splats.utils` at `:132` is the unrelated top-level package.
> Step 5's correction still holds: `docs/splats.md` is 101 lines with one table at `:81-86` and no
> per-file table.
> Step 6's work is real: `CLAUDE.md:103` still reads
> `# Gaussian-splat training on upstream gsplat: cameras, losses, rendering, trainer, outputs` —
> note that the line Step 6 displays is the **replacement**, not the current text.

> **CORRECTION PASS T16-11..T16-21 — controller pre-verification, measured at `dae59f89`.**
> **These supersede T16-1..T16-10 and C16-1/C16-2 wherever they disagree.** Three are BLOCKING.
> Every test-side line number in T16-3's table had already drifted by the time this was written;
> navigate by grepping for the named symbol, never by line.
>
> The T16 block was measured at `248473c4`; C16-1 at `37da9287`; C16-2 at `5bf13f1f`. The branch has
> moved a long way since. I re-measured. Here is what changed and what did not.
>
> ### T16-11 (BLOCKING, restating T16-1). Delete Step 10's `git add -f`.
>
> Step 10 ends with:
>
> ```bash
> git add -f docs/splats.md docs/superpowers/CHANGELOG.md && \
> git commit --only ...
> ```
>
> All six paths are **tracked** and none is ignored. `git add` writes the shared index. `--only` alone
> does the whole job: it commits a tracked file's worktree content without staging anything. **Delete
> the `git add -f` line entirely.** For the one genuinely new file
> (`tests/splats/test_model_interface.py`) use `git add -N` (intent-to-add) on that path alone, or
> commit it in its own `--only` invocation.
>
> Step 10's path list is also incomplete for its own corrections (T16-2): it omits
> `tests/splats/test_scaffold.py` (the `_field` rename), the plan file itself, and the per-module test
> files that receive C16-1's keyword-only tests. **Commit the paths you actually touched.**
>
> ### T16-12 (BLOCKING). T16-10's "do NOT re-measure" is now VOID.
>
> T16-10 says the twelve-member interface table has not drifted and must not be re-measured. That was
> true at `248473c4`. **Task 15's block-comment sweep is landing on `collab_splats/splats/*.py` right
> now and shifts every line number in that file set.** The member *names* are stable; the *line
> numbers* are not.
>
> I re-derived the consumed interface at `dae59f89` and it is still exactly twelve, all present on
> both classes:
>
> ```
> checkpoint  denormalize  export_gaussians  frame_report  n_primitives  optimizers
> params      post_backward  pre_backward    primitive_unit  render      schedulers
> ```
>
> Re-derive it yourself with this script rather than trusting any line number:
>
> ```python
> """Every member trainer.py and rendering.py read off a local named `model`."""
> import ast, pathlib, collections
>
> W = pathlib.Path("/workspace/collab-splats/.worktrees/clean-splats")
> hits = collections.defaultdict(list)
> for name in ["trainer.py", "rendering.py"]:
>     p = W / "collab_splats/splats" / name
>     for node in ast.walk(ast.parse(p.read_text())):
>         if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
>                 and node.value.id == "model"):
>             hits[node.attr].append(f"{name}:{node.lineno}")
> for k in sorted(hits):
>     print(f"{k:20s} {', '.join(hits[k])}")
> ```
>
> C16-1's remedy stands: add `frame_report`, `primitive_unit` and `n_primitives` to the declared lists,
> **and** keep the AST width-guard test (`consumed <= declared`). The parametrize guards each member's
> *presence on both classes*; the AST test guards the *list's own width*. **Neither substitutes for the
> other** — a parametrize over a set cannot guard that set's width, which is precisely how
> `frame_report` and `primitive_unit` came to be missing in the first place.
>
> ### T16-13. Every test-side line number in T16-3's table has already drifted. All ten.
>
> The source-side citations still hold at `dae59f89`. The test-side ones do not — measured:
>
> | T16-3 said | actually |
> |---|---|
> | `test_gaussian.py:204` | `:208` |
> | `test_gaussian.py:222` | `:227` |
> | `test_pgsr.py:385` | `:415` |
> | `test_pgsr.py:394` | `:423` |
> | `test_pgsr.py:363` | `:393` |
> | `test_rendering.py:490` | `:495` |
> | `test_scaffold.py:753` | `:771` |
> | `test_trainer.py:486` | `:491` |
> | `test_utils.py:39` | `:44` |
> | `test_utils.py:184` | `:187` |
>
> Ten out of ten. **Treat every `file.py:NNN` in the plan as a hint about which construct is meant,
> never as an address.** Grep for the named symbol at your own HEAD. And note `tests/splats/` is under
> active edit by other agents as you work, so these will move again.
>
> ### T16-14. T16-3's *substance* holds: five unguarded boundaries, and they are the five it names.
>
> Re-derived at `dae59f89`: **15** keyword-only boundaries, **24** keyword-only args, **10** guarded,
> **5** unguarded. The five needing a new refusal test:
>
> | boundary | keyword-only args |
> |---|---|
> | `cameras.py` `CameraOpt.__init__` | `optimize_pose`, `optimize_appearance` |
> | `cameras.py` `CameraOpt.from_config` | `weight_decay` |
> | `losses.py` `compute_losses` | `l1_weight`, `ssim_weight` |
> | `pgsr.py` `forward_backward_noise` | `min_depth` |
> | `scaffold.py` `Scaffold.__init__` | `lr_decay` |
>
> Re-derive before writing — Task 15 moves the line numbers, and a signature change moves the set:
>
> ```python
> """Keyword-only boundaries in collab_splats/splats/ and whether a test guards each."""
> import ast, pathlib
>
> W = pathlib.Path("/workspace/collab-splats/.worktrees/clean-splats")
> SRC, TESTS = W / "collab_splats/splats", W / "tests/splats"
>
> # Resolve __init__ to its enclosing class -- that is the name the constructor is called by
> boundaries = []
> for path in sorted(SRC.glob("*.py")):
>     for parent in ast.walk(ast.parse(path.read_text())):
>         cls = parent.name if isinstance(parent, ast.ClassDef) else None
>         for node in ast.iter_child_nodes(parent):
>             if isinstance(node, ast.FunctionDef) and node.args.kwonlyargs:
>                 called = cls if (node.name == "__init__" and cls) else node.name
>                 label = f"{cls}.{node.name}" if cls else node.name
>                 boundaries.append((path.name, node.lineno, label, called,
>                                    [a.arg for a in node.args.kwonlyargs]))
>
> guarded = {}
> for path in sorted(TESTS.glob("test_*.py")):
>     text = path.read_text()
>     for node in ast.walk(ast.parse(text)):
>         if not isinstance(node, ast.With):
>             continue
>         src = ast.get_source_segment(text, node) or ""
>         if "raises" not in src or "TypeError" not in src:
>             continue
>         strong = "positional argument" in src   # the full count string, not a bare match="positional"
>         for *_, called, _ in boundaries:
>             if called in src:
>                 guarded.setdefault(called, []).append((path.name, node.lineno, strong))
>
> for fname, lineno, label, called, args in boundaries:
>     g = guarded.get(called)
>     where = (", ".join(f"{f}:{ln}{'' if s else '  <-- WEAK'}" for f, ln, s in g)
>              if g else "*** UNGUARDED ***")
>     print(f"{fname}:{lineno:<5} {label:26s} {args}\n{'':40s}{where}")
> ```
>
> **A false negative to know about:** resolving `__init__` to its class is load-bearing. Without it the
> script reports `Gaussians.__init__` and `Scaffold.__init__` as unguarded, because the constructor is
> never called by the name `__init__`. The version above handles it; if you rewrite the script, keep
> that.
>
> ### T16-15. T16-4's arithmetic is wrong: **four** existing tests are weak, not two.
>
> T16-4's *reasoning* is right and important — for a keyword-only parameter, passing it positionally
> **is** an arity error, so `match="positional"` matches both the intended violation and any unrelated
> arity mistake in the same call. What gives the test teeth is passing exactly the right number of
> leading positional args so the extra one lands in the keyword-only slot, **and pinning the full count
> string**.
>
> But T16-4 claims "eight of the ten existing tests do" pin the full string. Measured at `dae59f89`:
> **six** do. Four do not:
>
> - `test_gaussian.py` `test_make_strategy_tuning_literals_are_keyword_only` — `match="positional"`
> - `test_gaussian.py` `test_gaussians_init_tuning_literals_are_keyword_only` — `match="positional"`
> - `test_utils.py` `compute_scene_scale` guard — **no `match=` at all**
> - `test_utils.py` `test_view_order_seed_is_a_keyword_argument` — **no `match=` at all**
>
> The two with no `match=` accept a `TypeError` raised for any reason whatsoever. Compare the shape you
> want, from `test_pgsr.py`:
>
> ```python
> with pytest.raises(TypeError, match="takes 3 positional arguments but 4 were given"):
>     plane_depth(normal, distance, intrinsics, 1e-4)
> ```
>
> Strengthen all four while you are here, and write your five new tests in that shape.
>
> ### T16-16 (BLOCKING). C16-2 understates the `_field` rename by five-fold.
>
> C16-2 says rename `_field` in `tests/splats/test_scaffold.py` and "update C12-7 in the same commit".
> But `_field` is named in the plan at **16 lines across at least five distinct blocks**, not one:
>
> ```
> 3000, 3099, 3103, 3296     Task 8's correction block
> 5492, 5498, 5503, 5509     C12-7
> 5528                       C12-8
> 5696, 5721, 5786           the C12-19..C12-25 flake/vacuity blocks
> 7525, 7529, 7533           C16-2 itself
> ```
>
> Leaving any of them naming a helper that no longer exists is exactly the failure C16-2 says it exists
> to prevent. **Re-derive the list at your HEAD** (`_field` is also 56 occurrences in
> `test_scaffold.py`, and Task 12's fix round is editing that file right now, so both counts move).
>
> Update every site in the same commit as the rename, or **report BLOCKED and skip the rename
> entirely** — a partial rename is strictly worse than none. Skipping it is an acceptable outcome; say
> so plainly if you choose it.
>
> ### T16-17. Do NOT run Step 9's `graphify update .`.
>
> Step 9 says to run it. `graphify-out/` **does not exist in this worktree** — it exists only in the
> main checkout `/workspace/collab-splats/`, which you must not touch. Running it here would build a
> fresh graph inside the worktree, which is not what the step means and is not this task's job.
>
> **Skip Step 9 and report it as owed to whoever owns the main checkout.** This is already a known
> outstanding item on this effort; you are not introducing a gap.
>
> ### T16-18. The `CLAUDE.md` size bar is real but unenforced, and the file is already over it.
>
> Step 6 says "a PreToolUse hook enforces it". Measured: `.claude/hooks/` exists **neither** in this
> worktree **nor** in the main checkout. And `CLAUDE.md` is **41745 bytes / 150 lines** — already past
> the 40 000 bar.
>
> Consequences: an oversize write gets **no signal**, and fixing `:29`'s seven false clauses cannot
> bring the file under the bar on its own. **Honor the bar, expect no enforcement, make every edit
> net-neutral or shrinking, and do not treat the existing overage as this plan's regression** — it is
> inherited from other subsystems' prose.
>
> ### T16-19. Small measured drifts in the T16 block's own numbers.
>
> - `docs/splats.md` is **100 lines**, not the 101 T16-8 asserts (Step 5's "100" was right). It has
>   **four** `##` sections — Quickstart 10, Two axes 37, Outputs 77, Losses 90 — plus the `#` title,
>   and one table at `:81-86`. The structural claim holds: no per-file table, no data-flow prose, so
>   Step 5 is genuinely an **insertion**, not a rewrite.
> - The `configs/base.yaml` `splats:` block spans **172-239**, not the 172-238 T16-5 gives.
> - T16-5's set arithmetic is **confirmed exactly**: `SplatsConfig` has **24** dataclass fields, the
>   yaml declares **24** keys, and the symmetric difference is **2** — the yaml has `enabled` (no
>   dataclass field) and the dataclass has `scaffold` (present in the yaml only as the deliberately
>   commented excerpt). So Step 7's "no orphan keys, no missing keys" is **false as written** and would
>   send you hunting a `scaffold:` key that is commented out on purpose. The frozen surface is intact
>   and Step 7's `IDENTICAL` check still passes, because it compares parsed **values**.
> - T16-9 confirmed at `dae59f89`: `tests/test_cu121_migration.py` carries exactly **eight** splats
>   entries at `:124-131`, `collab_splats.splats.utils` is the only one missing, neither `outputs` nor
>   `appearance` is present, and `collab_splats.utils` at `:132` is the unrelated top-level package.
>
> ### T16-20. Step 1's test file does not collect as written.
>
> Step 1 decorates a test with `@cuda` and never binds the name — `NameError` at collection, **zero**
> tests run, and Step 2's expected count is unreachable. Add the module-level marker the other three
> test files already use:
>
> ```python
> cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")
> ```
>
> This is the same defect Task 14's Step 1 carried. **CUDA is available here, so the guarded test RUNS
> — the suite's skip count must stay 0.**
>
> Step 2's "Expected: 26 passed" moves once you add the three missing interface members and the AST
> width test. **Re-derive the count; do not repeat the plan's number.**
>
> ### T16-21. Step 3's grep must match the filename, not the substring.
>
> Already corrected in the plan, restated because it is easy to miss: grepping the bare string
> `test_appearance` matches two legitimate test *names* in `test_scaffold.py`
> (`test_appearance_embedding_changes_color_only_when_enabled` and
> `test_appearance_embedding_needs_view_count`) and reads as a Task 5 regression that is not real.
> Match `test_appearance\.py`.
>
>
> ### T16-22. T6-9 was deferred to this task by Task 15 and never written down — it is yours.
>
> Task 15's spec review found the handoff missing: Task 15 declined the `test_trainer.py` dedup on
> the correct grounds that moving tests moves the suite count and would have confounded its own
> zero-delta gate, and routed it here. Nothing in this task's body mentioned it, so the finding was
> orphaned. It is restated in full so it is not lost a second time.
>
> `tests/splats/test_trainer.py` — `test_config_rejects_invalid` and `test_config_rejects_bad_decay`
> duplicate tests Task 6 added to `test_losses.py`. **Do not simply delete them.** Two decay operands
> (`weight: 0.0` and `end_weight: 0.0`, in `test_config_rejects_bad_decay`'s parametrize list) are
> covered ONLY there. Move those two into `test_losses.py`, then reduce the trainer side to a single
> delegation smoke test.
>
> **Two traps, both already paid for once on this branch:**
>
> - Task 10 rewrote this file, so the original line refs are dead. **Match by symbol name.**
> - A naive reading of "reduce the trainer side to a smoke test" also deletes the `depth_ratio`
>   parametrize (`[True, False, None, "0.6", [0.6]]`), which is the **only** coverage of the string
>   and list paths anywhere in the suite. It is not duplication and must survive. A struck finding
>   earlier in this plan reported that coverage as missing; it was false when written, and acting on
>   it would have deleted the original while adding a duplicate.
>
> This moves the suite count. That is expected here, and is why Task 15 deferred it — re-derive the
> expected totals rather than reusing any earlier number.
> ---

- [ ] **Step 1: Write the interface test**

This is the test that replaces a base class. Both models are built on the same synthetic scene and asked for the same members; nothing else in the codebase asserts that duck typing holds.

Create `tests/splats/test_model_interface.py`:

```python
"""
Both model classes expose one interface, verified structurally rather than by inheritance.

`Gaussians` and `Scaffold` share no base class on purpose: they have no implementation in
common, and an abstract base with two implementations and no shared code is a file that only
exists to be read. What they do share is a call surface, and that is what this file pins.
"""

import numpy as np
import pytest
import torch

from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.scaffold import Scaffold
from collab_splats.splats.trainer import SplatsConfig

# The interface, in one place: three attributes the trainer reads and six methods it calls
INTERFACE_ATTRIBUTES = ["params", "optimizers", "schedulers"]
INTERFACE_METHODS = ["render", "pre_backward", "post_backward", "denormalize", "export_gaussians", "checkpoint"]


def _model(model_class, device="cpu"):
    """
    Build either model over the same 200-point seed cloud.

    - Defaults to the CPU; the render test passes `device="cuda"` because gsplat's
      rasterization kernels are registered for CUDA only.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, (200, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (200, 3)).astype(np.uint8)
    overrides = {"representation": "scaffold", "scaffold": {}} if model_class is Scaffold else {}
    cfg = SplatsConfig.from_dict({"max_steps": 100, **overrides})
    return model_class(cfg, points, colors, scene_scale=1.0, n_views=4, device=device)


@pytest.mark.parametrize("model_class", [Gaussians, Scaffold])
@pytest.mark.parametrize("attribute", INTERFACE_ATTRIBUTES)
def test_model_exposes_interface_attribute(model_class, attribute):
    model = _model(model_class)

    assert hasattr(model, attribute)


@pytest.mark.parametrize("model_class", [Gaussians, Scaffold])
@pytest.mark.parametrize("method", INTERFACE_METHODS)
def test_model_exposes_interface_method(model_class, method):
    model = _model(model_class)

    assert callable(getattr(model, method))


@pytest.mark.parametrize("model_class", [Gaussians, Scaffold])
def test_model_optimizers_and_schedulers_are_flat_lists(model_class):
    model = _model(model_class)

    # The trainer does `model.optimizers + refine.optimizers`, so these must be lists, not dicts
    assert isinstance(model.optimizers, list)
    assert isinstance(model.schedulers, list)
    assert all(hasattr(optimizer, "step") for optimizer in model.optimizers)
    assert all(hasattr(scheduler, "step") for scheduler in model.schedulers)


@pytest.mark.parametrize("model_class", [Gaussians, Scaffold])
def test_model_n_primitives_is_an_int(model_class):
    model = _model(model_class)

    assert isinstance(model.n_primitives, int)
    assert model.n_primitives > 0


@cuda
@pytest.mark.parametrize("model_class", [Gaussians, Scaffold])
def test_model_render_returns_a_render_dict_and_an_info_dict(model_class):
    model = _model(model_class, device="cuda")
    cam_to_world = torch.eye(4)[None].cuda()
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[[40.0, 0.0, 20.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]]).cuda()

    # The signature is render(..., width, height, ...) — width FIRST. 40x24 rather than a
    # square frame is what makes a swap fail here; rgb comes back (1, H, W, 3).
    render, info = model.render(cam_to_world, intrinsics, 40, 24, torch.tensor([0]).cuda(), step=0)

    assert set(render) >= {"rgb", "depth", "alpha"}
    assert render["rgb"].shape == (1, 24, 40, 3)
    assert isinstance(info, dict)


@pytest.mark.parametrize("model_class", [Gaussians, Scaffold])
def test_model_checkpoint_round_trips(model_class):
    model = _model(model_class)
    ckpt = model.checkpoint()
    ckpt["config"] = {"primitive": "3dgs", "representation": "scaffold", "scaffold": {}}

    restored = model_class.from_checkpoint(ckpt, "cpu")

    assert restored.n_primitives == model.n_primitives
    assert set(restored.params) == set(model.params)
```

- [ ] **Step 2: Run it to verify it passes**

Unlike every other task's test, this one should pass immediately — Tasks 7 and 8 built the interface it describes. If it fails, one of those tasks is incomplete, and this is where you find out.

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats/test_model_interface.py -q
```

Expected: 26 passed.

- [ ] **Step 3: Confirm `test_appearance.py` is already gone**

Task 5 deleted it alongside `appearance.py` — its assertions live on as `test_cameras.py`'s color-affine tests, and `AppearanceModule` no longer exists as a separate class. This step only proves nothing reintroduced it and nothing still points at it:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  test ! -e tests/splats/test_appearance.py && \
  ! grep -rln "test_appearance\.py" tests/ \
  && echo "ALREADY DELETED"
```

Expected: `ALREADY DELETED`.

> **Corrected 2026-09-06 (measured at `8f4bfc37`).** The original command grepped the bare string
> `test_appearance`, which matches two *legitimate test names* —
> `tests/splats/test_scaffold.py:90` `test_appearance_embedding_changes_color_only_when_enabled`
> and `:105` `test_appearance_embedding_needs_view_count`. It would have failed here and read as a
> Task 5 regression that is not real. Match the filename, not the substring.

If the file is still there, Task 5 Step 4 was skipped — go back and run it there rather than deleting it here, because `test_cameras.py`'s replacement tests were written in that same task.

- [ ] **Step 4: Update the migration test's module list**

In `tests/test_cu121_migration.py`, bring the `collab_splats.splats.*` entries to the nine modules
that exist now:

> **Corrected 2026-09-06 (measured at `8f4bfc37`).** This step used to say "replace the **five**
> entries" and "`outputs` and `appearance` come out". Both are stale: the list already carries
> **eight** entries (`splats`, `.cameras`, `.losses`, `.rendering`, `.trainer`, `.gaussian`,
> `.scaffold`, `.pgsr`, at `tests/test_cu121_migration.py:124-131`), all importing cleanly, and
> **neither `outputs` nor `appearance` is present** — Task 10 already took them out. The only
> module actually missing is **`collab_splats.splats.utils`**. (`collab_splats.utils` at `:132` is
> the unrelated top-level package — do not confuse them.) So the real work here is: add the one
> entry, alphabetise, replace the docstring.

```python
        "collab_splats.splats",
        "collab_splats.splats.cameras",
        "collab_splats.splats.gaussian",
        "collab_splats.splats.losses",
        "collab_splats.splats.pgsr",
        "collab_splats.splats.rendering",
        "collab_splats.splats.scaffold",
        "collab_splats.splats.trainer",
        "collab_splats.splats.utils",
```

Note the file's own header (`:6-8`) says the splats list was written ahead of the package and
"stays red until Task 4 lands" — stale by two plans. It should read:

```python
"""
Import and environment gate for the cu121 + uv migration: any failure blocks the merge.
"""
```

Verify:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/test_cu121_migration.py -q --continue-on-collection-errors
```

Expected: `test_import_all_modules` passes. Other tests in this file may fail on the known gsplat version skew (Ground Rules §3) — that is pre-existing and not yours to fix.

- [ ] **Step 5: Give `docs/splats.md` a module-layout section, and fix its six false statements**

> **Corrected 2026-09-06 (measured at `8f4bfc37`).** This step used to say "Rewrite the file table
> … Replace the table with:" and "replace the data-flow paragraph". **Neither exists.**
> `docs/splats.md` is 100 lines with three `##` sections (Quickstart 10, Two axes 37, Outputs 77,
> Losses 90) and exactly one table — the Outputs artifact table at 81-86. There is no per-file
> table and no data-flow prose. Confirmed by `git log -- docs/splats.md`: a single commit,
> `700cf4e9`. **This is an insertion.** Add a new `## Module layout` section between the intro
> (`:3-6`) and Quickstart (`:10`), holding the table and the paragraph below.

Insert this table:

| File | Contents |
|------|----------|
| `trainer.py` | `SplatsConfig` and `train()`: setup, loop, finish. No representation branches. |
| `gaussian.py` | `Gaussians` — vanilla 3DGS/2DGS parameters, `make_strategy`, `SH_C0`. |
| `scaffold.py` | `Scaffold` — Scaffold-GS anchors, the three MLP heads, `AnchorStrategy`. |
| `rendering.py` | `render_gaussians` (the one call into gsplat), plus `render_views` / `write_outputs` / `load_checkpoint`. |
| `losses.py` | the loss schedule and the optional-loss registry. |
| `cameras.py` | `CameraOpt` — per-camera pose deltas and per-image color affine, each half optional — and `rotation_6d_to_matrix`. |
| `pgsr.py` | PGSR planar geometry, neighbor selection, `render_neighbor`. |
| `utils.py` | scene normalization, coarse-to-fine downscaling, view order, target preparation. |

And replace the data-flow paragraph:

> `train()` builds one model — `Gaussians` or `Scaffold`, chosen by `representation` — and one
> `CameraOpt`. The two model classes expose the same members, so the loop is
> representation-agnostic; `CameraOpt` is one module whose two halves — pose delta and color
> affine — are selected independently by `pose_opt` / `appearance_opt` and pass their input
> through untouched when off, so the loop does not branch on those either: refine the camera,
> render, apply the color affine, composite over a random
> background, compute the losses, backward, step every optimizer and scheduler, then let the
> model densify itself. At the end the model and
> cameras are put back in the world frame and `write_outputs` writes `splats.ply`, `ckpt.pt` and
> `splats_quality_report.json`. The checkpoint is self-contained: `load_checkpoint` plus
> `render_views` reproduce every training-view render, which is how the mesh stage and the eval
> scripts consume the model.

Then fix the **six measured false statements** in the existing prose (all at `8f4bfc37`):

| line | false claim | truth |
|---|---|---|
| 79 | "`train` writes four files into `out_dir`" | writes **three** (`rendering.py:287-369`) |
| 85 | the whole `splats.zarr` artifact-table row | the artifact is retired — delete the row |
| 84 | "`ckpt.pt` \| parameters, **pose refiner**, appearance module, and — under `scaffold` — the MLP heads and voxel size" | it holds `splats`, `config`, `cam_to_world`, `intrinsics`, `image_ids`, `image_size`, `appearance` (`rendering.py:332-341`). The **pose refiner is NOT stored** — its deltas are folded into `cam_to_world` (`rendering.py:304-305`) — and "appearance module" is now the color half of one `CameraOpt` |
| 97 | "Registered: `depth`, `normal_consistency`, `distortion`, `opacity_reg`, `scale_reg`, `appearance_reg`" | `OPTIONAL_LOSSES` (`losses.py:471-480`) has **eight** — add `pgsr_normal` and `pgsr_multiview`. `pgsr.py` is not mentioned anywhere in the doc |
| 68 (and the yaml snippet at 60-69) | `appearance_dim: 0` presented as the shipped default | `ScaffoldConfig.appearance_dim` defaults to **32** (`scaffold.py:69`). "0 = off" is correct; the implied default is not. Same defect at `configs/base.yaml:191` |
| 58 | "`sh_degree` is ignored — color comes from `mlp_color`" | a *deliberate* scaffold override of `sh_degree`/`sh_degree_interval` now **raises** `ValueError` (`trainer.py:155-162`); only values left at their defaults are ignored |

Accurate, leave alone: `:3-6` intro, `:14-33` quickstart (matches `train()` at `trainer.py:180-193`),
`:41-48` primitive axis, `:71-73` two-appearance-models paragraph, `:92-95` schedule semantics,
`:98-100` per-primitive defaults.

Then confirm no `splats.zarr` mention survives:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -n "splats.zarr" docs/splats.md || echo "NONE LEFT"
```

- [ ] **Step 6: Update the `CLAUDE.md` architecture tree**

The tree line at `CLAUDE.md:103`:

```
  splats/                  # gsplat training: trainer, gaussian, scaffold, losses, rendering, cameras, pgsr, utils
```

> **Corrected 2026-09-06:** the replacement line used to omit `pgsr`. All eight modules besides
> `__init__.py` belong on it.

**`CLAUDE.md:29` is also wrong — seven distinct false clauses in one long line** (measured at
`8f4bfc37`):

1. "`cameras.py` (`CameraOptModule`)" — the class is `CameraOpt` (`cameras.py:42`); the file also
   owns `rotation_6d_to_matrix` (`cameras.py:19`) and the merged appearance half.
2. "`losses.py` (`OPTIONAL_LOSSES` registry: depth, normal_consistency, distortion, opacity_reg,
   scale_reg)" — eight entries now; add `appearance_reg`, `pgsr_normal`, `pgsr_multiview`.
3. "`rendering.py` (`render_view` …)" — **`render_view` does not exist.** It is `render_gaussians`
   (`rendering.py:71`) and `render_views` (`rendering.py:241`).
4. "`trainer.py`, `outputs.py`" — `outputs.py` is deleted.
5. the output list `{splats.ply, ckpt.pt, splats.zarr, splats_quality_report.json}` — three files.
6. "PAGaS seam = `render_view` + one `OPTIONAL_LOSSES` entry" — the seam is `render_gaussians`.
7. "`mesh.source: splats` built (TSDF over `splats.zarr` depth/rgb, alpha as confidence)" — Task 18
   moves that reader; until it does this describes a *broken* pipeline, not merely stale prose.

**Adjacent, worth doing here:** `CLAUDE.md:23` still lists **scaffold-gs** under
`## In-Flight Work` ("do not assume their targets are done") even though `Scaffold` is landed,
renamed and carries 66 tests. Move it to the CHANGELOG.

Note `CLAUDE.md` has a hard size limit (a PreToolUse hook enforces it) — every rewrite above must
be net-neutral or shrinking.

- [ ] **Step 7: Trim the `configs/base.yaml` splats comments**

One line per key, no key changes. The block's values are frozen (Ground Rules §7); only the comments shrink.

> **Measured 2026-09-06 at `8f4bfc37`: the frozen surface is INTACT — nothing to repair, only
> comments to fix.** The `splats:` block (`configs/base.yaml:172-238`) declares 24 keys;
> `SplatsConfig` (`trainer.py:74-119`) has 23 dataclass fields plus `enabled`. Exact set match,
> every literal equals its dataclass default, and `losses` (222-226) is exactly
> `default_losses("3dgs")` (`losses.py:42-52`). No orphan keys, no missing keys, no value drift —
> Step 7's `IDENTICAL` check passes provided you touch only comments.

**Five comment-only drifts to fix while shrinking** (all measured at `8f4bfc37`):

- `:169` — "Depth targets = **feedforward depth** masked by `mesh.conf_percentile`". The artifact
  was renamed `feedforward.zarr` -> `pointcloud.zarr` repo-wide on 2026-08-23; this predates the
  refactor.
- `:176` — "scaffold **ignores** `sh_degree`". It now *raises* on a deliberate override
  (`trainer.py:155-162`); only defaults are ignored. Same defect as `docs/splats.md:58`.
- `:191` — `appearance_dim: 0` shown as the scaffold default; the real default is **32**
  (`scaffold.py:69`).
- `:194` — "+0.4 dB on the 30-frame tutorial, **~+3% time**" contradicts the CHANGELOG entry Step 8
  adds, which says "+0.44 dB for **+5%** time". Pick one number and use it in both places.
- `:168` — "Photometric (0.8 L1 + **0.2 SSIM**)" — it is `0.2 * (1 - SSIM)`; `docs/splats.md:94`
  states it correctly. And `:207` "pose_lr x world-frame camera extent" is the rotation term only:
  translation scales by `scene_scale` (`trainer.py:89-91`).

The commented `scaffold:` excerpt (180-191) omits `voxel_size`, `start_stat`, `update_depth`,
`update_hierarchy_factor` and all seven learning rates from `ScaffoldConfig`. That is fine — it is
an excerpt, not a mirror. Do not read "one line per key" as an instruction to expand it.

Prove the values did not move:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/python - <<'PY'
import yaml
import subprocess

after = yaml.safe_load(open("configs/base.yaml"))["splats"]
before = yaml.safe_load(subprocess.run(
    ["git", "show", "HEAD:configs/base.yaml"], capture_output=True, text=True, check=True
).stdout)["splats"]
print("IDENTICAL" if before == after else f"CHANGED: {set(before) ^ set(after)}")
PY
```

Expected: `IDENTICAL`.

- [ ] **Step 8: Write the CHANGELOG entry**

Append to `docs/superpowers/CHANGELOG.md`. This is where the measurements deleted from the trainer docstring go — they are results, not code comments:

> **Two stale entries sit above where you are appending** (measured at `8f4bfc37`):
>
> - **`CHANGELOG.md:12`** — the 2026-08-22 splats-module entry is a *verbatim duplicate of
>   `CLAUDE.md:29`* and therefore carries all seven false clauses listed in Step 6
>   (`CameraOptModule`, the 5-loss registry, `render_view` twice, `outputs.py`, the 4-file output
>   list, the `splats.zarr` TSDF source).
> - **`CHANGELOG.md:6`** — the 2026-08-27 depth-align entry says "`splats.zarr` carries
>   `median_depth`". The artifact is retired. (Its siblings — "2DGS renders keep `median_depth` and
>   its finite-differenced normal" and the `depth_ratio` blend — are still true.)
>
> **Decision: annotate, do not rewrite.** A CHANGELOG is a historical record and those entries were
> true when written, so leave the prose and append a single `(superseded 2026-09-06 — see the
> splats-cleanup entry below)` clause to `:12`. Without it the file says `outputs.py` is a module
> of the package three entries above an entry saying it was merged away, with no cross-reference.
>
> **Also missing:** there is no `scaffold-gs` entry anywhere in the CHANGELOG (zero grep hits),
> even though `CLAUDE.md:23` lists it as in-flight. Step 6 moves it out of `CLAUDE.md`; back-fill a
> one-paragraph entry for it here so the move does not lose the record.

```markdown
### 2026-09-06 — splats-cleanup

`collab_splats/splats/` restructured around two model classes exposing one interface.
`Gaussians` (new `gaussian.py`) and `Scaffold` (renamed from `AnchorField`) each own their
parameters, optimizers, schedulers and densification strategy and answer the same three
attributes and six methods, so `train()` has **zero** representation branches. Deliberately no
abstract base class — the two share a call surface, not an implementation; `tests/splats/
test_model_interface.py` is what holds the contract. New `utils.py` (scene normalization,
coarse-to-fine downscaling, `view_order`, target preparation); `appearance.py` and the vendored
`CameraOptModule` merged into **one** `CameraOpt(torch.nn.Module)` in `cameras.py` whose pose
and appearance halves are selected independently by the existing `splats.pose_opt` /
`splats.appearance_opt` flags and pass through when off (`camera()` / `color()` /
`denormalize()`, no `forward`; `zero_init` / `random_init` dropped — every embedding is
zero-initialized at construction, so a fresh module is the identity; same gsplat citation);
`outputs.py` merged into `rendering.py`;
`collab_splats/nerfstudio/` and `tests/nerfstudio_methods/` deleted (dead since the splats
module landed).

**Artifact change: `splats.zarr` is retired.** `ckpt.pt` gains `cam_to_world`, `intrinsics`,
`image_ids` and `image_size`, and drops `pose_adjust` (its deltas are folded into the stored
poses). `rendering.load_checkpoint` + `rendering.render_views` are the way to get renders back;
the mesh adapter, `evals/scripts/analyze_splats.py` and `eval_splats.py` read the checkpoint.
Existing scenes need the splats stage re-run — there is no reader for the old store.

Behavior is unchanged and was gated on it: three configs (3dgs vanilla, 2dgs vanilla, 3dgs
scaffold) trained 300 steps before the refactor and after each of its two risky phases, matched
on PSNR to 1e-3 dB and on the means distribution to `rtol=1e-4`.

Measurements moved here from the trainer docstring (all on `data/tutorial/`, 30 frames,
1920×1080, 30k steps unless stated): `pose_opt` +0.44 dB for +5% time locally and **+2.2 dB on
the 300-frame GoPro scene**, which is why it defaults on; `appearance_opt` +0.59 dB (3dgs) and
+0.53 dB (2dgs), the single largest splatfacto-parity lever, and anti-additive with the others
— 3dgs with every lever on scores *below* appearance alone; `distortion` shipped at weight
`100.0` where upstream uses `1e-2`, which pruned 2DGS to ~140k gaussians at PSNR 13.5 (now
`0.01`); 2dgs `absgrad` stays off because `DefaultStrategy`'s `grow_grad2d` is calibrated
against the plain gradient.

[spec](specs/2026-09-05-splats-cleanup-design.md) · [plan](plans/2026-09-06-splats-cleanup.md)
```

- [ ] **Step 9: Update the knowledge graph**

`CLAUDE.md` requires it after a structural change, and this refactor renamed a class, deleted three modules and created two:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && graphify update .
```

Expected: a summary line naming the changed files. This is AST-only and costs nothing.

- [ ] **Step 10: Run the splats suite and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/splats -q
echo "PYTEST_RC=$?"
```

Expected: all pass.

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black tests/splats/test_model_interface.py tests/test_cu121_migration.py && \
  /opt/venv/reconstruction/bin/isort tests/splats/test_model_interface.py tests/test_cu121_migration.py && \
  git add -f docs/splats.md docs/superpowers/CHANGELOG.md && \
  git commit --only tests/splats/test_model_interface.py tests/test_cu121_migration.py \
    docs/splats.md docs/superpowers/CHANGELOG.md CLAUDE.md configs/base.yaml \
    -m "refactor(splats): mirror the new layout in tests, docs and config"
```

---

## Task 17: Point the mesh adapter at `ckpt.pt`

The first of three tasks that move the readers off `splats.zarr`. `_splats_to_tsdf_inputs` currently reads five arrays out of the store; it now loads the checkpoint and renders them. Everything downstream of the arrays — the alpha gate, the two depth cuts, the logging — is unchanged.

The one structural change: the store held every render at once, so the function returned four stacked arrays. Rendering 300 views at 1080p costs ~14 GB held that way, and TSDF fusion consumes views one at a time regardless. The function keeps its return shape (the fusion helper and `optimize_color_map` both take stacked arrays) but builds it from a streamed render, so peak memory is one render plus the output arrays, not two copies.

**Files:**
- Modify: `collab_splats/mesh/utils.py:711-812`
- Modify: `tests/mesh/test_splats_adapter.py`

> **CORRECTED 2026-09-06 — the `third_party` skip downgrade does not apply to the wide gate as
> defined.** Every brief on this branch carries the rule that a fresh `git archive` scratch tree
> downgrades the gate, because `third_party/*` is gitignored and guarded tests SKIP instead of
> failing (+6 measured, exit 0), so the symlinks must be restored. Measured at `37da9287` by Task
> 10's round-5 reviewer: running the **three-path wide gate** in a scratch tree with **no
> `third_party` symlinks at all** gives **435 passed / 0 failed / 0 skipped / RC=0** — byte-identical
> to the symlinked run. Within `tests/splats tests/mesh tests/test_cu121_migration.py`, nothing
> guards on `third_party`. The +6 belongs to a wider scope.
>
> Keep symlinking anyway — it costs one line, it is required the moment the scope widens, and a
> reviewer who diffs the skip count against a shared-worktree run loses nothing by it. But **do not
> state the downgrade as a fact about this gate**, and do not treat a scratch-tree skip count of 0
> as evidence the symlinks worked.

> **CORRECTED 2026-09-06 — Step 5 names a checkpoint that does not exist, and the parity it runs
> is structurally blind to the defect Step 1 is built to catch. Measured at `e5d97734`.**
>
> **1. `parity_after_phase4/3dgs_vanilla/ckpt.pt` DOES NOT EXIST.** There is no
> `parity_after_phase4/` directory anywhere under the shared scratchpad. The parity runs actually
> on disk are `parity_before{,_v2,_v3}`, `parity_after_task5{,_repeat,_v2,_v3}`,
> `parity_after_task10`, `parity_after_fix1`/`fix2`/`fix3` and **`parity_after_task12_14`** — the
> last of which is Task 13's run, and it does hold `3dgs_vanilla/ckpt.pt` (50.3 KB, beside
> `splats.ply` and `splats_quality_report.json`). The path in Step 5 must be:
>
> ```
> /tmp/claude-0/-workspace-collab-splats/scratchpad/parity_after_task12_14/3dgs_vanilla/ckpt.pt
> ```
>
> **Do not hardcode that either — re-derive it.** Task 19 runs parity again into a new directory,
> and this path has already gone stale once. List the parity directories and take the newest run
> whose commit you can name.
>
> Note the scratchpad root: the plan's `/tmp/claude-0/-workspace-collab-splats/scratchpad` is the
> **shared** directory holding every parity artifact, not any session's own
> `.../<session-uuid>/scratchpad`. Both exist. Use the shared one.
>
> **2. `mesh_inputs_before.json` is PRESENT (5599 bytes) and its fixture is 64x64 SQUARE.** It
> records `shapes = [[8, 64, 64], [8, 64, 64, 3], [8, 4, 4], [8, 3, 3]]`, `depth_mean = 2.6730`,
> `depth_nonzero_frac = 1.0`, `rgb_mean = 33.0095`. So Step 5 compares 8 views at 64x64 — and
> **on a square frame a height/width swap passes in either direction.** Step 1's own comment says
> exactly this ("24x40, never square"). Therefore:
>
> **Step 5's parity CANNOT substitute for Step 1's shape assertions, and Step 1's tests cannot
> substitute for Step 5.** Both are required, and neither one's passing is evidence for the other.
> Say which you ran.
>
> Note also `depth_nonzero_frac = 1.0` in the before-fixture: that scene renders with no background
> at all, so Step 5's `d_cover < 1e-4` requires the alpha gate to zero nothing there. That is
> consistent with Step 1's `test_splats_adapter_alpha_gate_zeroes_empty_pixels`, which asserts
> `> 0.0` on a *different*, sparser fixture. Do not "reconcile" them.
>
> **3. The CUDA marker line reference is stale.** Step 1 says "same line as
> `tests/splats/test_rendering.py:25`"; the marker is at **`:29`**. (`test_scaffold.py:25`,
> `test_trainer.py:26`.) Copy the marker by symbol, not by line.
>
> **Verified correct as written at `e5d97734` — do not re-measure these:**
> `_splats_to_tsdf_inputs` is at `collab_splats/mesh/utils.py:711-812` exactly as the Files list
> says, and its first parameter is still named `splats_zarr`.
> `rendering.load_checkpoint(path, device)` returns exactly the 6-tuple Step 3 unpacks, in that
> order: `(model, camera_opt, cam_to_world, intrinsics, list(image_ids), (height, width))`.
> `rendering.render_views(model, camera_opt, cam_to_world, intrinsics, height, width)` matches
> Step 3's call.
> **`model.primitive` exists on both classes** as a `self`-assigned instance attribute, so Step 3's
> median-depth guard works — it is a different member from the class attribute `primitive_unit`
> (`gaussian.py:111`, `scaffold.py:264`) that `rendering.py:403` uses as the representation
> discriminator. Do not conflate the two.
> `tests/mesh/test_splats_adapter.py` and `tests/mesh/test_absent_confidence.py` both exist.

> **CORRECTED 2026-09-06 — T17-1..T17-5, pre-verification at `2d5c01d7`. The lead item is a hole in
> the gates that no gate can report.**
>
> **T17-1 (BLOCKING for how this task is judged). Nothing on this branch covers `mesh(source=
> "splats")` end-to-end. The three tests that would are dark today, for an unrelated reason, and
> they live outside the wide gate anyway.**
>
> `tests/wrapper/test_splats_stage.py` at HEAD: **3 failed, 13 passed**. The three failures are
> exactly the tests covering the path Tasks 17 and 18 rewrite:
>
> ```
> FAILED tests/wrapper/test_splats_stage.py::test_mesh_source_splats_fuses_from_splats_zarr
> FAILED tests/wrapper/test_splats_stage.py::test_mesh_sfm_aligned_zarr_fuses
> FAILED tests/wrapper/test_splats_stage.py::test_mesh_stage_forwards_splat_depth_from_the_config
> ```
>
> All three raise `KeyError: 'splat_max_depth_frac'` at `collab_splats/wrapper/reconstructor.py:1628`,
> which is **inside `mesh` (1560-1635)** — i.e. they die *before* dispatch to `_run_tsdf_mesh`, where
> `_splats_to_tsdf_inputs` is called at `:632`. They therefore cannot detect a Task 17 or Task 18
> regression: they never reach the code either task touches. And `tests/wrapper/` is outside the
> three-path wide gate (`tests/splats tests/mesh tests/test_cu121_migration.py`) regardless, so
> nothing this branch runs would report it either way.
>
> This is the quiet version of a false green: the unit tests in `tests/mesh/test_splats_adapter.py`
> pin the adapter's own contract, and the integration tests that would catch a wiring error between
> the adapter and its caller are switched off. Step 1's tests are necessary and are not sufficient.
>
> **The documented fix works — measured.** `docs/known-test-failures.md:1-30` already records this
> as pre-existing, owner "whoever owns the mesh floater work", with the remedy written out: two keys
> added to the hand-rolled `"mesh"` block at `tests/wrapper/test_splats_stage.py:58-73`. (The doc's
> heading says "one line"; it is two.) Applied in a pinned tree archived from HEAD with
> `third_party/*` symlinked in:
>
> ```python
>             "splat_depth": "expected",
>             "splat_max_depth_frac": None,
>             "splat_max_depth_grad": None,
> ```
>
> → **16 passed**, from 3 failed / 13 passed. **Task 18 already modifies that file** (10 hits), so
> restoring the coverage costs it nothing. Do it there, and run
> `pytest tests/wrapper/test_splats_stage.py` as an *extra* gate for Tasks 17 and 18 — it is not in
> the wide gate and will not appear unless asked for by name.
>
> **T17-2. `docs/known-test-failures.md:5-6` is falsified, and by more than the entry admits.** It
> states "the splats cleanup does not touch `reconstructor.py`, `configs/base.yaml`, or this test
> file." All three clauses are false on this branch: **Task 18** touches `reconstructor.py` (4
> logical sites / 15 hits) *and* `tests/wrapper/test_splats_stage.py` (10 hits), and **Task 16**
> touches `configs/base.yaml`. Task 18's existing correction names only `reconstructor.py`. Whoever
> lands the fixture fix should correct that sentence in the same commit rather than leaving a
> known-failures entry asserting a boundary this branch crosses three ways.
>
> **T17-3. `render_views` is a GENERATOR — Step 3's streaming claim is sound.** Confirmed at
> `collab_splats/splats/rendering.py:255`. The task's whole structural argument is that the store
> held every render at once and the checkpoint path can yield instead; that argument rests on this
> and it holds. Separately, **`render_view` (singular) is ABSENT from `rendering.py`** — which is
> what T19-5 already records as a stale reference. Do not reintroduce it.
>
> **T17-4. Two callers of `_splats_to_tsdf_inputs` live OUTSIDE this task's Files list.** Both are
> covered by Task 18, and the T18-depends-on-T17 ordering is already recorded, so this is a
> sequencing note, not a gap:
>
> - `collab_splats/wrapper/reconstructor.py:632` — one positional plus `conf_percentile`,
>   `splat_depth`, `max_depth_frac`, `max_depth_grad`.
> - `evals/scripts/analyze_splats.py:289` — two positional.
>
> Changing the signature in Task 17 alone therefore breaks both until Task 18 lands. **Run Tasks 17
> and 18 in that order and do not parallelise them**, and expect the interval between them to be red
> at those two call sites by construction.
>
> **T17-5. Re-verified unchanged at `2d5c01d7`, on top of what the `e5d97734` block already
> verified.** `_splats_to_tsdf_inputs` still at `collab_splats/mesh/utils.py:711-812`, signature
> `(splats_zarr, conf_percentile, splat_depth, max_depth_frac, max_depth_grad)`.
> `load_checkpoint(path, device)` still at `:418`, still returning the 6-tuple.
> `self.primitive` is assigned on both classes — `Gaussians` at `gaussian.py:146` and `:405`,
> `Scaffold` at `scaffold.py:292` and `:769` — and the `primitive_unit` AnnAssign is present on both,
> so the two-member distinction the earlier block warns about still holds exactly as written.

> **IMPLEMENTATION RECORD T17-6..T17-11 — Task 17 shipped as `8da1d931` (`collab_splats/mesh/utils.py`
> + `tests/mesh/test_splats_adapter.py`, 267+/133−). Five of these falsify text in this plan or in the
> dispatch brief; T17-9 changes a number Task 18's reviewer will otherwise measure as a regression.**
>
> **T17-6 (BLOCKING for Task 18's baseline). T17-1's "no gate can see the caller break" is FALSE.**
> `tests/wrapper/test_splats_stage.py` goes **3 failed / 13 passed → 4 failed / 12 passed** at
> `8da1d931`. The new failure is `test_run_tsdf_mesh_forwards_splat_depth_to_the_adapter`, which
> reaches `reconstructor.py:632` → `mesh/utils.py:764` → `load_checkpoint` → `torch.load` and raises
> `IsADirectoryError`. **Task 18's control is 4 failed, not 3.** A reviewer taking 3 from the earlier
> text will read the deliberate Task-17 break as a Task-18 regression.
>
> **T17-7. The two callers Task 17 leaves broken fail at RUNTIME, not at import.**
> `collab_splats/wrapper/reconstructor.py:632` and `evals/scripts/analyze_splats.py:289`: arity and
> all four keyword names (`conf_percentile`, `splat_depth`, `max_depth_frac`, `max_depth_grad`)
> survive the change, so there is no `TypeError` and no `ImportError`. Both still pass a `splats.zarr`
> **directory**; `Path.exists()` returns True so the `FileNotFoundError` guard passes, and `torch.load`
> then raises `IsADirectoryError: [Errno 21]`. With the store absent instead you get the intended
> `FileNotFoundError: … — run the splats stage first`. `tests/evals/test_analyze_splats.py` is green
> on both sides (1 passed) — it only exercises the module-level `depth_to_normal` helper and never
> reaches `:289`, so it is **not** a gate on this.
>
> **T17-8. Step 3's "keep `import zarr`, the feedforward path still uses it" is FALSE.** An AST sweep
> found `:745` — inside the very function being rewritten — as the module's **only** use of `zarr`.
> The import is deleted at `8da1d931`; nothing imports `zarr` from `collab_splats.mesh.utils`.
>
> **T17-9. Step 3 as written ships a `NameError`.** Its retained tail passes `depth_array` to
> `logger.info`, a name its own replacement head deletes (the head defines `depth_key`). Fixed in the
> commit; an AST free-name check over the shipped function now reports `free names: []`. **Do not copy
> Step 3's block verbatim into any later task.**
>
> **T17-10. Step 5's parity fixture is unusable, beyond the already-corrected missing directory.**
> `mesh_inputs_before.json` was captured from `parity_before` (04:52), whose ckpt carries only
> `['appearance','config','pose_adjust','splats']` — no `image_ids`, no cameras — so today's
> `load_checkpoint` raises `KeyError: 'image_ids'`. That run is also a **different model generation**
> from every loadable ckpt (psnr 16.5583 vs 17.5867); run as written it fails at poses 7.29e-4,
> Δdepth 1.3155, Δcover 0.51779, Δrgb 9.0850 — all fixture skew, no defect. Substituted with
> `parity_before_v3/3dgs_vanilla/splats.zarr` vs `parity_after_task12_14_v2/3dgs_vanilla/ckpt.pt`,
> proven same-generation first (splats tensors ≤ 1.4e-6, store c2w vs ckpt cam_to_world 5.09e-11, K
> exact). Result: rgb **bit-identical** (0 of 98,304 texels differ), max |Δdepth| 5.0545e-05, max
> |Δc2w| 5.09e-11, |ΔK| 0, coverage identical.
>
> **T17-11. One mutant is not a clean single-test kill, and that is correct here.** Of eight mutants
> planted one at a time in a pinned tree (control 74 passed, RC=0), seven — M2a/M2b view order, M3
> c2w inverted, M4 intrinsics broadcast from view 0, M5 `alpha > 0` → `>= 0`, M6 far cut disabled, M7
> grad cut disabled — each give **1 failed, the intended test, zero collateral**. **M1** (height/width
> swapped) is killed by `test_splats_adapter_reads_a_checkpoint` on `assert (3, 40, 24) == (3, 24, 40)`
> but takes **9 other tests with it**, all genuine detections of the same swap. Attribution is still
> satisfied; a reviewer diffing mutant output should expect the breadth rather than read it as noise.
>
> **Also measured at `8da1d931`, for anyone re-deriving a baseline:** `tests/mesh` 71 → **74 passed**,
> 0 skipped, RC=0. Wide gate **462 passed / 0 failed / 0 skipped**, against a like-for-like **459**
> re-measured at the true parent `3ecd780a` — the honest delta is **+3, its net new tests**, not the
> +10 a stale 452 baseline implies. The new tests fail against the unmodified adapter (14 failed / 2
> passed at `0136a481`, every failure from `zarr.open_group` choking on a `.pt`). Streaming cuts peak
> CUDA to **0.209 GB from 2.996 GB (14.4×)** at 60 views × 720×1280 for the same 0.360 GB of output.

- [ ] **Step 1: Write the failing test**

Replace the zarr-writing fixture in `tests/mesh/test_splats_adapter.py` with one that writes a checkpoint. This is the whole fixture — the old `_write_splats_zarr` helper is deleted:

```python
def _write_ckpt(tmp_path, *, primitive="3dgs", n_views=3, height=24, width=40):
    """
    A tiny trained-model checkpoint: 200 gaussians over `n_views` cameras looking at the origin.
    """
    rng = np.random.default_rng(0)
    cfg = SplatsConfig.from_dict({"primitive": primitive, "max_steps": 10, "losses": {}})
    model = Gaussians(
        cfg,
        rng.uniform(-0.5, 0.5, (200, 3)).astype(np.float32),
        rng.integers(0, 255, (200, 3)).astype(np.uint8),
        scene_scale=1.0,
        n_views=n_views,
        device="cpu",
    )
    cam_to_world = torch.eye(4)[None].repeat(n_views, 1, 1)
    cam_to_world[:, 2, 3] = -3.0
    cam_to_world[:, 0, 3] = torch.linspace(-0.5, 0.5, n_views)
    intrinsics = torch.tensor(
        [[[20.0, 0.0, width / 2], [0.0, 20.0, height / 2], [0.0, 0.0, 1.0]]]
    ).repeat(n_views, 1, 1)

    ckpt = model.checkpoint()
    ckpt["config"] = asdict(cfg)
    ckpt["cam_to_world"] = cam_to_world
    ckpt["intrinsics"] = intrinsics
    ckpt["image_ids"] = list(range(n_views))
    ckpt["image_size"] = (height, width)
    ckpt["appearance"] = None
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)
    return path


@cuda
def test_splats_adapter_reads_a_checkpoint(tmp_path):
    ckpt_path = _write_ckpt(tmp_path)

    depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(ckpt_path)

    # 24x40, never square: `image_size` is stored (height, width) and the adapter renders
    # (width, height) — on a square checkpoint a swap in either direction passes
    assert depths.shape == (3, 24, 40)
    assert rgbs.shape == (3, 24, 40, 3)
    assert rgbs.dtype == np.uint8
    assert c2w.shape == (3, 4, 4)
    assert intrinsics.shape == (3, 3, 3)


def test_splats_adapter_raises_on_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the splats stage first"):
        _splats_to_tsdf_inputs(tmp_path / "ckpt.pt")


def test_splats_adapter_refuses_median_depth_on_a_3dgs_checkpoint(tmp_path):
    ckpt_path = _write_ckpt(tmp_path, primitive="3dgs")

    with pytest.raises(ValueError, match="needs a 2dgs"):
        _splats_to_tsdf_inputs(ckpt_path, splat_depth="median")


def test_splats_adapter_rejects_an_unknown_depth_name_before_any_io(tmp_path):
    # The value check must fire on the config, not on a missing file
    with pytest.raises(ValueError, match="must be 'expected' or 'median'"):
        _splats_to_tsdf_inputs(tmp_path / "does_not_exist.pt", splat_depth="surface")


@cuda
def test_splats_adapter_alpha_gate_zeroes_empty_pixels(tmp_path):
    ckpt_path = _write_ckpt(tmp_path)

    depths, _, _, _ = _splats_to_tsdf_inputs(ckpt_path)

    # A 200-gaussian model over a 24x40 frame leaves background: those pixels must fuse as 0
    assert float((depths == 0).mean()) > 0.0
```

Extend the imports, and add the CUDA guard the module does not yet have (same line as
`tests/splats/test_rendering.py:25`). The two tests that actually render are decorated with
it; the three that assert on a raise never reach a rasterizer and stay unguarded:

```python
from dataclasses import asdict

import numpy as np
import pytest
import torch

from collab_splats.mesh.utils import _splats_to_tsdf_inputs
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.trainer import SplatsConfig

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")
```

Building a `Gaussians` on the CPU is fine — only rasterization is CUDA-only — so `_write_ckpt`
keeps `device="cpu"` and the checkpoint it saves is device-agnostic.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_splats_adapter.py -q
```

Expected: failures — the adapter calls `zarr.open_group` on a `.pt` file.

- [ ] **Step 3: Rewrite the adapter's head**

Replace `collab_splats/mesh/utils.py:711-763` (the signature through the four array reads) with:

```python
def _splats_to_tsdf_inputs(
    ckpt_path: Path,
    conf_percentile: float | None = None,
    splat_depth: str = "expected",
    max_depth_frac: float | None = None,
    max_depth_grad: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Render a trained splat checkpoint into (depths, rgbs, c2w, intrinsics) for TSDF fusion.

    - Renders every training view from `ckpt.pt` rather than reading a stored render, so the
      geometry fused is always the checkpoint's own. Views stream one at a time: 300 renders
      at 1080p held together would be ~14 GB.
    - `alpha` is the confidence: pixels with alpha == 0 never reach Open3D, and
      `conf_percentile` drops the lowest-alpha percentile globally (same rule as the
      feedforward path's confidence gate). Note that a trained splat field renders alpha
      near 1 almost everywhere (median 0.998 on GH010229), so this gate is weak on its own —
      the two depth filters below are what remove blown-out background.
    - `max_depth_frac` and `max_depth_grad` cut depth before fusion: too far to be
      constrained, or straddling a discontinuity. Both are ratios, never distances, and both
      ship off — measured on GH010229, the component cleaner already removes everything they
      remove, and a `max_depth_grad` low enough to catch anything deletes whole oblique
      surfaces. Turn one on only for a scene that shows the failure it names.
    - `splat_depth` picks which rendered depth to fuse: "expected" (alpha-weighted, the
      default) or "median" (RaDe-GS surface depth, 2dgs renders only).
    - Poses are the checkpoint's, which are the poses that were trained, pose-opt deltas
      included.
    - Already native to the training frames, so there is no upsampling path.

    Args:
        ckpt_path: the splats stage's `ckpt.pt`.
        conf_percentile: alpha percentile to drop, or None for the alpha > 0 gate alone.
        splat_depth: "expected" or "median".
        max_depth_frac: far cut as a fraction of the camera trajectory's extent, or None.
        max_depth_grad: relative depth-gradient cut, or None.

    Returns:
        (depths (N, H, W) float32 with masked pixels zeroed, rgbs (N, H, W, 3) uint8,
        c2w (N, 4, 4) float32, intrinsics (N, 3, 3) float32).
    """
    # Local import: collab_splats.splats pulls in gsplat's CUDA extension, and mesh.utils is
    # imported by the dashboard's fast-bind path, which must not pay for that.
    from collab_splats.splats.rendering import load_checkpoint, render_views

    # Value check before any IO so a typo fails on the config, not on a missing file
    if splat_depth not in ("expected", "median"):
        raise ValueError(f"mesh.splat_depth must be 'expected' or 'median', got {splat_depth!r}")

    # Loud failure before loading: the splats stage is never auto-run by mesh()
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} — run the splats stage first")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, camera_opt, c2w_tensor, intrinsics_tensor, image_ids, (height, width) = load_checkpoint(
        ckpt_path, device
    )

    # median_depth is a 2dgs render. Both model classes carry `primitive` (set from the config
    # they were built with), so this is decided before rendering rather than by an array that
    # turns out to be missing afterwards
    if splat_depth == "median" and model.primitive != "2dgs":
        raise ValueError(
            f"{ckpt_path} is a {model.primitive} run — mesh.splat_depth: median needs a 2dgs "
            "checkpoint; re-run the splats stage with primitive: 2dgs, or use "
            "splat_depth: expected."
        )

    # Stream the renders into the stacked arrays TSDF fusion and the color-map optimizer take
    n_views = len(image_ids)
    depths = np.empty((n_views, height, width), dtype=np.float32)
    rgbs = np.empty((n_views, height, width, 3), dtype=np.uint8)
    alpha = np.empty((n_views, height, width), dtype=np.float32)
    depth_key = "median_depth" if splat_depth == "median" else "depth"
    renders = render_views(model, camera_opt, c2w_tensor, intrinsics_tensor, height, width)
    for view, render in enumerate(renders):
        depths[view] = render[depth_key][0, ..., 0].cpu().numpy()
        rgbs[view] = (render["rgb"][0] * 255).round().byte().cpu().numpy()
        alpha[view] = render["alpha"][0, ..., 0].cpu().numpy()
    c2w = c2w_tensor.cpu().numpy().astype(np.float32)
    intrinsics = intrinsics_tensor.cpu().numpy().astype(np.float32)
```

Everything from the `# Alpha gate:` comment to the `return` is unchanged, except that its first line drops the store read — `alpha` is now the local array built above, so delete `alpha = store["alpha"][:]` and keep `keep = alpha > 0`.

The `zarr` import at the top of `mesh/utils.py` stays: the feedforward path still uses it.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_splats_adapter.py -q
```

Expected: all pass.

- [ ] **Step 5: Prove the rendered tuple matches the one the store held**

This is the spec's phase-8 parity check, and the only evidence that retiring `splats.zarr` did
not change what the mesh stage fuses. It compares the tuple this function now builds against
`mesh_inputs_before.json`, captured in Task 2 Step 4 from the store the same fixed-seed run
wrote. Task 13's parity run left a checkpoint from that same run at
`parity_after_phase4/3dgs_vanilla/ckpt.pt`.

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -c "
import json
import numpy as np
from pathlib import Path
from collab_splats.mesh.utils import _splats_to_tsdf_inputs

scratch = Path('/tmp/claude-0/-workspace-collab-splats/scratchpad')
before = json.loads((scratch / 'mesh_inputs_before.json').read_text())
depths, rgbs, c2w, K = _splats_to_tsdf_inputs(scratch / 'parity_after_phase4' / '3dgs_vanilla' / 'ckpt.pt')

assert [list(a.shape) for a in (depths, rgbs, c2w, K)] == before['shapes'], 'shape drift'
assert np.allclose(c2w, np.array(before['c2w']), atol=1e-6), 'poses drift'
assert np.allclose(K, np.array(before['K']), atol=1e-6), 'intrinsics drift'
d_depth = abs(float(depths.mean()) - before['depth_mean'])
d_cover = abs(float((depths > 0).mean()) - before['depth_nonzero_frac'])
d_rgb = abs(float(rgbs.mean()) - before['rgb_mean'])
print(f'd_depth={d_depth:.3e} d_cover={d_cover:.3e} d_rgb={d_rgb:.3e}')
assert d_depth < 1e-4 and d_cover < 1e-4 and d_rgb < 0.5, 'render drift'
print('MESH INPUTS PARITY OK')
"
```

Expected: a `d_*` line then `MESH INPUTS PARITY OK`. The rgb tolerance is half a quantisation
level because the store held `uint8` and the render is rounded to `uint8` here; depth and
coverage are float and should be near-exact.

If the poses or intrinsics drift, the checkpoint is not carrying the pose-opt deltas — check
`write_outputs` in Task 9 saves `refine.camera(...)`-corrected poses, not the raw input poses.
If only depth drifts, check the depth key: the store's `depth` array was the *expected* depth,
so `splat_depth="expected"` must select `render["depth"]`, not `render["median_depth"]`.

- [ ] **Step 6: Run the whole mesh suite**

The adapter is shared, and `tests/mesh/test_absent_confidence.py` exercises the feedforward path through the same module.

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/mesh -q
echo "PYTEST_RC=$?"
```

Expected: `PYTEST_RC=0`.

- [ ] **Step 7: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/mesh/utils.py tests/mesh/test_splats_adapter.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/mesh/utils.py tests/mesh/test_splats_adapter.py && \
  git commit --only collab_splats/mesh/utils.py tests/mesh/test_splats_adapter.py \
    -m "refactor(mesh): fuse splat meshes from ckpt.pt instead of splats.zarr"
```

**Task 12 — fix round 3 landed as `8d88f478`** (`tests/splats/test_scaffold.py` only, +138/−5, zero
`collab_splats/` paths). It closes the flaky export/decode parity assertion and the near-vacuity
behind it. `tests/splats` 367 → 371, wide gate 471 → 472, 0 skips, RC=0. Spec re-review in flight.

The dispatch brief's mechanism hypothesis held — `mlp_opacity` ends in `Tanh`, so when no offset is
open the shared fallback force-keeps the most opaque offset even though it is ≤ 0, after which
`decode` returns it raw (negative) while `export_gaussians` clamps it before the logit (`-9.2102`).
Both correct; the test was asserting parity across a branch designed to diverge. Per-draw census of
the real fixture, 600 draws: the 121 parity failures are **exactly** the 121 all-closed draws, zero
elsewhere.

> **The brief understated the vacuity, and its suggested remedy would not have caught the dominant
> case.** Full census: **22.83% full grid / 57.00% partial keep / 20.17% all-closed**. The dominant
> failure is the **partial** keep — decoding *some* rows, neither one nor all — and a "count exceeds
> 1" guard, which the brief proposed, is blind to it. Every replaced guard therefore asserts the
> **exact** grid size `n_primitives * n_offsets`, never a floor.

The remedy arms the fixture **structurally, without seeding**: `_open_every_offset` shifts
`mlps.mlp_opacity[-2].bias += 1.5`, leaving the head freshly random every run (600/600 armed draws
full grid). `+1.5` not `+3.0` because at `+3.0` the opacity spread collapses to `1.7e-4`, below the
test's own `2e-4` tolerance, which would make a permuted opacity column invisible; at `+1.5` the
spread is `3.9e-3` to `0.206`. A new deliberately-**unarmed** test covers the degenerate branch with
absolute literals (`decode < 0`; `export == -9.2102`) rather than round-trip parity, since a
co-mutation would satisfy parity. Twelve mutants all killed; M8 (force every neural opacity negative)
turns **all seven** armed tests red while the all-closed test correctly passes — the non-vacuity
proof.

Three further corrections from the round, all accepted:

- **Every `scaffold.py` line number in the brief was wrong at every HEAD**, and drifted a further +3
  between the implementer's measurement and its commit. The committed test file cites none; only the
  commit *message* prose drifted, and it was correctly left unamended rather than rewriting history
  under four live committers.
- **The flake rate "3 of 20" was a small-sample artefact** — 6/60 at pytest level, 20.17% per draw.
- **The degenerate branch was not entirely untested**:
  `test_decode_never_returns_zero_gaussians_when_every_offset_is_closed` already covered decode's
  fallback *count*. What was genuinely uncovered is the export-vs-decode opacity *divergence*.
- **A mutant can crash rather than fail.** M6 (delete decode's fallback) SIGFPEs the pytest process
  (rc=−8) through gsplat's projection kernel. A harness reading only the failure count misreads that
  as a survival.

Reported, out of scope, owed: `test_decode_is_differentiable_into_the_mlps` and
`test_render_gradient_reaches_the_anchor_features` assert only `grad is not None` — row-count
insensitive, and they would survive a one-row decode regardless of arming.

**Task 18 outcome — landed as `b6b9c7fe`** (nine paths, +337/−547). Every `splats.zarr` reader now
opens `ckpt.pt`. Gate `tests/wrapper/test_splats_stage.py tests/mesh`: **4 failed / 92 passed → 101
passed**, RC=0, 0 skipped. Spec review in flight.

> **A repo-metadata failure blocked commits for every agent on this branch, and is worth recognizing
> if it recurs.** Two commit attempts failed with
> `fatal: cannot update the ref 'refs/heads/clean/splats': unable to append to
> '.git/logs/refs/heads/clean/splats': Input/output error` (RC=128) while `git rev-parse` still
> showed the old tip. `/workspace` is a MooseFS FUSE mount; git's `O_APPEND` reflog write was
> reported EIO but the space persisted as a hole — attempt 2 grew the file by exactly **394 NUL
> bytes**. Repaired by a single `os.truncate` back to offset `23771`, the end of the last valid
> reflog line, asserted first to end in `\n` and contain no NULs. `git rev-parse clean/splats` read
> `8d88f478` before and after, so no peer commit was lost. **Verified by the coordinator after the
> fact: 112 reflog lines, 0 NUL bytes, no probe marker, file ends in a newline, last entry is
> `b6b9c7fe`.** The implementer had appended a 7-byte `#probe` marker while diagnosing and removed it
> in the same repair; it is gone.

**The mutation audit found a pre-existing hole the task was not sent to fix, and closed it.** Ten
mutants, gate `tests/wrapper/test_splats_stage.py`, each applied alone. Six `reconstructor.py`
mutants died with distinct kill sets — M3/M4 (the skip-path and trained-path `return` statements)
each uniquely, which is what discriminates them from M1/M2 (the literals, each feeding two call
sites). But **three of the four tunables `_run_tsdf_mesh` hands the splats adapter were unpinned**:
`conf_percentile=`→`None`, `max_depth_frac=`→`None` and `max_depth_grad=`→`None` all **survived**,
and survived a widened `tests/wrapper tests/mesh` gate too. Only `splat_depth` was covered. Closed
with **three separate tests, one knob each** — a single test asserting all four would die for any one
of them and could not say which broke. All four now killed uniquely.

Two decisions recorded rather than acted on:

- **`CLAUDE.md:29` left unchanged.** It is a dated `Recently completed (2026-08-22)` history entry
  recording what shipped on that date; editing it would falsify a dated record, CLAUDE.md routes
  completed work to CHANGELOG.md, and it was not in the task's ownership list.
- **The whole-file `black` run on `collab_splats/wrapper/reconstructor.py` was deliberately skipped.**
  `--diff` showed 20 hunks, only 2 overlapping the task's edits, and in both the lines black wanted to
  rewrite are pre-existing (`logger.info(...)`, `raise ValueError(...)`) — the task's own lines it
  leaves alone. The other 18 are unrelated peer code. Reformatting them on a branch with four live
  agents is collision risk for no benefit. Verified instead: no line over 120. `isort` likewise
  proposed exactly one rewrap, of a *pre-existing* import, and it was not applied.

**Notebooks shipped unexecuted, and the commit message says so.** `data/outputs/` does not exist in
this worktree, so `nbconvert --execute` cannot run. Cells were rewritten and outputs cleared
(0 output items, every `execution_count` None), JSON validated, every code cell parsed under `ast`.
**No outputs were fabricated.**

Ten corrections to the dispatch brief, all accepted:

- **The control was `4 failed, 92 passed`, not `4 failed, 86 passed`** — Task 17 added 6 tests to
  `tests/mesh` after the brief was written. Same 4 failures, so the substance held.
- **Step 10's "expect exactly two lines" is true only of code.** The three retirement notes the brief
  itself commissioned are hits, so the honest live count outside `docs/superpowers/` is **7**: two
  Task 10 guards (intact), three deliberate retirement notes, the verbatim historical `FAILED` block,
  and `CLAUDE.md:29`. Zero hits in `collab_splats/`, `evals/`, `configs/*.yaml` or either notebook.
- **`np.` appears 38 times in the test file, not 15** (conclusion unchanged).
- Three line citations had drifted (`…without_zarr_raises` at 173 not 172;
  `…forwards_splat_depth_to_the_adapter` spans 277–303).
- **T18-9 mischaracterised `docs/known-test-failures.md`**: its "Cause." paragraph was already
  correct; what was stale was the `(OPEN, mesh owner)` header, the owner note, and the sentence
  claiming the splats cleanup does not touch `reconstructor.py`.
- **T18-8 missed a second edit in `docs/splats.md`** — "`train` writes **four** files" goes wrong the
  moment the table row is removed.
- **T18-7 under-specified nb03**: cell 8's comment and cell 10's artifact loop were unnamed; the
  latter would have printed `MISSING …/splats.zarr` on every run after the retirement.
- `analyze_splats.py:222` exceeds 120 chars — **pre-existing**, verified against `HEAD`.
- **The brief never mentioned the 21 pre-existing `tests/wrapper` failures** outside the gate subset
  (3 stale-`base.yaml` + 18 `_run_sfm` VDA-context). An A/B swapping in `HEAD`'s `reconstructor.py`
  gives **identical failure sets, 21 both sides, none new, none fixed** — they are not Task 18's, but
  anyone widening the gate will hit them.
- T18-3's formatting instruction is not safe as written on `reconstructor.py` — see above.

**Task 17 outcome — implemented at `8da1d931`, fix round 1 at `efc4d7b1`.**

The production change landed as `8da1d931`: `_splats_to_tsdf_inputs` takes a `ckpt_path`, calls
`load_checkpoint` and streams `render_views` instead of opening `splats.zarr`. The spec review
confirmed the production change is exactly Task 17 and nothing more, and upheld five findings — one
stale docstring and four test-strength gaps, every one of them an instance of this branch's defect
class. Fix round 1 closed all five in `efc4d7b1` (`collab_splats/mesh/utils.py` +1/-1,
`tests/mesh/test_splats_adapter.py` +157), scope verified by `git show --stat`.

| finding | what was wrong | mutant that survived |
|---|---|---|
| F1 | `mesh_from_tsdf_inputs`'s docstring still said `(pointcloud.zarr and splats.zarr)` | — (blocks Task 18's sweep) |
| F2 | the `render_views` call's **six** arguments were unguarded — the fake ignored all of them, so only `(height, width)` was under any guard, via output shape | MX2 (identity K) and MX3 (inverted poses): both turn the render into pure background, zero-fraction 0.7653 → 1.0000, suite stays green |
| F3 | `returns_each_views_own_intrinsics` asserted `K[0,0]` and `K[1,1]` — **2 of 9 entries** — while the returned K feeds Open3D TSDF fusion directly | MX8 (cx/cy zeroed), MX9 (K transposed) |
| F4 | `stacks_views_in_render_order` used rgb `0.2/0.4/0.6`; ×255 those are 51.0/102.0/153.0, which **round and truncate to identical bytes**, so the uint8 quantisation rule was untested | MX4 (drop `.round()`) |
| F5 | two far-depth tests used square 2×2 fixtures, blind to an H/W swap (minor) | — |

All five mutants are now killed with **unique CPU-side killers**. `tests/mesh` 74 → **80**, the wide
gate 463 → **469**, 0 skips, RC=0.

> **Deliberate plan deviation, upheld.** `test_splats_adapter_alpha_gate_zeroes_empty_pixels` now
> asserts `0.0 < frac < 1.0`. The plan mandates `float((depths == 0).mean()) > 0.0` verbatim — a
> **one-sided** bound that an all-background render satisfies, which is precisely how MX2 and MX3
> slipped past it. A guard whose passing condition is also its failure condition is not a guard.

Two corrections to the dispatch brief, both measured by the implementer and accepted:

- The brief claimed mutant M1 (`image_size` unpacked as `(width, height)`) is killed **only** by
  `@cuda` tests. False — M1 kills 10 tests, 9 of them scripted CPU tests. The concern was real but
  attached to the wrong mutant: **MX10** (H/W swapped only at the render call) genuinely was
  `@cuda`-only-killed, and was given a CPU killer.
- `git archive` **does** create `third_party/` — `third_party/README.md` is tracked. The `mkdir -p`
  in every pinned-tree recipe on this branch is a harmless no-op with a wrong stated reason. The
  symlinks are still required (the clones are gitignored) and **the assertion that the skip count is
  0 is what actually protects the gate.**

**The implementer caught its own next instance mid-work** — the recording fake captures six arguments
and its first cut asserted only four; `model` and `camera_opt` were recorded but never checked. It
closed the gap with **two separate tests** so each arm keeps a unique killer (MX14 fails only the
model test, MX12 only the camera_opt test); a single combined assertion would have left neither arm
with one. That is the rule applied correctly: *when a guard aggregates N things, the test needs N
cases.*

---

## Task 18: Move the remaining `splats.zarr` readers

Everything else that opened the store. **Nine files, 47 tracked hits**, no shared logic between them — work down the list and re-run the named gate after each.

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (4 logical sites, 15 hits)
- Modify: `evals/scripts/analyze_splats.py` (11), `evals/scripts/eval_splats.py` (2)
- Modify: `tests/wrapper/test_splats_stage.py` (10)
- Modify: `docs/splats.md` (1), `configs/README.md` (2)
- Modify: `docs/known-test-failures.md` (1)
- Modify: `docs/source/tutorials/03_splats/train_splats.ipynb` (6), `docs/source/tutorials/06_mesh/splats_mesh.ipynb` (8)

> **Measured 2026-09-06 at `8f4bfc37`, after Task 10 landed.** The sweep covered the literal
> `splats.zarr`, the identifier `splats_zarr`, the constant `SPLATS_ZARR`, every `zarr.open*` call
> and every `/ "*.zarr"` join across `collab_splats/`, `evals/`, `configs/*.yaml`, `scripts/*.sh`,
> `batch.py` and the dashboard. **No concatenated or indirect path exists** — the only
> variable-held path is the notebook constant `SPLATS_ZARR`. `collab_splats/splats/` itself now
> has zero zarr references.
>
> Two files carry hits that are **NOT** this task's: `collab_splats/mesh/utils.py` (8) and
> `tests/mesh/test_splats_adapter.py` (29) belong to Task 17, and
> `docs/known-test-failures.md` also asserts at lines 5-6 that "the splats cleanup does not touch
> `reconstructor.py`" — which this task falsifies, so fix that sentence too, not just line 9.
>
> **CORRECTED 2026-09-06 — every count in this task is a count of matching LINES, not of
> occurrences, and the two differ by a third. Re-measured at `a0cb4933`; nothing has drifted.**
>
> An implementer who counts occurrences will find more hits than this plan promises in six of the
> eleven files and reasonably conclude the plan went stale. It has not. Both counts, at HEAD:
>
> | file | plan | matching lines | occurrences |
> |---|---|---|---|
> | `collab_splats/wrapper/reconstructor.py` | 15 | 15 | 18 |
> | `evals/scripts/analyze_splats.py` | 11 | 11 | 13 |
> | `evals/scripts/eval_splats.py` | 2 | 2 | 2 |
> | `tests/wrapper/test_splats_stage.py` | 10 | 10 | 15 |
> | `docs/splats.md` | 1 | 1 | 1 |
> | `configs/README.md` | 2 | 2 | 2 |
> | `docs/known-test-failures.md` | 1 | 1 | 1 |
> | `.../03_splats/train_splats.ipynb` | 6 | 6 | 6 |
> | `.../06_mesh/splats_mesh.ipynb` | 8 | 8 | 13 |
> | `collab_splats/mesh/utils.py` *(Task 17)* | 8 | 8 | 9 |
> | `tests/mesh/test_splats_adapter.py` *(Task 17)* | 29 | 29 | 33 |
>
> Nine-file totals: **56 matching lines, 71 occurrences** — the header's "47 tracked hits" matches
> neither, and is the one number in this block that is simply wrong. **Work the file list, not the
> total.**
>
> **Zero drift since the original sweep.** Counted at `8f4bfc37` — the commit the measurement block
> above names — and at `a0cb4933`, every file gives an identical number on both counting methods.
> Nothing this branch has done since Task 10 has added or removed a single reference. So do not
> re-derive the file list; verify it and move on.
>
> Method note, because it is how the discrepancy was found: **RTK rewrites `grep` and mangles its
> output, and `grep -c` counts matching LINES anyway** — which is almost certainly where the original
> numbers came from. Count with Python (`re.findall` for occurrences, a per-line `search` for lines),
> never with `grep -c | wc -l`.

> **Deviation from the spec:** `evals/scripts/eval_splats.py:189-211` also reads `splats.zarr`. The spec's reader list misses it; it is fixed here.

> **CORRECTED 2026-09-06 — Step 10's expected output is stale in three places, the header's hit
> total contradicts its own file list, one file with a hit is missing from that list, and this task
> is NOT safely parallel with Task 17. Re-measured at `e5d97734`.**
>
> **1. THIS TASK DEPENDS ON TASK 17 — do not run them in parallel.** Step 4 rewrites
> `_run_tsdf_mesh` to pass `splats_ckpt` into `_splats_to_tsdf_inputs`, and Step 9's gate runs
> `tests/mesh`. Until Task 17 lands, `_splats_to_tsdf_inputs`'s first parameter is still
> `splats_zarr` and its body still calls `zarr.open_group`, so Step 4's change makes Step 9 fail on
> work that is not yours. The two tasks touch **disjoint files**, which is what made them look
> parallelisable, but they are coupled through that one function signature. **Task 17 first.**
>
> **2. Step 10's expected-output block is wrong on both lines.** It expects:
>
> ```
> ./tests/splats/test_rendering.py:339:    assert not (tmp_path / "splats.zarr").exists()
> ./tests/splats/test_trainer.py:256:    assert not (out_dir / "splats.zarr").exists()
> ```
>
> Measured, the two guards are at **`test_rendering.py:378`** and **`test_trainer.py:289`**, and
> `test_trainer.py`'s reads **`tmp_path`, not `out_dir`**. Both line numbers and one variable name
> are stale. An implementer diffing the grep output literally against that block would report a
> mismatch that is not one. **The guards themselves are intact and must not be deleted** — that part
> of the step stands.
>
> **3. The header says "Nine files, 47 tracked hits"; its own file list sums to 56**
> (15 + 11 + 2 + 10 + 1 + 2 + 1 + 6 + 8). Every individual per-file count in that list is still
> exactly right at `e5d97734` — `reconstructor.py` 15, `analyze_splats.py` 11, `eval_splats.py` 2,
> `test_splats_stage.py` 10, `docs/splats.md` 1, `configs/README.md` 2,
> `docs/known-test-failures.md` 1, nb03 6, nb06 8 — so it is the total that is wrong, not the
> inventory. Task 17's two files also still measure exactly as stated: `mesh/utils.py` 8,
> `tests/mesh/test_splats_adapter.py` 29.
>
> **4. `CLAUDE.md` carries a hit and is not in the Files list.** `CLAUDE.md:29`, a dated
> "Recently completed (2026-08-22)" line. It is **history, not a reader** — the likely disposition
> is to leave it, exactly as `docs/superpowers/` is left. But Step 10's grep is
> `--include='*.py' --include='*.ipynb'` and so cannot see it, and the step's "check them by eye"
> list names only `configs/README.md`, `docs/splats.md` and `docs/known-test-failures.md`. Add
> `CLAUDE.md` to that eyeball list and record the decision rather than leaving it unexamined.
>
> **5. Step 8 under-specifies `configs/README.md`.** It says to delete the processed-scene layout
> row. There are **two** hits: `:560` is that layout row, and **`:367` is the `mesh.source`
> parameter table**, whose description reads "`splats` fuses splats.zarr renders". Both need
> editing.
>
> **Confirmed still true at `e5d97734`:** `docs/known-test-failures.md:5-6` does assert "the splats
> cleanup does not touch `reconstructor.py`, `configs/base.yaml`, or this test file", which this
> task falsifies for two of the three — fix that sentence as the existing note says, and leave the
> `configs/base.yaml` clause alone, since this task does not touch it. Line 9 is the stale test
> name. No `splats.zarr` reference exists in `configs/base.yaml` or any other `*.yaml`.

> **CORRECTED 2026-09-06 at `dae59f89` — three BLOCKING items, and the header's own account of
> this task's three "pre-existing" failures is wrong about who owns them.** Every citation in the
> task body below was re-verified live at `dae59f89` and, unusually, **nothing had drifted** — the
> `_run_tsdf_mesh` sites (599 / 612 / 617 / 633 / 644), the `mesh()` sites (1565 / 1591-1598 / 1632),
> the `splats()` sites (1755 / 1758 / 1761 / 1845), the `_stage_output_exists` marker (1909),
> `depth_vs_gt` (189 / 193), `analyze_normals` (96 / 103 / 269 / 30), the four
> `test_splats_stage.py` sites (158-170 / 180-186 / 268 / 282-298) and every per-file hit count
> (15 / 11 / 2 / 10 / 1 / 2 / 1 / 6 / 8) are all exact. Navigate by symbol anyway.
>
> **T18-1 (BLOCKING) — the three `KeyError: 'splat_max_depth_frac'` failures belong to THIS task,
> not to the mesh owner, and Step 9's `PYTEST_RC=0` is reachable only once they are fixed.**
> `docs/known-test-failures.md` §1 attributes them to `configs/base.yaml` drift. Measured:
> `configs/base.yaml`'s `mesh:` block **does** carry `splat_max_depth_frac: null` and
> `splat_max_depth_grad: null`. The stale thing is `_stub_reconstructor`'s literal config dict in
> `tests/wrapper/test_splats_stage.py`, whose `"mesh"` entry lists `splat_depth` but omits both
> `splat_max_depth_*` keys, so `mesh()` raises at `reconstructor.py:1628`. Both the stub gap and the
> read existed identically at the branch base `6f060dfa`, so "pre-existing, not caused by the splats
> cleanup" is **true** — the attribution to base.yaml and to the mesh owner is **false**. The file is
> this task's own; adding the two keys closes all three. Do not touch `configs/base.yaml`.
>
> **T18-2 (BLOCKING) — Step 5's skip-check snippet breaks a currently-passing test.** It replaces
> `self._stage_output_exists("splats")` with a direct `splats_ckpt.exists()`, but
> `test_splats_stage_skips_when_output_exists` monkeypatches `_stage_output_exists` and asserts
> `train` is never called. Keep the existing structure and rebind the path only; the marker change
> belongs in `_stage_output_exists`, which is what Step 2's second test pins.
>
> **T18-3 (BLOCKING) — Step 11's `black` / `isort` calls are missing their flags.** The venv's black
> is newer than the repo pin and defaults to 88 columns. Use
> `black --target-version py311 --line-length 120` and `isort --profile black --line-length 88`, on
> the four touched `.py` files only — never the notebooks, never repo-wide.
>
> **T18-4 — the checkpoint fixture already exists.** Task 17 built `_write_ckpt`,
> `_fake_render_views` and `_patch_renders` in `tests/mesh/test_splats_adapter.py`. Copy the pattern
> into `tests/wrapper/test_splats_stage.py`; do not edit Task 17's file and do not import across test
> packages. `_splats_to_tsdf_inputs` imports `render_views` at call time, which is why
> `monkeypatch.setattr("collab_splats.splats.rendering.render_views", ...)` works. Only two of the
> four affected tests need a renderable checkpoint —
> `test_mesh_stage_forwards_splat_depth_from_the_config` patches `_run_tsdf_mesh` wholesale and needs
> nothing but an existing `ckpt.pt` file.
>
> **T18-5 — `evals/scripts/analyze_splats.py` needs two edits Step 6 does not mention:** drop
> `import zarr` (its only `zarr.` use is the `open_group` Step 6 deletes, so it becomes an F401), and
> rewrite the **module** docstring lines 4-6, which still say "in `<results>/<prim>/splats.zarr`" and
> "the **stored** normal". Keep the `_splats_to_tsdf_inputs` import and the `--zarr` argument — that
> one is `pointcloud.zarr`.
>
> **T18-6 — `tests/wrapper/test_splats_stage.py` KEEPS `import zarr`.** Measured: `zarr.open_group`
> is still used four times for `pointcloud.zarr` fixtures, and `np.` appears 15 times. Only
> `eval_splats.py`'s `import zarr` actually becomes unused.
>
> **T18-7 — Step 8 cannot execute the notebooks in this worktree.** `data/outputs/` does not exist
> here; the notebooks read a scene under a different tree. Take the documented fallback: change the
> code cells, clear the outputs, and say in the commit message that they are unexecuted. Never
> fabricate outputs. Related: the plan's notebook counts **include cell outputs** — nb03 is 5 source
> lines + 1 output line = 6, nb06 is 7 + 1 = 8. A source-only count finds 5 and 7 and looks like
> drift; it is not.
>
> **T18-8 — Step 8's "Task 16 removed the mentions" is not yet true.** `docs/splats.md` is 100 lines
> and line 85 still documents the store. If Task 16 has not landed when this task runs, fix it here
> rather than assuming.
>
> **T18-9 — `docs/known-test-failures.md` gets CLOSED, not amended.** Lines 3, 5-6, 9, 12 and 15 all
> need work: the `(OPEN, mesh owner)` header and the cause, the "does not touch `reconstructor.py` …
> or this test file" sentence (falsified for two of its three clauses; leave the `configs/base.yaml`
> clause, which stays true), the stale test name, the `3 failed, 13 passed` count, and the
> `reconstructor.py:1628` citation.
>
> **T18-10 — Step 10 may legitimately return a THIRD line.** The two Task 10 guards measure at
> `tests/splats/test_rendering.py:378` and `tests/splats/test_trainer.py:289`, both reading
> `tmp_path`. A third hit survives in `collab_splats/mesh/utils.py`, in `mesh_from_tsdf_inputs`'s
> docstring: `- Shared tail of both input adapters (pointcloud.zarr and splats.zarr).` — a Task 17
> leftover in a Task 17 file, reported to that review. If it is still present, report it; do not fix
> it. Run the sweep in Python, not grep: at `dae59f89` it returns 30 lines.
>
> **T18-11 — `CLAUDE.md:29` is history; leave it and record the decision.** It is a dated
> "Recently completed (2026-08-22)" entry, the same disposition as `docs/superpowers/`.
>
> **T18-12 — `configs/README.md` has two hits and Step 8 names only one:** `:367` is the `mesh.source`
> parameter table ("`splats` fuses splats.zarr renders"), `:560` is the processed-scene layout row.
>
> **T18-13 — Step 9's control, measured at `dae59f89` with the proof line and 0 skips:**
> `tests/wrapper/test_splats_stage.py tests/mesh` gives **4 failed, 86 passed, PYTEST_RC=1** in
> 31.24s. `tests/mesh` is entirely green (74 of those passes). Three failures are T18-1's; the fourth,
> `test_run_tsdf_mesh_forwards_splat_depth_to_the_adapter`, raises `IsADirectoryError` on
> `.../splats.zarr` — Task 17 moved the adapter to a checkpoint while the caller still passes a zarr
> directory, which is this task's core. After the task: **0 failed, 90 passed, 0 skipped, RC=0.**
>
> **T18-14 — Step 4's deletion of the `n_views != n_poses` check removes the only read of
> `result.extrinsics` on the splats path.** `result` stays, used by the feedforward branch;
> `test_run_tsdf_mesh_forwards_splat_depth_to_the_adapter`'s `result=SimpleNamespace(extrinsics=...)`
> becomes inert but harmless.

- [ ] **Step 1: Find every remaining reader**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "splats\.zarr\|splats_zarr" --include='*.py' --include='*.ipynb' --include='*.md' --include='*.yaml' . \
  | grep -v "^\./docs/superpowers/" | grep -v "^\./\.git/"
```

Work the list this prints. Three groups of hits are **expected and are not yours**:

- `collab_splats/mesh/utils.py` (8) and `tests/mesh/test_splats_adapter.py` (29) — **Task 17's**.
  If Task 17 has not landed yet these will still be here; leave them.
- `tests/splats/test_rendering.py:339` and `tests/splats/test_trainer.py:256` — deliberate
  `assert not (... / "splats.zarr").exists()` **guards** landed by Task 10. They pin the
  retirement. Never delete them.
- `docs/superpowers/` — history, already filtered out by the grep above.

Anything else is a reader this plan missed, and it needs the same treatment.

- [ ] **Step 2: Write the failing test**

In `tests/wrapper/test_splats_stage.py`, the stage's return value is asserted. Change it:

```python
def test_run_splats_returns_the_checkpoint_path(tmp_path, monkeypatch):
    """
    The stage's artifact of record is the checkpoint — splats.zarr no longer exists.
    """
    reconstructor = _reconstructor(tmp_path)

    def fake_train(cfg, images, world_to_cam, intrinsics, points, colors, out_dir, depth_targets=None, **kwargs):
        (Path(out_dir) / "ckpt.pt").write_bytes(b"")
        (Path(out_dir) / "splats.ply").write_bytes(b"")

    monkeypatch.setattr("collab_splats.splats.trainer.train", fake_train)

    out = reconstructor.splats()

    assert out.name == "ckpt.pt"
    assert out.exists()


def test_splats_stage_output_marker_is_the_checkpoint(tmp_path):
    reconstructor = _reconstructor(tmp_path)
    splats_dir = reconstructor.backend_dir / "splats"
    splats_dir.mkdir(parents=True)

    assert not reconstructor._stage_output_exists("splats")
    (splats_dir / "ckpt.pt").write_bytes(b"")
    assert reconstructor._stage_output_exists("splats")
```

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_splats_stage.py -q
```

Expected: both fail — the stage still returns and marks the zarr path.

**These two tests are not the whole file.** Eight more `splats.zarr` hits live in
`tests/wrapper/test_splats_stage.py` and every one of them has to move:

- `_write_minimal_splats_zarr` (**158-170**) — the fixture helper that builds a store. It becomes a
  helper that writes a minimal `ckpt.pt`, or it goes away entirely if the tests that call it can
  take the real checkpoint fixture.
- `test_mesh_source_splats_fuses_from_splats_zarr` (**180-186**) — rename and rewire. This is one
  of the three tests already failing at HEAD on `KeyError: 'splat_max_depth_frac'` (pre-existing
  `base.yaml` drift, not yours) — do not mistake that for a regression you caused.
- the call at **268**.
- `test_run_tsdf_mesh_forwards_splat_depth_to_the_adapter` (**282-298**) — it passes
  `splats_zarr=` straight into `_run_tsdf_mesh`, so Step 4's parameter rename breaks line 298
  outright.

The inventory table already says "zarr fixture -> ckpt fixture" for this file; that is the whole
job, not just the two tests above.

- [ ] **Step 3: Reconstructor site 1 — the mesh precondition (`mesh()`, measured at `8f4bfc37`: docstring 1565, precondition 1591-1598, call kwarg 1632)**

```python
        pointcloud_zarr = self.pointcloud_zarr
        splats_ckpt = None
        if source == "splats":
            splats_ckpt = self.backend_dir / "splats" / "ckpt.pt"
            if not splats_ckpt.exists():
                raise ValueError(
                    f"mesh.source: splats needs {splats_ckpt} — run the splats stage first "
                    "(it is never auto-run)"
                )
```

and the call at the bottom of the same method: `splats_zarr=splats_zarr,` becomes `splats_ckpt=splats_ckpt,`.

- [ ] **Step 4: Reconstructor site 2 — `_run_tsdf_mesh` (measured at `8f4bfc37`: `def` 599, param 612, docstring 617, call 633, frame-count message 644)**

Rename the parameter and delete the frame-count cross-check:

```python
    splats_ckpt: Path | None = None,
```

```python
        depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(
            splats_ckpt,
            conf_percentile=conf_percentile,
            splat_depth=splat_depth,
            max_depth_frac=splat_max_depth_frac,
            max_depth_grad=splat_max_depth_grad,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
```

The `n_views != n_poses` check goes: it compared the store's view count against the COLMAP reconstruction's, guarding against a stale store from a different run. The checkpoint carries its own cameras and the fusion uses those, so a mismatched COLMAP model is no longer something the fusion can be wrong about. Update the function's docstring line accordingly:

```python
    """Fuse depth + RGB from pointcloud.zarr (or renders from a splats ckpt.pt) into a TSDF mesh."""
```

- [ ] **Step 5: Reconstructor sites 3 and 4 — the stage's return and marker (measured at `8f4bfc37`: `splats()` docstring 1755, bind 1758, skip-return 1761, final return **1845**, `_stage_output_exists` **1909**)**

Note the spread: `splats()`'s final `return splats_zarr` sits ~380 lines below the first site, and
the docstring at 1755 also names the store. Do not stop at the first two hits.

```python
        splats_ckpt = out_dir / "ckpt.pt"
        if splats_ckpt.exists() and not overwrite:
            logger.info("Splats already trained at %s", splats_ckpt)
            return splats_ckpt
```

and the final `return splats_ckpt`. Then check `_stage_output_exists` for a `splats` entry naming the zarr and point it at `ckpt.pt`:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -n "splats" collab_splats/wrapper/reconstructor.py | grep -i "stage_output\|marker\|exists"
```

- [ ] **Step 6: `evals/scripts/analyze_splats.py`**

`analyze_normals(splats_zarr, primitive)` becomes `analyze_normals(ckpt_path, primitive)` and renders instead of reading:

```python
def analyze_normals(ckpt_path: Path, primitive: str) -> dict:
    """
    Table 1 statistics for one primitive's checkpoint, rendered and scored one view at a time.

    - Unit-norm fraction of the rendered normal (and after alpha division, meaningful for 2DGS).
    - Mean/median angle vs depth normals, raw and sign-folded (min(angle, 180 - angle)).
    """
    model, camera_opt, cam_to_world, intrinsics_all, image_ids, (height, width) = load_checkpoint(
        ckpt_path, "cuda"
    )
    n_views = len(image_ids)

    n_pixels = 0
    n_unit = 0
    n_unit_alpha = 0
    angle_sum = 0.0
    folded_sum = 0.0
    angle_chunks = []
    folded_chunks = []

    renders = render_views(model, camera_opt, cam_to_world, intrinsics_all, height, width)
    for view, render in enumerate(renders):
        depth = render["depth"][0, ..., 0].cpu().numpy().astype(np.float64)
        alpha = render["alpha"][0, ..., 0].cpu().numpy().astype(np.float64)
        normal = render["normal"][0].cpu().numpy().astype(np.float64)
        intrinsics = intrinsics_all[view].cpu().numpy().astype(np.float64)

        # Opaque pixels only; depth normals are undefined on the one-pixel border
        mask = alpha > ALPHA_MIN
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
```

Everything from `# Norm sanity, raw and alpha-divided` to the end of the function is unchanged:
it already works from the `depth` / `alpha` / `normal` / `intrinsics` locals the loop head now
binds from a render instead of from a store. Delete `store = zarr.open_group(...)`, the
`n_views = store["depth"].shape[0]` line and the `intrinsics_all = store["K"][:]` line, which
the block above replaces. Then update the discovery block near line 269:

```python
        ckpt_path = args.results / primitive / "ckpt.pt"
        if ckpt_path.exists():
            available.append((primitive, ckpt_path))
        else:
            logger.warning("Skipping %s: %s missing", primitive, ckpt_path)
```

and the import at line 30. **Keep the `_splats_to_tsdf_inputs` import** — it does *not* become
unused, `analyze_splats.py:289` still calls it for the mesh comparison. Add:

```python
from collab_splats.splats.rendering import load_checkpoint, render_views
```

- [ ] **Step 7: `evals/scripts/eval_splats.py` (`depth_vs_gt` at 189, the store open at 193; the module docstring at 7 also names the store)**

`depth_vs_gt` opens the store for its `depth` and `alpha` stacks. It needs them stacked (the
median alignment is global across all frames), so unlike the mesh adapter this one renders into
arrays and keeps them:

```python
def depth_vs_gt(out_dir: Path, seq: Path) -> dict:
    """
    Rendered-depth error against 7-Scenes GT where alpha > ALPHA_THRESHOLD, after median alignment.
    """
    # Render rather than read a stored render: ckpt.pt is the only artifact the stage writes
    model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width) = load_checkpoint(
        Path(out_dir) / "ckpt.pt", "cuda"
    )
    n_frames = len(image_ids)
    depth = np.empty((n_frames, height, width), dtype=np.float32)
    alpha = np.empty((n_frames, height, width), dtype=np.float32)
    renders = render_views(model, camera_opt, cam_to_world, intrinsics, height, width)
    for view, render in enumerate(renders):
        depth[view] = render["depth"][0, ..., 0].cpu().numpy()
        alpha[view] = render["alpha"][0, ..., 0].cpu().numpy()
```

Everything from `# GT on the rendered pixel grid` to the `return stats` is unchanged — it reads
`depth`, `alpha` and `n_frames`, all of which the block above binds. Add the import at the top
of the file:

```python
from collab_splats.splats.rendering import load_checkpoint, render_views
```

and drop `import zarr` if nothing else in the file uses it.

- [ ] **Step 8: Docs and notebooks**

- `docs/splats.md`: Task 16 removed the mentions; re-run the grep to confirm.
- `configs/README.md`: the processed-scene layout table lists `splats/splats.zarr`. Delete that row and note the change under the layout table:

```markdown
`splats/splats.zarr` was retired on 2026-09-06 — `splats/ckpt.pt` is self-contained (model,
cameras, image size, config) and `collab_splats.splats.rendering.load_checkpoint` +
`render_views` reproduce every render. Scenes processed before that date still have the store;
nothing reads it, and it can be deleted.
```

- Both notebooks: replace the `zarr.open_group(... "splats.zarr")` cells with `load_checkpoint` + `render_views`. Then re-run them headless so the outputs match the code:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    docs/source/tutorials/03_splats/train_splats.ipynb \
    docs/source/tutorials/06_mesh/splats_mesh.ipynb
```

If a notebook needs a scene that is not on this machine, do not fake the outputs: change the code cells, clear the outputs (`--ClearOutputPreprocessor.enabled=True`), and note in the commit message that they are unexecuted.

- [ ] **Step 9: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_splats_stage.py tests/mesh -q
echo "PYTEST_RC=$?"
```

Expected: `PYTEST_RC=0`.

- [ ] **Step 10: Confirm nothing reads the store**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "splats\.zarr" --include='*.py' --include='*.ipynb' . | grep -v "^\./docs/superpowers/"
```

**Expected output is exactly these two lines — not an empty result:**

```
./tests/splats/test_rendering.py:339:    assert not (tmp_path / "splats.zarr").exists()
./tests/splats/test_trainer.py:256:    assert not (out_dir / "splats.zarr").exists()
```

Those are Task 10's deliberate guards; they are what *pins* the retirement, and deleting them to
make the grep come back empty would remove the only regression test for it. The earlier version of
this step expected `NO READERS LEFT` and would therefore have failed forever — the `|| echo` can
never fire while the guards exist.

`configs/README.md`, `docs/splats.md` and `docs/known-test-failures.md` are `*.md` and are not
matched by this grep — check them by eye. Mentions in `docs/superpowers/` (this plan, the spec,
the CHANGELOG) are history and stay.

- [ ] **Step 11: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  /opt/venv/reconstruction/bin/black collab_splats/wrapper/reconstructor.py evals/scripts/analyze_splats.py \
    evals/scripts/eval_splats.py tests/wrapper/test_splats_stage.py && \
  /opt/venv/reconstruction/bin/isort collab_splats/wrapper/reconstructor.py evals/scripts/analyze_splats.py \
    evals/scripts/eval_splats.py tests/wrapper/test_splats_stage.py && \
  git commit --only collab_splats/wrapper/reconstructor.py evals/scripts/analyze_splats.py \
    evals/scripts/eval_splats.py tests/wrapper/test_splats_stage.py docs/splats.md configs/README.md \
    docs/known-test-failures.md \
    docs/source/tutorials/03_splats/train_splats.ipynb docs/source/tutorials/06_mesh/splats_mesh.ipynb \
    -m "refactor(splats): move every splats.zarr reader onto ckpt.pt"
```

---

## Task 19: Final parity and the full gate

The last task. Two things have to be true before this refactor is done: the numbers are still the numbers, and every test in the tree that touches splats or its consumers passes.

**Files:**
- Modify: `/tmp/claude-0/-workspace-collab-splats/scratchpad/parity_final.json` (scratchpad, not committed)
- Modify: `CLAUDE.md`

> **CORRECTION PASS T19-1..T19-7 — seven measured errors in this task's steps, taken at `225abd6f`.**
> Three of them are steps whose stated expectation is unreachable, and two of those fail *silently*.
> Verify every one against the tree before trusting it; a correction block is itself a claim.
>
> **T19-1 (BLOCKING — Step 5 fails silently). The baseline SHA is wrong.**
> `git merge-base HEAD main` is **`bfeefc5e`**, dated **2025-09-19**, and
> `collab_splats/splats/` **does not exist in it** — `git show bfeefc5e:collab_splats/splats/`
> returns `fatal: path 'collab_splats/splats/' exists on disk, but not in 'bfeefc5e'`, which
> Step 5's own `2>/dev/null` swallows. The `while read` loop then iterates over nothing and
> `paste -sd+ | bc` prints an empty line, so the step reports an "after" number with **no
> "before" number at all** and an inattentive reader records "smaller". The branch's fork point
> is **`6f060dfa`** (verified: `git merge-base --is-ancestor 6f060dfa HEAD` → true), where
> `splats/` holds nine files. Use `6f060dfa`, not the merge-base.
>
> **T19-2. The after total is LARGER, and Step 5's "Expected" line says it will be smaller.**
> Measured at `225abd6f`, counting the same way on both sides:
>
> | before (`6f060dfa`) | | after (`225abd6f`) | |
> |---|---|---|---|
> | trainer.py | 843 | scaffold.py | 1122 |
> | scaffold.py | 774 | pgsr.py | 580 |
> | pgsr.py | 484 | losses.py | 498 |
> | losses.py | 343 | rendering.py | 465 |
> | outputs.py | 335 | trainer.py | 415 |
> | rendering.py | 264 | gaussian.py | 411 |
> | cameras.py | 88 | cameras.py | 218 |
> | appearance.py | 28 | utils.py | 174 |
> | __init__.py | 10 | __init__.py | 20 |
> | **TOTAL** | **3169** | **TOTAL** | **3903** |
>
> **+734 lines, +23%.** `outputs.py` and `appearance.py` are gone; `gaussian.py` and `utils.py`
> are new. Step 5's escape hatch ("not automatically a failure — report the split per file")
> is the branch that applies, so take it. Tasks 15-18 are still outstanding and Task 15 is a
> **docstring sweep**, so expect the total to grow further. The consequence for Task 16's
> CHANGELOG line: **do not write a raw-LOC shrink claim** — it is false. If a size claim goes in
> at all it must be AST LOC or the per-file split, and the honest headline is that two modules
> were retired and the branch logic they carried moved into the two representation classes.
>
> **T19-3 (BLOCKING — Step 3 would mask a regression). The first "pre-existing failure" is
> STALE.** Step 3 tells the implementer to accept `tests/test_cu121_migration.py` collecting
> with `1 error` and `No module named 'gsplat.losses'`. Measured now: the shared venv holds
> gsplat **1.5.3**, `import gsplat.losses` succeeds, and `tests/test_cu121_migration.py`
> **passes**. **Delete that bullet.** A collection error there today is a real regression, not
> the known venv skew, and the step as written instructs you to wave it through.
>
> **T19-4. Step 3's gate is red at HEAD, so `PYTEST_RC=0` is unreachable as stated.** Measured at
> `225abd6f`: `pytest tests/wrapper/test_splats_stage.py tests/test_cu121_migration.py -q
> -p no:randomly` → **3 failed / 34 passed, RC=1**. The three are
> `test_mesh_source_splats_fuses_from_splats_zarr`, `test_mesh_sfm_aligned_zarr_fuses` and
> `test_mesh_stage_forwards_splat_depth_from_the_config`, all raising
> `KeyError: 'splat_max_depth_frac'` at `collab_splats/wrapper/reconstructor.py:1628`. They are
> documented at `docs/known-test-failures.md:3` with the one-line fix, and they belong to the
> mesh/wrapper owner: this branch never touches `reconstructor.py` —
> `git diff --name-only 6f060dfa HEAD` outside `collab_splats/splats/`, `tests/splats/` and
> `docs/superpowers/` is exactly `configs/README.md`, `configs/base.yaml`,
> `docs/known-test-failures.md`, `pyproject.toml`, `setup.sh`, `tests/test_cu121_migration.py`,
> `tests/wrapper/test_splats_stage.py`. Step 3's generic "anything in known-test-failures"
> clause does cover them, but **name the three and state RC=1 as the expected result**, so that
> a fourth failure is visible instead of being absorbed into a vague allowance.
>
> **T19-5. Step 4 returns four hits at `225abd6f`, and its one stated exemption does not work.**
> Measured with a Python regex sweep over tracked `*.py` / `*.ipynb` outside
> `docs/superpowers/` (RTK mangles `grep`, so do not count with `grep -c`):
>
> | pattern | hits |
> |---|---|
> | `splats\.outputs`, `splats import outputs`, `activate_vanilla`, `AnchorField`, `AppearanceModule` | 0 each |
> | `render_view\b` | 1 — `tests/splats/test_gaussian.py:272` |
> | `SH_DC_NORMALIZER` | 1 — `collab_splats/splats/gaussian.py:24` |
> | `CameraOptModule` | 1 — `collab_splats/splats/cameras.py:6` |
> | `splats\.appearance` | 1 — `collab_splats/splats/scaffold.py:66` |
>
> - The `grep -v "upstream name"` filter is there to spare the gsplat citation at
>   `cameras.py:6`, but **that line does not contain the string "upstream name"** — it reads
>   ``` ``CameraOptModule``) — ``examples/`` is not shipped in the gsplat wheel ```. The filter
>   matches nothing and the citation is reported as a stale reference.
> - `splats\.appearance` matches the **config key** `splats.appearance_opt` at `scaffold.py:66`,
>   which is live configuration, not a retired module path.
> - `gaussian.py:24` is a comment that names `SH_DC_NORMALIZER` in order to record that it was
>   retired. A grep for a retired name will always hit the note explaining the retirement.
> - **One hit is a genuine stale reference:** `tests/splats/test_gaussian.py:272` names
>   `rendering.render_view`, singular. `rendering.py`'s top-level defs at HEAD are
>   `gaussian_normals_in_camera_frame`, `render_gaussians`, `render_views`, `write_outputs`,
>   `load_checkpoint` — there is no `render_view`. **Fix it in Task 15's comment sweep**, so
>   Step 4 comes back clean here.
>
> So Step 4's `NO STALE REFERENCES` is not reachable without adjudicating four lines. Rewrite the
> exemptions against the text that is actually on those lines, or drop the `||` and require the
> implementer to account for each hit by name.
>
> **T19-6. Step 6's size check already fails, before this refactor adds anything.**
> `wc -c CLAUDE.md` at `225abd6f` is **41745**; the step expects "well under 40000". The excess
> is other subsystems' inline "Recently completed" prose, which trunk has already migrated to
> `docs/superpowers/CHANGELOG.md` — this branch's `CLAUDE.md` is simply stale relative to trunk,
> so it is a rebase concern and **not this plan's to fix**. Report the number and move on; do not
> trim another owner's entries to get under the bar. Separately, `grep -c "splats-cleanup"
> CLAUDE.md` is **0**, so Step 6's deletion half is a no-op and will print
> `NOT LISTED — nothing to remove`.
>
> **T19-7. Step 2's prose and its command name different baselines, and the prose is wrong.**
> The command is `--compare before_v3 final`; the prose immediately below says "Compare against
> `before`, never against `after_phase4`". `parity_before.json` is the **old three-config,
> 2.5 KB** fingerprint format; `parity_before_v3.json` is the current 5.9 KB one. Follow the
> command, not the prose. The harness's argument shape is confirmed:
> `splats_parity.py` declares `label` as `nargs="?"` and `--compare` as `nargs=2`, so the
> command is valid as written. The scratchpad root in Steps 1-2 is the **shared**
> `/tmp/claude-0/-workspace-collab-splats/scratchpad`, which is correct — not a session-UUID one.

> **CORRECTION PASS T19-8..T19-10 — three further errors, measured at `25d2fed0`, after Tasks 12,
> 14, 17 and 18 landed. Two of T19-1..T19-7 have themselves gone stale; both are named below.**
>
> **T19-8. Steps 1 and 2 expect THREE configs. The harness has SEVEN.**
> `CONFIGS` in `splats_parity.py` is `3dgs_vanilla`, `2dgs_vanilla`, `3dgs_scaffold`,
> `3dgs_norm_nopose`, `scaffold_norm_nopose`, `3dgs_appearance`, `3dgs_pgsr`, and
> `parity_before_v3.json` holds all seven. Step 1's "three `psnr=` lines" and Step 2's three-line
> `Expected` block both date from the retired three-config harness — the same rebuild that moved
> `STEPS` from 300 to 50. **Seven `psnr=` lines is the pass; three means four configs were silently
> skipped.**
>
> **T19-9 (supersedes T19-4). The gate is GREEN at HEAD, so `PYTEST_RC=0` is now the bar.**
> T19-4 recorded three `KeyError: 'splat_max_depth_frac'` failures in
> `tests/wrapper/test_splats_stage.py` and instructed the implementer to expect RC=1. **Task 18
> fixed them.** At `25d2fed0`, `docs/known-test-failures.md:3` reads `## 2026-09-06 — RESOLVED:`
> for that entry, and `reconstructor.py` declares `splat_max_depth_frac` at its signature and reads
> it from `mesh_cfg`. Combined with T19-3 (the `gsplat.losses` allowance was already stale), **Step 3
> now has no live pre-existing-failure allowance at all.** Read `docs/known-test-failures.md` at the
> time of the run and enumerate whatever it claims is open; do not carry a generic "anything in
> known-test-failures" clause, which is how a fourth failure gets absorbed into a vague exemption.
>
> **T19-10 (BLOCKING for the Done-When check). The first Done-When bullet names a module that has
> never existed.** It lists nine modules including **`strategy.py`**. That string appears exactly
> once in this entire plan — in that bullet — and `git log --all -- collab_splats/splats/strategy.py`
> returns nothing on any branch. The real ninth module is **`utils.py`**, created by Task 6 of this
> plan. The bullet's "no `anchors.py`" is likewise a phantom: `anchors.py` never existed; the two
> modules actually retired are `outputs.py` and `appearance.py`. The correct list is `__init__.py`,
> `cameras.py`, `gaussian.py`, `losses.py`, `pgsr.py`, `rendering.py`, `scaffold.py`, `trainer.py`,
> `utils.py`.
>
> **Two updates to the standing measurements, both re-taken at `25d2fed0`:**
>
> - **Step 4's sweep is down to three hits, all benign, and `render_view` is now clean.** T19-5's one
>   genuine stale reference (`tests/splats/test_gaussian.py` naming `rendering.render_view`,
>   singular) was removed by Task 15's comment sweep — `render_view\b` returns **0** across 337
>   tracked `*.py`/`*.ipynb` outside `docs/superpowers/`. What remains is `SH_DC_NORMALIZER` in
>   `gaussian.py` (a comment recording the retirement), `CameraOptModule` in `cameras.py` (the gsplat
>   citation that must survive, and which the step's `grep -v "upstream name"` filter still fails to
>   spare, because that line does not contain the string), and `splats.appearance` in `scaffold.py`
>   (the live config key `splats.appearance_opt`). Three hits to adjudicate by name, none to fix.
> - **Step 5's totals have grown again, and the growth survives docstring-stripping.** Measured
>   `6f060dfa` → `25d2fed0`: raw **3169 → 3986** (+817, +26%), AST LOC **2823 → 3566** (+743, +26%).
>   Per file after: `scaffold.py` 1143/1090, `pgsr.py` 593/500, `losses.py` 507/434, `rendering.py`
>   469/419, `gaussian.py` 428/390, `trainer.py` 420/374, `cameras.py` 230/206, `utils.py` 176/140,
>   `__init__.py` 20/13. So the T19-2 instruction stands and hardens: **no LOC-shrink claim may ship
>   in the CHANGELOG entry on either measure.** The honest headline is that two modules were retired
>   and the branch logic they carried moved into the two representation classes.
>
> One more hazard the steps carry: **Step 7 runs `git add -f` on two already-tracked paths.** `-f`
> only overrides gitignore; on a tracked path it does nothing but widen the blast radius against a
> shared index. Drop the `git add` line and let `--only` do the work. Step 1's command also omits the
> `cd` into the worktree, which would fingerprint the **main tree** — `PYTHONPATH` alone is not
> enough, because `sys.path[0]` is the cwd.

> **CORRECTION PASS T19-11..T19-15 — five errors found while EXECUTING Task 19, measured at
> `3107ed0f`. Task 19 itself passed: parity 7/7, worst drift 3.42e-07 against a 1e-3 gate; wide gate
> 534 passed / 0 failed / 0 skipped / RC=0; no commit, tree clean. These are the errors it hit on the
> way, recorded under the termination rule.**
>
> **T19-11. Step 1's prose is FALSE: "14 touched only PGSR (no parity config enables a PGSR loss)".**
> `CONFIGS["3dgs_pgsr"]` in `splats_parity.py` sets **both** `pgsr_normal` and `pgsr_multiview` to
> `0.01`. A PGSR config has been in the harness since the seven-config rebuild (T19-8). The step's
> conclusion — run it, it is the proof — is right; its stated reason is the opposite of true, and it
> is exactly the reason someone would use to skip the run. Task 14's changes **are** exercised by the
> harness, which is why this run is load-bearing.
>
> **T19-12. Done-When bullet 2 says "six methods". The pinned interface is five attributes and
> SEVEN methods.** `tests/splats/test_model_interface.py` pins `render`, `pre_backward`,
> `post_backward`, `denormalize`, `export_gaussians`, `frame_report`, `checkpoint` — seven — beside
> `params`, `optimizers`, `schedulers`, `n_primitives`, `primitive_unit`. The CHANGELOG entry already
> says seven; the Done-When bullet is the stale one.
>
> **T19-13. Done-When bullet 6 says "all three configs". The harness has seven** — same stale
> three-config assumption as T19-8, in a second place. Three passing configs means four were skipped.
>
> **T19-14. Done-When bullet 7 omits `tests/test_cu121_migration.py`.** The wide gate this branch
> actually runs is `tests/splats tests/mesh tests/wrapper/test_splats_stage.py
> tests/test_cu121_migration.py` = **534 passed / 0 failed / 0 skipped**, decomposing as
> `tests/splats` **411**, `tests/mesh` **80**, `tests/wrapper/test_splats_stage.py` **21**,
> `tests/test_cu121_migration.py` **22**. Since T19-3 that last file went from a collection error to
> 22 passing tests; leaving it out of the Done-When bar drops the file that proves the venv skew is
> gone.
>
> **T19-15. T19-10's stated reason is wrong, though its instruction survives.** It says the growth
> "survives docstring-stripping". Its own AST-LOC metric is **span-based** — `end_lineno - lineno`
> over the tree — so it still counts comments and docstrings interleaved inside a function body, and
> it therefore tracked raw almost exactly (+26% vs +26%), which is the tell. Measured on a metric that
> genuinely removes both — non-blank lines of `ast.unparse()` over the docstring-stripped tree —
> `collab_splats/splats/` goes **1243 → 1277, +34, +2.7%**, against raw **3169 → 4006, +837, +26%**.
> **So roughly 800 of the 837 added lines are comments and docstrings**, i.e. Task 15's sweep, not
> code. The binding instruction is unchanged and still stands: every measure is up, so **no LOC-shrink
> claim may ship**. But the honest headline is now sharper than T19-2's: **+2.7% executable code, two
> modules retired.**
>
> **One measurement in the T19 brief was REFUTED, and must not be propagated.** The claim that
> `wc -c` disagrees with Python by a constant 356 bytes on `CLAUDE.md` is **false**: `wc -c`,
> `os.stat().st_size` and `len(open(p).read().encode())` all return **41717**, through both the RTK
> proxy and a bare `subprocess`. Step 6's byte count can be taken with any of the three.
>
> **T19-16. The `color` rename is a CHECKPOINT FORMAT CHANGE for Scaffold, not a cosmetic one.**
> `refactor(splats): use US spelling for color throughout` (`2ed799d4`) renamed the `nn.Module`
> attribute `mlp_colour` -> `mlp_color`, so it renamed **state_dict keys**.
> - break: `Scaffold.from_checkpoint` calls `model.mlps.load_state_dict(ckpt["mlps"])` with torch's
>   default `strict=True` (scaffold.py:822)
> - symptom: any Scaffold `ckpt.pt` written before `2ed799d4` now raises `RuntimeError` — unexpected
>   `mlp_colour.*` keys, missing `mlp_color.*`
> - blast radius: Scaffold only; vanilla `Gaussians` saves no `mlps` entry
> - config surface NOT affected: yaml fields are `pose_opt` / `appearance_opt`, already US-spelled
> - judged acceptable — scaffold-gs is in-flight per CLAUDE.md, so no released checkpoint exists —
>   but it must ship in the CHANGELOG as a breaking change, not silently
>
> **A NEW RTK hazard, measured here.** RTK can return the **wrong commit** for `git log --format`:
> `git log -1 --format='%h %ad %s' bfeefc5e` returned `82e09e76 "okay ignoring the notebook again"`,
> while `subprocess` + `git show -s --format=%H|%ad|%s` returned the correct `bfeefc5e`. This is
> beyond the known `grep` / `wc -l` / `diff` mangling: **resolve SHAs through Python, not through the
> shell proxy.**
>
> **Parity is genuine on every recorded field, including the three `compare()` never asserts.**
> `compare()` gates `psnr`, `means_head` and `n_gaussians` only. Hand-checked at `3107ed0f`:
> max |Δmeans_mean| **2.6e-09**, |Δmeans_std| **3.1e-09**, |Δssim| **1.5e-08**. Note what this
> implies for the CHANGELOG wording — `means_head` is the **first five rows**, 2.5-3.1% of exported
> means, so "matched on the means distribution" overstates the gate and has been corrected to "the
> first five exported means".

- [ ] **Step 1: Re-run the three parity configs one last time**

Tasks 14-18 should not have moved anything the harness measures — 14 touched only PGSR (no parity config enables a PGSR loss), 15 touched only comments, and 16-18 touched tests, docs and readers. This run is the proof of that, not a formality: it is the only check that the docstring sweep did not eat a line of code, and the only check that the reader move did not change `write_outputs`.

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -u \
  /tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py final \
  > /tmp/claude-0/-workspace-collab-splats/scratchpad/final.log 2>&1
echo "RC=$?"; tail -5 /tmp/claude-0/-workspace-collab-splats/scratchpad/final.log
```

Expected: `RC=0` and three `psnr=` lines. ~6 minutes on the A40; use `run_in_background: true` rather than a longer timeout if it overruns.

- [ ] **Step 2: Compare against the original baseline, not the last checkpoint**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python \
  /tmp/claude-0/-workspace-collab-splats/scratchpad/splats_parity.py --compare before_v3 final
echo "PARITY_RC=$?"
```

Expected:

```
PASS 3dgs_vanilla: dPSNR=0.00e+00 means_ok=True n_ok=True
PASS 2dgs_vanilla: dPSNR=0.00e+00 means_ok=True n_ok=True
PASS 3dgs_scaffold: dPSNR=0.00e+00 means_ok=True n_ok=True
PARITY OK
PARITY_RC=0
```

Compare against `before`, never against `after_phase4`: three consecutive within-tolerance steps can add up to a drift that exceeds the tolerance, and only the end-to-end comparison catches that.

- [ ] **Step 3: Run the full gate**

Everything the refactor touched, plus the consumers it moved:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-splats \
  /opt/venv/reconstruction/bin/python -u -m pytest \
    tests/splats tests/mesh tests/wrapper/test_splats_stage.py tests/test_cu121_migration.py \
    -q > /tmp/claude-0/-workspace-collab-splats/scratchpad/final_gate.log 2>&1
echo "PYTEST_RC=$?"; tail -20 /tmp/claude-0/-workspace-collab-splats/scratchpad/final_gate.log
```

Expected: `PYTEST_RC=0`.

Two failures are pre-existing and are **not** yours to fix — confirm they look exactly like this and move on:

- `tests/test_cu121_migration.py` collecting with `1 error` and `No module named 'gsplat.losses'` is the shared-venv skew (Ground Rules §3). Re-run that file alone with `--continue-on-collection-errors` and check the rest is green.
- Anything listed in `docs/known-test-failures.md`.

Anything else is a regression from this plan. Do not commit over it.

- [ ] **Step 4: Prove the whole tree still imports**

The refactor deleted two modules and moved a package-level export. A stale import somewhere outside `tests/` would not show up above.

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -rn "splats\.outputs\|splats import outputs\|render_view\b\|activate_vanilla\|AnchorField\|SH_DC_NORMALIZER\|CameraOptModule\|AppearanceModule\|splats\.appearance" \
    --include='*.py' --include='*.ipynb' . \
    | grep -v "^\./docs/superpowers/" | grep -v "upstream name" \
  || echo "NO STALE REFERENCES"
```

Expected: `NO STALE REFERENCES`. Two traps here. `render_view` — `render_views` (plural) is the new name and a substring grep for the old one would match it, so the `\b` is load-bearing. `CameraOptModule` — the second `grep -v` spares the gsplat citation in `cameras.py`'s docstring, which is the one place the upstream name must survive; every other match is a real stale reference, since Task 5 merged both retired classes into `CameraOpt`.

- [ ] **Step 5: Check the line-count claim**

The spec's motivation is that the module is smaller. Measure it rather than assert it:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  echo "--- after ---" && wc -l collab_splats/splats/*.py | tail -1 && \
  echo "--- before ---" && git show $(git merge-base HEAD main):collab_splats/splats/ 2>/dev/null \
    | tail -n +3 | while read -r f; do git show "$(git merge-base HEAD main):collab_splats/splats/$f" | wc -l; done \
    | paste -sd+ | bc
```

Expected: the "after" total is smaller. Record both numbers — they go in the CHANGELOG entry Task 16 wrote, which has a line for them.

If the after total is *larger*, that is not automatically a failure: `gaussian.py` and `scaffold.py` absorbed the trainer's branches and each grew a checkpoint round-trip. Report the split per file so the reviewer can see where it went.

- [ ] **Step 6: Update `CLAUDE.md`'s in-flight list**

The refactor is done, so it leaves the in-flight list. Per the repo's own rule the completed entry goes to `docs/superpowers/CHANGELOG.md` (Task 16 wrote it) and never into `CLAUDE.md`. Add one line to the in-flight block only if the plan is being handed over unfinished; otherwise this step is a deletion:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  grep -n "splats-cleanup" CLAUDE.md || echo "NOT LISTED — nothing to remove"
```

If it is listed, delete that bullet. Then check the size guard the repo enforces:

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && wc -c CLAUDE.md
```

Expected: well under 40000. A larger file is silently truncated by Claude Code and the rules below the cut stop applying.

- [ ] **Step 7: Final commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-splats && \
  git add -f docs/superpowers/plans/2026-09-06-splats-cleanup.md docs/superpowers/CHANGELOG.md && \
  git commit --only docs/superpowers/plans/2026-09-06-splats-cleanup.md docs/superpowers/CHANGELOG.md CLAUDE.md \
    -m "docs(plans): splats cleanup complete — final parity confirmed"
```

- [ ] **Step 8: Report what shipped**

Write the handover in the session, not in a file. It must state:

1. The three parity fingerprints, before and final, with the deltas.
2. The gate result and which failures were pre-existing.
3. The line count before and after, per file.
4. Anything in Ground Rules §11 (the deviations list) that the reviewer needs to accept.
5. The branch name and the commit range: `git log --oneline $(git merge-base HEAD main)..HEAD`.

The worktree stays until the reviewer has merged it. Do not remove it — a live run inside a removed worktree dies with `FileNotFoundError` on chdir, not with a test failure.

---

**Task 14 outcome — fix round 5, `ef37777e`** (one path, +10/−5, `tests/splats/test_pgsr.py`).

Closes the round-4 `sorted` finding. The helper's return is now discriminated against **five**
orderings, not two. Input `"logger = 1\nMIN_DEPTH = 1e-6\nTHETA0 = 5.0\n"` asserting
`["MIN_DEPTH", "THETA0", "logger"]`. Gates flat: `tests/splats` 371/0/0/RC=0, wide 472/0/0/RC=0.

> **INSTANCE TEN, and it was in the dispatch brief itself.**
> The brief prescribed a different three-name input — `"ALPHA = 0\nlogger = 1\nBETA = 5.0\n"` —
> presented with a measurement table showing it kills `[::-1]`. It does. It also **silently loses a
> kill round 4 already had.** Verified independently by the coordinator:
>
> | mutation of the helper's return | brief-prescribed input | shipped input |
> |---|---|---|
> | `list(...)` (source order) | killed | killed |
> | `list(...)[::-1]` | killed | killed |
> | `sorted(reverse=True)` | killed | killed |
> | `sorted(key=str.lower)` | **SURVIVES** | killed |
> | `sorted(key=len)` | killed | killed |
>
> `ALPHA`/`BETA`/`logger` sorts identically under ASCII and under case-folding, so the brief's input
> cannot separate `sorted()` from `sorted(key=str.lower)`. The shipped input can, because `logger`
> case-folds **ahead** of the two upper-case names while ASCII sorts it **behind** them. The
> implementer measured the prescription rather than trusting it, refused it, and shipped an input
> that kills all five. **A coordinator brief is not exempt from the defect class it is written to
> close — and a measurement table in a brief is exactly the thing that makes a prescription look
> already-verified.**
>
> Its own first comment draft then reproduced the class a third time in one round, claiming mixed
> case is what separates the two sorts. Mixed case is necessary but not sufficient — the brief's own
> input is mixed case and the two sorts agree on it. Caught before commit and rewritten to the real
> mechanism plus the counterexample.

Two corrections to the brief, both accepted: `8d88f478` added **+1** test to `tests/splats`, not +4
(collected counts `a1fd1e46` 370 → `8d88f478` 371 → `b6b9c7fe` 371), and the wide-gate absolute moved
471 → 472 with the branch. One property is **knowingly open and deliberately not closed**:
`return sorted(set(...))` survives the full suite at HEAD and with the fix. It is pre-existing, it is
a *multiplicity* property rather than an ordering one, and folding a duplicate-binding case into the
sorted test would conflate two properties in one case. Reported rather than added, following the
tuple-target adjudication precedent.

**Task 17 outcome — fix round 2, `2aa5d25d`** (one path, +18/−9, `tests/mesh/test_splats_adapter.py`).

Closes the one-ended rounding rule. The escape was reproduced first on a pinned tree at `b6b9c7fe`
with `inspect.getsource` proof: `(render["rgb"][0] * 255).ceil().byte()` gave **80 passed, RC=0 —
survives**. The test now runs eighths `m/8` for `m = 1..7` over seven views; since `255 mod 8 == 7`,
the product's fractional part shrinks as `m` grows and crosses a half at `m = 4`, so `m=1..3` round
up, `m=4` is the tie, and `m=5..7` round down. With the fix, `.ceil()` fails at `m=5`, `.floor()` and
bare `.byte()` fail at `m=1`, each 1 failed / 79 passed on the single node id.

**The arms were proven load-bearing, not merely present** — the corollary this branch keeps
relearning. Half-tests: the up half alone lets `.ceil()` survive; the down half alone lets both
`.floor()` and bare `.byte()` survive. Neither arm is redundant.

> **The brief's prescribed rgb value was wrong on its own premise.** It named `0.025` (→ 6.375, whose
> fractional part is in (0, 0.5)). The arithmetic is right, but **`0.025` is not exact in float32**,
> which contradicts the comment's own "exact in float32" framing; eighths are exact and keep it.
> Confirmed: `float32(0.025) != 0.025`, while every `m/8` round-trips exactly.
>
> The implementer's own first comment draft carried instance N+1 and was caught before commit — it
> said *"the tie separates neither (127.5 goes to 128 either way)"*, which is false: `.floor()` and
> bare `.byte()` give 127 at the tie, so the tie **does** separate truncation from rounding and only
> fails to separate `.ceil()`.

Two deliberate non-arms, both recorded rather than silently omitted: `.round().byte()` →
`(x + 0.5).byte()` still survives (killing it would pin torch's half-to-even tie-break, which is not
part of the rule — `torch.tensor([127.5]).round()` is 128.0), and `* 255` → `* 256` was already
killed by `stacks_views_in_render_order` before this change. Three further brief errors: HEAD had
moved one commit further than stated and moved three more times mid-run; the wide-gate control was
469, actually 472; and "Task 18 holds `collab_splats/mesh/utils.py`" was a false premise — Task 18's
nine paths never included it. **The claim that accompanied this — "every other two-ended rule in the
file was swept for the same shape and found already complete" — is FALSE**, and the round-2 reviewer
refuted it by measurement. `keep &= confidence_mask(alpha, conf_percentile)` in
`collab_splats/mesh/utils.py` is the same two-armed shape and **survives** the mutation to
`keep = confidence_mask(...)` (80 passed / 0 failed / RC=0). It is separable, so the gap is real
rather than vacuous: `confidence_mask` uses a strict `>` cutoff and falls back to all-True when
nothing exceeds it, so at `conf_percentile=100` the mutant keeps zero-alpha pixels the shipped code
drops. The two existing tests cannot separate the arms — one passes `conf_percentile=None` (the line
never runs), the other passes 50 (where `above` is a subset of `alpha > 0`, making the arms
identical). The `alpha > 0` arm on its own IS pinned (`alpha >= 0` → 79/1/0 RC=1). Pre-existing,
unchanged by Task 17, recorded as open observation 6.

**Task 15 outcome — fix round 2, `e576ae80`** (three paths, +36/−16, `collab_splats/splats/`
`gaussian.py` / `scaffold.py` / `utils.py`). Docstrings only.

Closes the blocking `view_order` provenance finding and four soft overstatements. The commit is
**gated on AST equality**, independently re-verified by the coordinator: parsing `80efc10e` and
`e576ae80` with every docstring replaced by a sentinel gives identical trees for all three files,
while the raw text differs in all three — so the pass is neither a code change nor a vacuous no-op.
Gates flat both sides: `tests/splats` 371/0/0/RC=0, wide 472/0/0/RC=0, 0 skips.

The blocking finding: `view_order`'s docstring claimed our stream is identical seed for seed to
nerfstudio's. It is not — upstream pops the **front** of the shuffled pool
(`full_images_datamanager.py:396-399` at `50e0e3c`, verified against the pin over the network) while
ours yields from the back. The citation was kept rather than deleted, because our drain-and-refill
loop genuinely is upstream's; only the pop end differs, and naming the divergence is strictly more
informative than hiding it. The "identical seed for seed" claim was re-targeted at the retired
in-repo `ViewSampler`, where it is true — measured equal across **all 36** (n_views × seed)
combinations.

> **INSTANCE ELEVEN, and the second one in a coordinator brief.** The brief's prescribed wording was
> *"state plainly that ours yields the reverse of upstream's `pop(0)` order."* Measured, that is
> **false at stream level and true only per epoch**. Independently re-verified by the coordinator at
> `e576ae80` for n = 2, 3, 5, 8, 17: `whole_stream_reverse=False` in every case,
> `per_epoch_reverse=True` in every case. The refill reshuffles, so the reversal does not compose
> across epoch boundaries. The shipped docstring says "*within* each epoch, not across the stream."
>
> Two further prescriptions in the same brief were also wrong. S3's remedy called the 13-item list
> "the shared list" when there are **17** shared public members; and 5 of the 13
> (`primitive_unit`, `export_gaussians`, `frame_report`, `checkpoint`, `from_checkpoint`) are
> `rendering.py`'s, not the trainer's — so the implementer's own first draft, "the trainer-facing
> surface", was wrong too and was caught before commit. Member counts re-derived on live instances
> across four config combinations (3dgs/2dgs × appearance_dim 0/32): **17 shared, 5 Gaussians-only,
> 9 Scaffold-only**, stable in all four.

Two further instances found by sweeping the implementer's own files, both fixed in the same commit:

- `utils.py` stated the per-view visit spread as a single constant, "±17%". It is a **two-parameter**
  quantity: the relative sd of visit counts under sampling with replacement is
  `sqrt((n_views - 1) / max_steps)`, measured 3.1%–20.7% over the plausible range and agreeing with
  the closed form to within about a point. Replaced with the formula plus two anchored values.
- `scaffold.py`'s class docstring carried the *identical* five-of-seven omission as the S2 finding,
  four lines above the correction of it.

Deliberately left: `Scaffold.decode`'s summary line has the same shape, but its own bullets
immediately enumerate `log_scales` and `visible_ids`, so it is self-correcting in place.

> **A standing rule in every brief on this branch was stated with the wrong reason, and the
> implementer caught it.** The briefs say "the git index is shared across all worktrees." It is not:
> this worktree has its own index at `.git/worktrees/clean-splats/index` — confirmed, two distinct
> files with different sizes and mtimes from `.git/index`. The real hazard is that **peer sessions
> work inside this same worktree**, which shares one index among them: five peer commits landed
> during this agent's run (HEAD `25d2fed0` → `80efc10e`), invalidating its first control and forcing
> a re-baseline at the true parent. **The `git commit --only` discipline is unchanged and still
> mandatory** — only the justification was wrong. Fix the wording in later briefs; do not relax the
> rule.

**Task 18 outcome — spec review of `b6b9c7fe`, one blocking finding, closed at `5e0ceca3`.**

The engineering was upheld in full: nine paths moved off `splats.zarr` onto `ckpt.pt`, **zero readers
survive** the sweep, and **12 of 12 mutants died** on `tests/wrapper/test_splats_stage.py`, ten of
them with a unique killer. The three new tunable tests are the *correct* remedy shape — one knob per
test, each asserting the forwarded **value** rather than mere key presence, verified by mutating to a
different non-`None` value (`→99.0`) rather than to `None`, plus a swap case for positional mix-up
and a drop case for key presence. No instance of the defect class in them.

The one blocking finding was a **false claim in `docs/known-test-failures.md`**: the entry said the
splats cleanup *"never touches `configs/base.yaml`."* It does — `97c2fae0` is an ancestor of the
reviewed tip and edits that file. The implementer had split a scoped present-tense sentence and
upgraded one clause into an unbounded absolute, falsifying it, against an explicit plan instruction
to leave that clause alone.

> **INSTANCE TWELVE — and this one was in the REVIEWER's remedy, and in the plan's own instruction.**
> The reviewer prescribed restoring the clause to *"did not, at that point, touch `reconstructor.py`,
> `configs/base.yaml`, or this test file"* — the plan's original wording. The coordinator checked it
> before applying it. **Two of its three clauses are false:** `configs/base.yaml` was touched by
> `97c2fae0` and `tests/wrapper/test_splats_stage.py` by `b4e67d22`, both before `6611ef7c`. Only the
> `reconstructor.py` clause holds.
>
> The *conclusion* was right and the *evidence* was wrong — three times running, in three different
> artifacts. What shipped instead is the true and strictly stronger statement: both files were
> touched, but **only inside comments** (`CameraOptModule`→`CameraOpt` in a yaml comment;
> `prepare_training_target`→`prepare_target` in a test comment), so neither edit can change behavior
> and neither can raise a `KeyError`. That is checkable, and it survives someone re-running
> `git log -- <path>` and finding a hit.
>
> The lesson generalises past this branch: **"we did not touch X" is a claim about a whole history
> and ages badly; "our only edit to X was inside a comment" is a claim about content and does not.**
> Prefer provenance claims that stay true as the branch grows.

Findings recorded under the termination rule rather than fixed (none is an escape, a false claim, or
a gate break):

- The sweep found **6** hits, not the 7 reported — an overcount, not peer contamination (6 in both
  the pinned and the live tree). Both remaining test hits are *negative* guards asserting the store
  is not written, and correctly survive.
- **`tests/evals` was never gated** by the implementer despite both eval scripts changing. The
  reviewer ran it A/B: **121 passed / RC=0 on both sides**. Benign, but the gap was real.
- The commit report's control was cited at `efc4d7b1`; the true parent is **`8d88f478`**, six commits
  back. The numbers coincidentally match (4 failed / 92 passed), so the conclusion survives — but as
  reported it was not a control.
- Three further report errors: `known-test-failures.md` went **+19/−14 = net +5**, not "−33 net";
  **3** overlapping black hunks by line range, not 2; the commit message says "five new cases" while
  enumerating six arms (the sixth is an edited pre-existing test, not a new one).
- The notebooks' alarming raw **−520 lines** is entirely cleared output — source lines *grew* (+14,
  +2), 0 cells carry outputs or an `execution_count` on either notebook, and the parent had 6 and 10
  output-bearing cells. Nothing fabricated, no code lost.

## Review termination rule (adopted 2026-09-07, mid-execution)

Review rounds on this branch ran away: five fix rounds on Task 10, five on Task 14, three on Task
12. The cause is a **process defect, not a code defect** — every brief instructs the reviewer to hunt
for the defect class inside the remedy, and that instruction has **no termination condition**. Every
artifact has some imperfection, so every round finds one, so every round spawns another.

The findings were real, and the early ones were worth the cost: Task 17 round 1 caught mutants (MX2
identity K, MX3 inverted poses) that turned a render into pure background while the suite stayed
green. But by round 4 the findings were the choice of three variable names in an input to a helper
that only tests other tests, and the wide gate had been flat at 472/0/0/RC=0 for every round since.

**From here, a finding earns a fix round only if it meets one of:**

1. It is an escape in **shipping behavior** — a mutation of `collab_splats/**` that the suite does
   not catch, demonstrated by a measured survivor.
2. It is a **false statement** in a comment, docstring or plan claim — wrong, not merely narrow.
3. It **breaks a gate**: any nonzero failure, any nonzero skip, or a collected-count change with no
   corresponding test change.

**Everything else is recorded, not fixed.** Test-input strength on an already-green test file,
docstring scope, comment wording, style, and "this could be stated more precisely" go into *Open
observations carried to the final review* and are dispositioned once, in the final
whole-implementation review, by a reviewer who can see all of them together and weigh them against
each other. That reviewer may still decide some are worth closing — but as one batch with one
judgement, not as N serial rounds each spawning the next.

**Corollary for briefs.** Keep the "hunt for instance N+1" instruction — it has a measured 10-for-11
hit rate and it is why the escapes were found. Change only its *disposition*: the hunt reports, the
termination rule decides. And keep the standing line *"Do not trust this brief; measure every
prescription before you follow it"* — two of the last three rounds were caused by wrong prescriptions
in a coordinator brief, and both were caught only because the implementer measured instead of
complying.

**Task 12 outcome — spec re-review round 3 of `8d88f478` NOT APPROVED; closed by one fix at
`9ee2ee66` + `9ac21cfd`, on the gate rather than on a fourth re-review.**

The re-review confirmed round 2's two dispatched findings genuinely closed: `tests/splats` 370 → 371,
wide 471 → 472, 0 skipped everywhere, scope exactly `tests/splats/test_scaffold.py +138/−5` with zero
`collab_splats/` paths, the flake gone (25.0% nonzero over 60 isolated runs at the parent → **0/60**
at the child), and mutant M8's killers 3 → 10 with all seven armed sites asserting exact
`== model.n_primitives * model.cfg.n_offsets`. It failed on two findings, **both category 1** under
the termination rule — each is a measured survivor, not a strength preference:

- **B1 — instance N+1 inside the new test.** `test_export_gaussians_clamps_the_all_closed_opacity_...`
  closed with four round-trip assertions under the comment *"Every other raw form still round-trips
  on that row, so the clamp is the only difference"*, on an untouched fixture where `offsets` is
  exactly 0.0 and all six `scaling` columns are equal. The trap is documented **45 lines above** in
  the same file, in the arming comment of `..._writes_the_decoded_values_in_the_ply_s_raw_forms`.
  Coordinator re-measured both named survivors on a pinned tree, mutating the **export path only**:

  | export-path mutant | unarmed test | armed test |
  |---|---|---|
  | `means = anchors + offsets * scaling[:, None, :3]` → `offsets * 0.0` | PASS | **FAIL** |
  | `scales = scaling[:, 3:6]…` → `scaling[:, 0:3]…` | PASS | **FAIL** |

- **B2 — an unarmed caller whose stated reason the measurement contradicts.** See open observation 3,
  now closed. The reviewer's 7-of-120 figure reproduced **exactly**: `zero_anchor_grad=7`,
  `all_closed=7`, `both=7`.

Both fixes are test-only, in the one file T16's addendum had already withdrawn from T16's ownership.
**Task 12 closes on the gate** — 513 passed / 0 failed / 0 skipped / RC=0 — with no fourth re-review,
which is the termination rule's intent applied to its own first case: the fix is 27 lines of test
arming whose kill attribution the coordinator measured itself.

Non-blocking, recorded not fixed (pre-existing, both survive at parent *and* child): the clamp's
**upper** bound is unpinned (`clamp(1e-4, 1 − 1e-4)` → `clamp(1e-4, 10.0)` survives all 371), and the
per-anchor scale mapping is untestable in the only fixture (`repeat_interleave(K,0)` → `repeat(K,1)`
survives all 371). Seven corrections to the round-2 brief were also reported; the two that matter:
the true flake rate is **25.0%**, not the brief's 15% or the implementer's 10%, and **"M6 SIGFPEs
rather than failing cleanly" is wrong** — M6 fails first by two clean assertion failures and crashes
later in an unrelated test.

**Task 14 outcome — APPROVED at `ef37777e`. Task 14 is closed.**

Round 5's re-review reproduced every cell of the ordering table and went wider: **15** ordering
mutations of the helper's return all die on exactly one node id, against the five the brief claimed.
Both other columns reproduced independently — at the round-4 two-name input `list(...)[::-1]`
survives while the other four die; at the brief-prescribed `ALPHA/logger/BETA` input
`sorted(key=str.lower)` survives. No arm was weakened: each of the four still has a case that fires
only for it. Four real plants in `pgsr.py` each give `1 failed / 370 passed / RC=1`. Gates
371 / 472, 0 skipped, RC=0. Two corrections to the brief: mutating `ast.Assign` fails **6** tests,
not the 4 claimed (the argument is unaffected — attribution, not radius, is the bar), and the
"branch has moved past it" note missed that `e576ae80` is a **source** commit; the reviewer re-gated
the tip anyway and got 371/0/0/RC=0. One nit worth a half-sentence if the file is ever touched again:
"reversing a two-element source list lands back on sorted order" holds only for a descending source —
though the conclusion it supports is unconditionally true, proven exhaustively over all 56 ordered
pairs from an 8-name pool (**zero** kill both `list()` and `list()[::-1]`).

**Task 17 outcome — APPROVED at `2aa5d25d`. Task 17 is closed.**

All three rounding mutants die on one node id each (`.ceil()`, `.floor()`, bare `.byte()`), and both
arms are load-bearing — the reviewer built half-tests and measured `.ceil()` surviving the up half
alone and both `.floor()` and bare `.byte()` surviving the down half alone. Three cases per arm, so
neither arm has a single point of failure. Float exactness holds for every m = 1..7 as `Fraction`
equality, and the reviewer checked the *neighboring* ordering test's 0.2/0.4/0.6 too — those are not
exact as float32 but their products land on exactly 51/102/153, so that test's "tell none apart"
claim is true. The reviewer also rebuilt the parent state and got 80 passed / RC=0, confirming the
escape was real and this commit is what closes it. Gates 80 (mesh) and 472 (wide), 0 skips in **all
25** runs. Two nits, neither instance N+1 because both understate rather than overstate: "truncation
shows only on the up half" omits the tie, and "crosses a half at m = 4" is off by one step (it
*equals* a half at m=4). One commit-message-only overstatement: asserting 128 at the tie excludes a
tie-down rule but does not distinguish half-to-even from half-away-from-zero. Nothing shipped in the
file is wrong.

**Task 16 outcome — landed as `ad2e6f54`** (11 paths, +375/−74). New `tests/splats/test_model_interface.py`
plus four closed keyword-only boundaries and four strengthened `match=` guards. Wide gate
**472 → 513**, 0 failed, 0 skipped, RC=0; the +41 decomposes exactly as 36 + 2 + 1 + 1 + 1. The
unguarded-boundary sweep went from 10 guarded / 5 unguarded / 4 weak to **14 / 1 / 0**; two of the
four strengthened guards had **no `match=` at all**, so they passed on any `TypeError`. All 24
interface mutants (12 members × 2 classes) die, and the membership case **never fires cross-class** —
which is the evidence that the parametrisation is genuinely N-case rather than one case over an
aggregate; five members are killed by their own membership case alone. Dropping a module from the
`test_cu121_migration.py` width guard's `MODULES` list fails exactly that guard, confirming the
parametrised import test loses a case *silently* without it.

**Instance thirteen of the defect class, and it was in the plan's own prescription.** Task 16 Step 1
prescribes an AST width assertion of `consumed <= declared`, which **passes vacuously** the moment a
consumer renames its local away from `model`. Shipped as `consumed == declared`. Step 1's prescribed
round-trip test is also broken as written — `ckpt["config"]` is a dict lacking `sh_degree` and
`sh_degree_interval`, so it raises `KeyError`; shipped with `asdict(_config(model_class))`, matching
`rendering.py`'s own `config_dict = asdict(cfg)`.

Twelve further corrections to this plan, measured at `ad2e6f54` and recorded rather than applied
inline (the headings have drifted repeatedly and any line number written here rots again):

1. Task 16's heading is at line **7684**, not 7402; Task 17 at 8637, Task 18 at 9283.
2. `forward_backward_noise` is a **false positive** in both briefs' AST sweep — already guarded by an
   `inspect.Parameter.KEYWORD_ONLY` test. The sweep only detects `pytest.raises(TypeError)` `with`
   blocks. A `raises` test was added anyway.
3. `docs/splats.md` was **104** lines at dispatch, not 100.
4. The `configs/base.yaml` block ends at **238**, not 239.
5. Step 5's `grep splats.zarr → NONE LEFT` check is **wrong**: Task 18 deliberately shipped a
   retirement paragraph that mentions the name.
6. The plan says "append" to the CHANGELOG; that file's own header says **newest first**.
7. The CHANGELOG draft says "three attributes and six methods" — really **five and seven** — and
   "300 steps / three configs" — really **50 steps / seven configs, PASS 7/7 at `2d5c01d7`**.
8. `test_scaffold.py` had **96** tests, not the 66 the plan states.
9. The plan lists seven false `CLAUDE.md` clauses; there is an **eighth** it omits — the stale
   trainer loop order, corrected to `backward → one optimizer.step over model.optimizers +
   refine.optimizers → schedulers → model.post_backward`.
10. The `_field` rename counts were 56 / 16 / 5; measured **57 occurrences / 21 plan lines /
    6 blocks**, and the cited plan lines 7525/7529/7533 have drifted to 7807/7811/7815.
11. **Task 16-22 is refuted and must be re-scoped, not executed.** "Reduce the trainer side to a
    single delegation smoke test" would **delete real coverage**: `test_config_rejects_invalid` has
    2 of 6 operands with no losses-side equivalent, and `test_config_rejects_bad_decay`'s operands 3
    and 4 are trainer-side only. It was also blocked on ownership (`tests/splats/test_trainer.py` is
    not in the addendum's list).
12. `wc -c` disagrees with Python on `CLAUDE.md` byte counts by a constant 356 (41745 vs 41389).
    **Python is authoritative** — another instance of the standing rule that RTK mangles counting.

Two items were deferred out of Task 16 with bodies measured and ready to paste, and are owed to
whoever next opens these files: the `Scaffold.__init__` `lr_decay` keyword-only refusal test (message
pinned to `takes 7 positional arguments but 8 were given`, gammas measured 0.99309 set vs 0.95499
default at `max_steps=100`, so the assertion discriminates rather than approximating a constant), and
the single-file `_field` → `_scaffold` rename. Step 9's `graphify update .` was skipped — there is no
`graphify-out/` in this worktree, so it is owed to the main-checkout owner.

**Limit stated rather than claimed:** the four new refusal tests each pin `match=` to the message
measured from the real call, but proving the guard dies when the `*` is removed needs a
`collab_splats/` edit, which Task 16's brief banned. Not verified.

## Open observations carried to the final review

Four things surfaced during Tasks 12, 14 and 15 that belong to no task's remedy and were never
written down. None blocks a task. The first three were re-measured by the coordinator at `8fd2701e`,
the fourth at `ef37777e`; where the coordinator could not reproduce a reported number, that is said
rather than papered over.

**1. `box_sum` is the only def at any depth in `pgsr.py` without a docstring.** It sits at
`pgsr.py:190`, nested inside `lncc`. The repo rule in `CLAUDE.md` binds "every public function and
class", and a closure is not public, so this is not a style violation — but it is the file's sole
outlier, and a one-line summary would make `pgsr.py` uniform. Route it to Task 14's code-quality
review as a judgement call, not a defect.

**2. `collab_splats/splats/` holds FIVE nested closures, not the four Task 15's fix round reported.**
Measured by AST at `8fd2701e`: `box_sum` in `lncc` (`pgsr.py:190`), and `param_fn` / `optimizer_fn`
twice each in `scaffold.py` — inside `_append_anchors` (`:1039`, `:1042`) and inside `prune` (`:1106`,
`:1109`). All five are pre-existing and all four in `scaffold.py` are the gsplat strategy-callback
shape, which is upstream's API and not ours to change. The count discrepancy is the point worth
recording: **do not carry a checker's number without running the checker.** The coordinator's own
crude re-derivation of the companion "name-restatement" metric returned **20** hits against the three
reported, because a heuristic that asks whether a docstring's first line contains the words of the
function name flags legitimate summaries (`CameraOpt.camera` → "Apply the learned pose deltas on the
right of camera-to-world"). Whatever number goes in the final report must come from Task 15 Step 1's
actual checker, run at the SHA being reported.

**3. Task 12's two gradient tests were row-count insensitive — CLOSED at `9ee2ee66` and
`9ac21cfd`.** Both assertions collapsed an aggregate over 163 anchors to one boolean, and the
coordinator's measurement showed each one holding on a state its own name rejects, so both met
termination-rule category 1 (a measured survivor) rather than the "soft finding" this entry
originally called them:

| test | unarmed measurement | armed floor | assertion added |
|---|---|---|---|
| `::test_render_gradient_reaches_the_anchor_features` | 7 of 120 CUDA draws reached **zero** anchors and the test passed on all 7 — exactly the 7 all-closed draws | ≥133 of 163 over 200 draws, never zero | `(grad.abs().sum(dim=-1) > 0).any()` |
| `::test_decode_is_differentiable_into_the_mlps` | 6 of 200 CPU draws fell back to one slot and gradient reached **exactly 1** anchor; the test passed | ≥43 of 163 over 200 draws | `int((grad.abs().sum(dim=-1) > 0).sum()) > 1` |

Both are armed with `_open_every_offset`, so the fallback branch is out and the thresholds have
margin — the entry's caveat about needing a seeded fixture turned out to be avoidable: arming the
opacity head removes the variance that made a literal unsafe, without seeding anything.


**4. `_module_level_bindings` deduplicates, and nothing asserts it.** `return sorted(set(...))`
survives the full suite at `ef37777e` — measured, RC=0. Task 14's rounds 3-5 pinned the helper's
*ordering* against five mutations and its four node-shape arms against one discriminating case each,
but **multiplicity is a separate property** and no case has a repeated binding. It was deliberately
not folded into the round-5 sorted-order test: a duplicate-binding input would conflate two
properties in one case and cost the ordering test its single-killer attribution, which is the
standard the rest of this branch is held to. Closing it needs its own case — e.g. a source binding
the same name twice, asserting the result contains it once. Route to Task 14's code-quality review.
Note this is pre-existing, not something a fix round introduced.

**5. `evals/scripts/analyze_splats.py`'s two rewritten entry points hardcode `"cuda"` and have no
test.** `analyze_normals` and `depth_vs_gt` were rewritten onto `load_checkpoint` / `render_views` by
Task 18 and are referenced by no test anywhere. Task 18's reviewer smoke-ran the rewritten
`analyze_normals` on synthetic 3dgs and 2dgs checkpoints in a pinned tree and **both succeeded**
(3dgs 1672 opaque px, unit-norm fraction 1.0, angle median 50.33°; 2dgs 188 px, 43.29°), so they are
runtime-correct as shipped. The hardcoded device is a portability regression on a CPU-only host — but
it is consistent with the rest of `evals/scripts/`, so changing it is a decision about that whole
directory, not a Task 18 defect. Route to the final review.

**6. `keep &= confidence_mask(alpha, conf_percentile)` in `collab_splats/mesh/utils.py` has two arms
and no test separates them.** Measured by Task 17's round-2 reviewer: `&=` → `=` survives
`tests/mesh` (80 passed / RC=0). Separable and therefore a real gap — at `conf_percentile=100` the
mutant keeps zero-alpha pixels the shipped code drops. The `alpha > 0` arm alone IS pinned
(`alpha >= 0` → 79/1/0 RC=1). Pre-existing and outside Task 17's scope, which is why it was recorded
instead of fixed; it is also the counter-example that made the plan's "swept and found already
complete" sentence false, corrected above.

**7. The splats-adapter dtype assertion covers 2 of the 4 returns.**
`test_splats_adapter_reads_a_checkpoint` asserts `depths` float32 and `rgbs` uint8, while the
production docstring's Returns section states float32 for `c2w` and `intrinsics` as well. Measured:
casting either of those to float64 survives (80 passed / RC=0 each). Pre-existing, minor, same shape
as observation 6.

**8. Task 12's clamp upper bound and per-anchor scale mapping are unpinned.** Both survive all 371
`tests/splats` at the parent *and* the child, so neither was introduced by Task 12's rounds:
`clamp(1e-4, 1 − 1e-4)` → `clamp(1e-4, 10.0)`, and `repeat_interleave(K, 0)` → `repeat(K, 1)`. The
second is not closable in the only fixture this file has, which is the more interesting half — a
second fixture is exactly what `_field`'s docstring warns against, so closing it is a design call for
the final review rather than a test to add.

## Done When

- [ ] `collab_splats/splats/` is nine modules: `__init__.py`, `cameras.py`, `gaussian.py`, `scaffold.py`, `losses.py`, `pgsr.py`, `rendering.py`, `strategy.py`, `trainer.py`. No `outputs.py`, no `anchors.py`.
- [ ] `Gaussians` and `Scaffold` implement the same six methods, and `tests/splats/test_model_interface.py` proves it parametrised over both.
- [ ] `train()` contains no `if cfg.representation` and no `if cfg.primitive` branch.
- [ ] `splats.zarr` is neither written nor read anywhere outside `docs/superpowers/`.
- [ ] `ckpt.pt` round-trips: `load_checkpoint(path)` reproduces every render `train()` wrote.
- [ ] Parity `before` → `final` passes on all three configs.
- [ ] `pytest tests/splats tests/mesh tests/wrapper/test_splats_stage.py` is green.
- [ ] `docs/splats.md`, `configs/README.md`, `CLAUDE.md` and both tutorials describe the module that exists.

---

## Parity Baseline (measured, Task 2)

> **SUPERSEDED — the harness this table was measured with was measuring its own noise. The
> replacement baseline is the `before_v3` table below. Tasks 11, 13 and 19 all gate on the new
> one.** Read the rebuild note before running any parity check.

~~Captured on `clean/splats` @ `fd8094a2` (pre-refactor), A40, gsplat `d2f5c0f` + fused-ssim
`a7c48d6`, `tests/splats/synthetic.make_scene()` defaults, `STEPS = 300`, seed 0.~~

| config | psnr | ssim | n_gaussians |
| --- | --- | --- | --- |
| ~~`3dgs_vanilla`~~ | ~~16.558302~~ | ~~0.1434~~ | ~~200~~ |
| ~~`2dgs_vanilla`~~ | ~~15.695609~~ | ~~0.1367~~ | ~~200~~ |
| ~~`3dgs_scaffold`~~ | ~~17.906347~~ | ~~0.7718~~ | ~~163~~ |

### Harness rebuild (2026-09-06, during Task 5's review)

Two defects were found in the harness itself, both of which made it fail on trees that were
numerically identical. Neither was a refactor regression.

**1. At `STEPS = 300` the harness failed when compared against itself.** Running the *same
commit* twice and diffing the two runs measured this noise floor over three repeats per config:

| steps | `3dgs_vanilla` | `2dgs_vanilla` | `3dgs_scaffold` |
| --- | --- | --- | --- |
| 50 | 6.832e-09 | 9.934e-08 | 3.866e-07 (`head_ok` / `mean_ok` both True) |
| 300 | 6.659e-06 | 3.441e-04 | **1.810e-02 (`head_ok` False, `mean_ok` False)** |

The gate is 1e-3, so at 300 steps the scaffold configuration's self-noise sat **18× above the
gate** and the exported means did not match themselves. Cause: CUDA atomic nondeterminism in
gsplat's backward compounds over training steps — the spread grows three to four orders of
magnitude between 50 and 300 steps. `n_gaussians` was identical at every budget.

**`STEPS` is now 50.** This costs no coverage: gsplat's `refine_start_iter` is 500 for both
`MCMCStrategy` and `DefaultStrategy`, so **densification never ran at 300 steps either**.

**2. The harness seeded `torch` and `numpy` but not `random`.** The pgsr multi-view loss picks
its neighbor view with `random.randrange` on the **global** `random` module
(`trainer.py:586`, and identically at the pre-refactor tree), and nothing in production seeds
it. Unseeded, every process draws a different neighbor sequence: the `3dgs_pgsr` config
measured `dPSNR = 5.36e-03`, `means_ok = False` between two trees that agreed to 1e-7 on all
six other configs. With `random.seed(0)` added beside the existing `torch.manual_seed(0)` /
`np.random.seed(0)`, that config drops to `dPSNR = 3.35e-07`.

> This is a **production non-reproducibility**, not just a harness one: any run with
> `pgsr_multiview` enabled is unseeded and irreproducible. The harness papers over it; the
> trainer does not. Candidate fix for Task 14 (`pgsr.py` tidy), which is where
> `select_near_views` lives — but it is a behavior change, so it needs its own decision.

**3. The config set went from 3 to 7.** The original three all left `pose_opt` at its default
`true` and `normalize_scene` at `false`, so the denormalize path, the appearance path and the
pgsr path were all unexercised — Task 5's `denormalize` `AttributeError` would have passed
parity. The new set covers each: `3dgs_norm_nopose` and `scaffold_norm_nopose`
(`normalize_scene: true, pose_opt: false`), `3dgs_appearance` (`appearance_opt: true`), and
`3dgs_pgsr`.

> **`3dgs_pgsr` runs with `pose_opt: false`, and that is not a preference.** Measured
> 2026-09-06 against the **pre-refactor** tree at `edd1fafb`: `pgsr_multiview` together with
> `pose_opt: true` drives the pose deltas to NaN within 50 steps, and the final render then
> dies in `torch.linalg.inv` with
> `linalg.inv: (Batch element 0): The diagonal element 2 is zero`. The loss *value* stays
> finite throughout — the NaN is produced in the **backward** — and the `geo` / `ncc`
> sub-weights make no difference (`w1e-2` and `w1e-4` both fail with pose on, both pass with
> pose off; `pgsr_normal` alone is fine at either setting). **This is a pre-existing production
> bug, not caused by this refactor, and `splats.pose_opt` defaults `true`, so any user who
> enables `pgsr_multiview` hits it.** It belongs in `docs/known-test-failures.md` and is out of
> scope for this plan.

### Parity baseline `before_v3` (measured, use this one)

Pre-refactor tree at `edd1fafb`, extracted with `git archive` to
`scratchpad/base_tree/`; A40, gsplat `d2f5c0f` + fused-ssim `a7c48d6`,
`tests/splats/synthetic.make_scene()` defaults, `STEPS = 50`, `torch` / `numpy` / `random` all
seeded 0. Compare with `splats_parity.py --compare before_v3 <label>`; the gate is unchanged —
dPSNR <= 1e-3, `means_head` allclose(rtol=1e-4, atol=1e-6), equal `n_gaussians`.

| config | psnr | ssim | n_gaussians |
| --- | --- | --- | --- |
| `3dgs_vanilla` | 17.586695 | 0.5226 | 200 |
| `2dgs_vanilla` | 18.208287 | 0.5905 | 200 |
| `3dgs_scaffold` | 17.180856 | 0.7617 | 163 |
| `3dgs_norm_nopose` | 17.576920 | 0.5207 | 200 |
| `scaffold_norm_nopose` | 17.236253 | 0.7639 | 166 |
| `3dgs_appearance` | 17.624784 | 0.5441 | 200 |
| `3dgs_pgsr` | 17.589819 | 0.5233 | 200 |

`compare()` now fails on a config-set mismatch (`set(a) ^ set(b)`) instead of silently
skipping the configs one side is missing.

**Result after Tasks 1-5 (`after_task5_v3` @ `1d1d036b`): PARITY OK, all seven configs, worst
dPSNR 3.35e-07.**

Mesh-adapter reading from the `splats.zarr` path (Task 2 Step 4, the last moment it can be
measured — Task 9 retires the store), `_splats_to_tsdf_inputs` on
`parity_before/3dgs_vanilla/splats.zarr`, full fingerprint in
`scratchpad/mesh_inputs_before.json`:

- shapes: `depths (8, 64, 64)`, `rgbs (8, 64, 64, 3)`, `c2w (8, 4, 4)`, `K (8, 3, 3)`
- `depth_mean = 2.673015594482422`, `depth_nonzero_frac = 1.0`

Task 17 must reproduce these from `ckpt.pt` rendering, not from a store.
