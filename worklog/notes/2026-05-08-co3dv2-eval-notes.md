# Evaluation Notes

## CO3Dv2 BA Gap Investigation (2026-05-08)

### Key findings

**Preprocessing (ratio vs square):**
- `load_and_preprocess_images_square` fixes the BA regression at 50 frames
- With ratio: ba (70.1) < baseline (74.4) — BA hurts
- With square: ba (72.9) > baseline (70.8) — BA helps
- ba_hightrack marginal difference: ratio=81.74, square=83.12 (+1.4pts)
- **Takeaway:** square preprocessing is the paper-matching setup; use it for evals

**Track density (`ba_hightrack`) is the dominant factor:**
- Default BA (max_query_pts=2048, query_frame_num=5): AUC@30 ≈ 70–73
- ba_hightrack (max_query_pts=4096, query_frame_num=8): AUC@30 ≈ 82–83
- ~10pt improvement from denser tracks alone
- Paper's 90.0 likely uses even denser tracks or longer query windows

**Full-sequence (202 frames) single-pass fails — but not because the model can't handle it:**
- RPE (local error) stays similar: 50fr=0.137m, 202fr=0.161m
- VGGT-X produces good LOCAL poses at 202 frames
- But ATE (global) explodes: 50fr=0.81m → 202fr=5.8m
- Root cause: **cumulative drift** over 202 frames, not model failure
- BA with query_frame_num=5–8 only spans nearby frames → no long-range constraints → can't correct global drift
- BA identical to baseline at 202 frames (all conditions ≈ 19 AUC@30)
- **Fix: use `--submap_size 50` (windowed inference)** — processes 50-frame windows, stitches globally

**Summary table — seq1 (110_13051_23361), apple category:**

| Preproc | Frames | Mode | Condition | AUC@30 |
|---|---|---|---|---|
| ratio | 50 | single-pass | baseline | 74.4 |
| ratio | 50 | single-pass | ba | 70.1 |
| ratio | 50 | single-pass | ba_hightrack | 81.7 |
| square | 50 | single-pass | baseline | 70.8 |
| square | 50 | single-pass | ba | 72.9 |
| square | 50 | single-pass | ba_hightrack | **83.1** |
| square | 202 | single-pass | baseline | 19.4 |
| square | 202 | single-pass | ba | 19.3 |
| square | 202 | single-pass | ba_hightrack | 19.3 |
| square | 202 | windowed (submap=50) | baseline | TBD |
| square | 202 | windowed (submap=50) | ba_hightrack | TBD |

Paper targets: VGGT init = 88.2, VGGT+BAE = 90.0

### Recommended eval config going forward

```bash
# Short sequences (≤ 100 frames): single-pass
python evals/eval_gt.py --dataset co3dv2 --max_frames 50 --conditions ba_hightrack

# Long sequences (> 100 frames): windowed
python evals/eval_gt.py --dataset co3dv2 --submap_size 50 --conditions baseline ba_hightrack
```

With `load_and_preprocess_images_square` active in `vggtx.py`.

### Open questions

1. Does windowed (submap=50) + ba_hightrack on 202 frames approach paper's 88.2/90.0?
2. Can increasing `query_frame_num` further (e.g., 16) close the remaining gap?
3. Does the gap persist across categories (only apple tested so far)?
