# Omega LC verify-layer sweep — d5_long (baseline ATE = 0.0308m)

| layer | lc ATE (m) |
|---|---|
| 8 | 0.547 |
| 12 | 0.708 |
| 16 (default) | 0.582 |
| 24 | invalid (exceeds depth; lc failed) |

**Every valid verify layer is catastrophic** (17-23x baseline). LC failure is
loop-edge / pose-graph corruption, NOT verify-layer calibration: the layer only
gates which candidates pass, so its best-case outcome is reject-all (LC->no-op),
never improvement. Omega lc != baseline => loops ARE applied and wreck the graph
(contrast spark/mapanything: 0 loops applied => no-op).
