"""LC decision serialization for the parity harness.

Routed through evals/runners/lc_parity_common.py (not eval_gt.py directly):
importing eval_gt in this worktree raises ModuleNotFoundError("modules") from
collab_splats/localization/extractors.py at import time (pre-existing env issue,
one of the 17 known-failing tests/evals/test_eval_gt_helpers.py cases). eval_gt.py
re-exports/imports _serialize_lc_decisions from here.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Local `evals/` is shadowed by an installed `evals` pip package; insert the
# evals dir on sys.path and import the module directly (sibling-test convention,
# mirrors tests/evals/test_lc_parity_common.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from runners.lc_parity_common import _serialize_lc_decisions


def test_serialize_lc_decisions():
    matches = [
        SimpleNamespace(
            similarity_score=0.41,
            query_submap_id=3,
            detected_submap_id=0,
            query_frame_idx=2,
            detected_frame_idx=7,
            accepted=True,
            reject_reason=None,
        ),
        SimpleNamespace(
            similarity_score=0.80,
            query_submap_id=3,
            detected_submap_id=1,
            query_frame_idx=0,
            detected_frame_idx=1,
            accepted=False,
            reject_reason="no_joint_poses",
        ),
    ]
    out = _serialize_lc_decisions(matches)
    assert out[0]["accepted"] is True and out[0]["reject_reason"] is None
    assert out[1]["reject_reason"] == "no_joint_poses"
    assert out[1]["l2_score"] == 0.80
    # loops_applied = accepted count
    assert sum(d["accepted"] for d in out) == 1
