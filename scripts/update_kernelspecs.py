"""Update all notebook kernelspecs to Python 3.11 (nerfstudio env)."""
import json
import sys
from pathlib import Path

ROOTS = [
    Path("/workspace/collab-splats"),
    Path("/workspace/collab-splats/.claude/worktrees/docs-site"),
    Path("/workspace/collab-splats/.worktrees/bae-parity"),
    Path("/workspace/collab-splats/.worktrees/model-free-query-mesh"),
]

SKIP_DIRS = {".claude/worktrees/agent-", ".git"}

NEW_KERNELSPEC = {
    "display_name": "Python 3 (nerfstudio)",
    "language": "python",
    "name": "python3",
}
NEW_LANG_INFO = {
    "name": "python",
    "version": "3.11.0",
}

updated = []
skipped = []

for root in ROOTS:
    if not root.exists():
        print(f"SKIP (missing): {root}")
        continue
    for nb_path in sorted(root.rglob("*.ipynb")):
        # Skip agent worktrees and .git
        rel = str(nb_path.relative_to(root))
        if any(skip in str(nb_path) for skip in [".claude/worktrees/agent-", "/.git/"]):
            skipped.append(str(nb_path))
            continue
        # Skip checkpoints
        if ".ipynb_checkpoints" in str(nb_path):
            skipped.append(str(nb_path))
            continue

        try:
            text = nb_path.read_text(encoding="utf-8")
            nb = json.loads(text)
        except Exception as e:
            print(f"ERROR reading {nb_path}: {e}")
            continue

        meta = nb.setdefault("metadata", {})
        old_ks = meta.get("kernelspec", {})
        old_li = meta.get("language_info", {})

        changed = False
        if old_ks != NEW_KERNELSPEC:
            meta["kernelspec"] = NEW_KERNELSPEC
            changed = True
        if old_li.get("version") != "3.11.0" or old_li.get("name") != "python":
            meta["language_info"] = {**old_li, **NEW_LANG_INFO}
            changed = True

        if changed:
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            updated.append(str(nb_path))

print(f"\nUpdated: {len(updated)} notebooks")
print(f"Skipped: {len(skipped)} (agent worktrees / checkpoints)")
for p in updated:
    print(f"  + {p}")
