"""Regenerate sglang_b_markers.patch's frontier_b_markers.py hunk from
marker_runtime.py, keeping the other 8 files' hunks byte-identical.

Code review ("Parts to simplify"): marker_runtime.py is the maintained
source; the patch's own copy of it was previously kept in sync by hand (a
one-off shell diff+cat dance, repeated -- and occasionally forgotten --
every time marker_runtime.py changed). This script makes that
regeneration a single reproducible step: diff -u /dev/null against the
current marker_runtime.py, reformatted to the patch's path convention, with
everything from the second file's "--- a/..." header onward in the existing
patch carried over unchanged.

Never guesses which hunk is which: fails loud if the existing patch's first
hunk is not exactly the frontier_b_markers.py "new file" diff this script
expects to replace.
"""
import argparse
import subprocess
from pathlib import Path

from b_common import require

NEW_FILE_HEADER = "--- /dev/null\n+++ b/python/sglang/srt/frontier_b_markers.py\n"


def build_patch(marker_runtime_path, existing_patch_path):
    existing = Path(existing_patch_path).read_text()
    require(existing.startswith("--- /dev/null\n"),
            f"{existing_patch_path}: first hunk is not a new-file diff; refusing to guess which "
            "hunk to replace")
    # Everything from the *second* file's own "--- a/..." header onward.
    marker = "\n--- a/"
    cut = existing.find(marker, len(NEW_FILE_HEADER))
    require(cut != -1, f"{existing_patch_path}: could not find the second file's hunk header")
    rest_of_patch = existing[cut + 1:]

    diff = subprocess.run(["diff", "-u", "/dev/null", str(marker_runtime_path)],
                           capture_output=True, text=True)
    require(diff.returncode == 1, f"diff against {marker_runtime_path} produced no changes")
    lines = diff.stdout.splitlines(keepends=True)
    require(len(lines) >= 3 and lines[0].startswith("--- /dev/null") and lines[1].startswith("+++ "),
            "unexpected diff output shape")
    new_hunk = NEW_FILE_HEADER + "".join(lines[2:])

    return new_hunk + rest_of_patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--marker-runtime", default=str(Path(__file__).with_name("marker_runtime.py")))
    p.add_argument("--patch", default=str(Path(__file__).with_name("sglang_b_markers.patch")),
                   help="existing patch to read the other 8 files' hunks from, and to overwrite")
    a = p.parse_args()
    new_patch = build_patch(a.marker_runtime, a.patch)
    Path(a.patch).write_text(new_patch)
    print(f"wrote {a.patch} ({len(new_patch)} bytes)")


if __name__ == "__main__":
    main()
