#!/usr/bin/env python3
"""Build STORY.ipynb from STORY.md + figures/*.png.

Every '## ' section of STORY.md becomes one or more markdown cells; each figure referenced in the text
(`fNN_*.png`) is embedded as a base64 attachment in its own markdown cell right after the paragraph that
mentions it, so the notebook is self-contained (no dependency on the figures/ folder).

    python3 build_story_notebook.py            # writes STORY.ipynb next to STORY.md
"""
import base64
import os
import re
import sys

import nbformat
from nbformat.v4 import new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
FIG_RE = re.compile(r"`?(f\d\d_[A-Za-z0-9_]+\.png)`?")


def image_cell(fname, caption):
    path = os.path.join(FIG_DIR, fname)
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    cell = new_markdown_cell(f"![{caption}](attachment:{fname})\n\n<sub>`{fname}`</sub>")
    cell["attachments"] = {fname: {"image/png": b64}}
    return cell


def caption_from_item(item, fname):
    text = " ".join(l.strip() for l in item).lstrip("-* ")
    text = text.replace(f"`{fname}`", "")
    text = re.sub(r"\*\*Figure\*\*", "", text)
    text = text.replace("**", "").strip(" —-–:.")
    text = re.sub(r"\s+", " ", text)
    return text[:220] if text else fname


def items_of(block):
    """Split a paragraph block into bullet items (continuation lines joined to their bullet)."""
    items, cur = [], []
    for l in block:
        if re.match(r"\s*[-*] ", l) and cur:
            items.append(cur); cur = []
        cur.append(l)
    if cur:
        items.append(cur)
    return items


def build(md_path, out_path):
    lines = open(md_path, encoding="utf-8").read().split("\n")
    cells, buf, block = [], [], []
    seen = set()

    def flush():
        if any(l.strip() for l in buf):
            cells.append(new_markdown_cell("\n".join(buf).strip("\n")))
        buf.clear()

    def end_block():
        if not block:
            return
        buf.extend(block)
        figs = [m for l in block for m in FIG_RE.findall(l) if os.path.exists(os.path.join(FIG_DIR, m))]
        if figs:
            flush()
            for item in items_of(block):
                for f in FIG_RE.findall(" ".join(item)):
                    if f in seen or not os.path.exists(os.path.join(FIG_DIR, f)):
                        continue
                    seen.add(f)
                    cells.append(image_cell(f, caption_from_item(item, f)))
        block.clear()

    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            block.append(line); continue
        if in_fence:
            block.append(line); continue
        if line.startswith("## ") or line.startswith("# "):
            end_block(); flush()
        if line.strip() == "---":
            end_block(); continue
        if not line.strip():
            end_block(); buf.append(line); continue
        block.append(line)
    end_block(); flush()

    missing = sorted(set(os.listdir(FIG_DIR)) - seen)
    if missing:
        print("figures not referenced in STORY.md:", missing, file=sys.stderr)

    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "markdown"}, "story": {
        "source": os.path.basename(md_path), "built_from": "build_story_notebook.py"}})
    nbformat.validate(nb)
    nbformat.write(nb, out_path)
    print(f"wrote {out_path}: {len(cells)} cells, {len(seen)} figures, {os.path.getsize(out_path)/1e6:.1f} MB")


if __name__ == "__main__":
    build(os.path.join(HERE, "STORY.md"), os.path.join(HERE, "STORY.ipynb"))
