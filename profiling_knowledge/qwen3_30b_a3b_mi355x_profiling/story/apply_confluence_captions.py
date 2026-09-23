#!/usr/bin/env python3
"""Add captions to the figures of the Confluence story page.

Input: the page body fetched in Confluence HTML+ format (getConfluencePage contentFormat=html), in which every figure is a
<figure data-type="media-single"> whose media node carries data-alt="fNN_...png". For each figure this script:
  1. drops the terse one-line lead-in paragraph that immediately precedes it (the bullet text left over from STORY.md),
  2. inserts <figcaption>Fig. N: title</figcaption> inside the figure,
  3. appends a paragraph after the figure: What it shows / Where to look / Why it is in the story (from captions.json).
Everything else (media ids, layout, widths, local ids) is left untouched.

    python3 apply_confluence_captions.py page.html [--out page_captioned.html]
"""
import argparse
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_RE = re.compile(r'<figure[^>]*data-type="media-single"[^>]*>(.*?)</figure>', re.S)
ALT_RE = re.compile(r'data-alt="(f\d\d_[A-Za-z0-9_]+\.png)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page_html")
    ap.add_argument("--out")
    ap.add_argument("--captions", default=os.path.join(HERE, "captions.json"))
    args = ap.parse_args()
    caps = json.load(open(args.captions, encoding="utf-8"))
    src = open(args.page_html, encoding="utf-8").read()

    out, pos, n = [], 0, 0
    for m in FIG_RE.finditer(src):
        fig = m.group(0)
        alt = ALT_RE.search(fig)
        if not alt or alt.group(1) not in caps:
            continue
        c = caps[alt.group(1)]
        n += 1
        before = src[pos:m.start()]
        # drop an immediately preceding short lead-in paragraph (the leftover bullet text), keep everything else
        before = re.sub(r"<p>(?:(?!</p>).){0,400}</p>\s*$", "", before, flags=re.S)
        out.append(before)
        if "<figcaption>" in fig:
            fig = re.sub(r"<figcaption>.*?</figcaption>", "", fig, flags=re.S)
        cap = f"<figcaption>Fig. {n}: {html.escape(c['title'])}</figcaption>"
        fig = fig.replace("</figure>", cap + "</figure>")
        out.append(fig)
        out.append(f"<p><strong>Fig. {n}.</strong> <strong>What it shows.</strong> " + html.escape(c["shows"]) +
                   " <strong>Where to look.</strong> " + html.escape(c["look"]) +
                   " <strong>Why it is in the story.</strong> " + html.escape(c["why"]) + "</p>")
        pos = m.end()
    out.append(src[pos:])
    result = "".join(out)
    dest = args.out or os.path.splitext(args.page_html)[0] + "_captioned.html"
    open(dest, "w", encoding="utf-8").write(result)
    print(f"captioned {n} figures -> {dest} ({len(result)} chars)")


if __name__ == "__main__":
    main()
