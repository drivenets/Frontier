#!/usr/bin/env python3
"""Render STORY.md as a Confluence HTML+ body (the format the Atlassian MCP page tools accept).

    python3 build_confluence_html.py [--media media_map.json] [--out STORY.confluence.html]

Without --media every figure reference becomes a visible placeholder paragraph; with a media map
({"fNN_name.png": {"id": MEDIA_ID, "collection": COLLECTION}, ...}, as returned by the attachment upload)
each reference becomes a real media-single figure with the caption underneath.
Requires pandoc (gfm -> html) for the markdown itself.
"""
import argparse
import html
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_RE = re.compile(r"<code>(f\d\d_[A-Za-z0-9_]+\.png)</code>")
TAG_RE = re.compile(r"<[^>]+>")


CAPTIONS = json.load(open(os.path.join(HERE, "captions.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "captions.json")) else {}
_FIGNO = {}


def figure_html(fname, caption, media):
    if media and fname in media:
        m = media[fname]
        c = CAPTIONS.get(fname)
        n = _FIGNO.setdefault(fname, len(_FIGNO) + 1)
        cap = html.escape(f"Fig. {n}: {c['title']}") if c else html.escape(TAG_RE.sub("", caption)).strip(" —-–:.")
        dims = (f' data-width="{m["width"]}" data-height="{m["height"]}"' if m.get("width") and m.get("height") else "")
        fig = (f'<figure data-type="media-single" data-layout="center" data-width="544" data-width-type="pixel">'
               f'<div data-type="media" data-media-type="file" data-id="{m["id"]}" data-collection="{m["collection"]}" '
               f'data-alt="{fname}"{dims}></div>{"<figcaption>" + cap + "</figcaption>" if cap else ""}</figure>')
        if c:
            fig += (f"<p><strong>Fig. {n}.</strong> <strong>What it shows.</strong> " + html.escape(c["shows"]) + " <strong>Where to look.</strong> " + html.escape(c["look"]) +
                    " <strong>Why it is in the story.</strong> " + html.escape(c["why"]) + "</p>")
        return fig
    c = CAPTIONS.get(fname)
    n = _FIGNO.setdefault(fname, len(_FIGNO) + 1)
    out = (f'<p><span data-type="status" data-color="yellow">image pending</span> <strong>Fig. {n}: {html.escape(c["title"]) if c else fname}</strong> — '
           f'attach <code>figures/{fname}</code> here.</p>')
    if c:
        out += (f"<p><strong>Fig. {n}.</strong> <strong>What it shows.</strong> " + html.escape(c["shows"]) + " <strong>Where to look.</strong> " + html.escape(c["look"]) +
                " <strong>Why it is in the story.</strong> " + html.escape(c["why"]) + "</p>")
    return out


def convert(md_text, media):
    body = md_text.split("\n", 1)[1] if md_text.startswith("# ") else md_text   # page title carries the H1
    out = subprocess.run(["pandoc", "-f", "gfm", "-t", "html", "--wrap=none"], input=body, text=True,
                         capture_output=True, check=True).stdout
    # pandoc code blocks -> Confluence code blocks
    out = re.sub(r'<div class="sourceCode"[^>]*><pre class="sourceCode (\w+)"><code[^>]*>', r'<pre><code class="language-\1">', out)
    out = out.replace("</code></pre></div>", "</code></pre>")
    out = re.sub(r'<pre class="(\w+)"><code>', r'<pre><code class="language-\1">', out)
    # strip pandoc's syntax-highlighting spans/anchors inside code blocks (Confluence highlights itself)
    def clean_code(m):
        inner = re.sub(r"<a [^>]*></a>", "", m.group(2))
        inner = re.sub(r"<span[^>]*>", "", inner).replace("</span>", "")
        return m.group(1) + inner + "</code></pre>"
    out = re.sub(r'(<pre><code class="language-\w+">)(.*?)</code></pre>', clean_code, out, flags=re.S)
    out = re.sub(r"<colgroup>.*?</colgroup>", "", out, flags=re.S)
    out = re.sub(r'<h(\d) id="[^"]*">', r"<h\1>", out)
    out = re.sub(r"<(td|th) style=\"[^\"]*\">", r"<\1>", out)
    out = re.sub(r'<tr class="[^"]*">', "<tr>", out)

    # number figures in document order, independent of the order the replacements below run in
    _FIGNO.clear()
    for f in FIG_RE.findall(out):
        _FIGNO.setdefault(f, len(_FIGNO) + 1)

    # figure references: list items or paragraphs that name a figure file
    def li_repl(m):
        inner = m.group(1)
        figs = FIG_RE.findall(inner)
        if not figs:
            return m.group(0)
        caption = FIG_RE.sub("", inner).strip()
        caption = re.sub(r"^<p>(.*)</p>$", r"\1", caption, flags=re.S)   # loose lists already wrap items in <p>
        caption = re.sub(r"^\s*[—–-]\s*", "", caption.strip())
        # hoist the figure out of the list: Confluence's editor drops media nodes nested in list items on save
        return "</ul><p>" + caption + "</p>" + "".join(figure_html(f, caption, media) for f in figs) + "<ul>"
    out = re.sub(r"<li>(.*?)</li>", li_repl, out, flags=re.S)
    out = re.sub(r"<ul>\s*</ul>", "", out)

    def p_repl(m):
        inner = m.group(1)
        figs = FIG_RE.findall(inner)
        if not figs:
            return m.group(0)
        caption = re.sub(r"<strong>Figure</strong>\s*", "", FIG_RE.sub("", inner))
        caption = re.sub(r"^\s*[—–-]\s*", "", caption.strip())
        return "<p>" + caption + "</p>" + "".join(figure_html(f, caption, media) for f in figs)
    out = re.sub(r"<p>(.*?)</p>", p_repl, out, flags=re.S)

    toc = ('<div data-type="extension" data-extension-key="toc" data-extension-type="com.atlassian.confluence.macro.core" '
           'data-parameters=\'{"macroParams":{"maxLevel":{"value":"2"}},"macroMetadata":{"schemaVersion":{"value":"1"},"title":"Table of Contents"}}\'></div>')
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        head = "HEAD"
    intro = ('<div data-type="panel-info"><p>Source: <code>profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/STORY.md</code> '
             f'on branch <code>smatar/qwen3-30b-mi355-profiling</code> of the Frontier fork (commit <code>{head}</code>); the same folder holds the '
             'figures, the notebook <code>STORY.ipynb</code> and the scripts that regenerate them. Numbered references such as '
             '<code>05_</code> or <code>09_</code> §3a point at the investigation documents in the parent directory.</p></div>')
    return toc + intro + out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", help="JSON file mapping figure file name -> {id, collection}")
    ap.add_argument("--out", default=os.path.join(HERE, "STORY.confluence.html"))
    args = ap.parse_args()
    media = json.load(open(args.media)) if args.media else None
    md = open(os.path.join(HERE, "STORY.md"), encoding="utf-8").read()
    body = convert(md, media)
    open(args.out, "w", encoding="utf-8").write(body)
    n_media = len(re.findall(r'data-type="media-single"', body)); n_pending = body.count("image pending")
    print(f"wrote {args.out}: {len(body)} chars, {n_media} embedded figures, {n_pending} placeholders")


if __name__ == "__main__":
    main()
