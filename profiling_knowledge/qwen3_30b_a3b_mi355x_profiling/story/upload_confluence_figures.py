#!/usr/bin/env python3
"""Attach figures/*.png to a Confluence page and write the media map that build_confluence_html.py --media consumes.

    export ATLASSIAN_EMAIL=you@drivenets.com
    export ATLASSIAN_API_TOKEN=...            # https://id.atlassian.com/manage-profile/security/api-tokens
    python3 upload_confluence_figures.py --page-id 7166984214 [--site drivenets.atlassian.net]
    python3 build_confluence_html.py --media media_map.json      # then update the page body with STORY.confluence.html

Uses the Confluence Cloud REST API (POST /wiki/rest/api/content/{id}/child/attachment, minorEdit) — re-running replaces
an attachment of the same name instead of duplicating it. Only the standard library is needed.
"""
import argparse
import base64
import glob
import json
import mimetypes
import os
import sys
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def request(url, auth, method="GET", data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Basic " + auth)
    req.add_header("X-Atlassian-Token", "nocheck")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode() or "{}")


def multipart(fname, blob, comment):
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode() + blob + b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"minorEdit\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"comment\"\r\n\r\n{comment}\r\n".encode(),
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-id", required=True)
    ap.add_argument("--site", default="drivenets.atlassian.net")
    ap.add_argument("--out", default=os.path.join(HERE, "media_map.json"))
    args = ap.parse_args()
    email, token = os.environ.get("ATLASSIAN_EMAIL"), os.environ.get("ATLASSIAN_API_TOKEN")
    if not (email and token):
        sys.exit("set ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN")
    auth = base64.b64encode(f"{email}:{token}".encode()).decode()
    base = f"https://{args.site}/wiki/rest/api/content/{args.page_id}/child/attachment"

    media = {}
    for path in sorted(glob.glob(os.path.join(FIG_DIR, "f*.png"))):
        fname = os.path.basename(path)
        body, ctype = multipart(fname, open(path, "rb").read(), "story figure (make_story_figures.py)")
        # PUT to .../child/attachment creates or updates by file name
        resp = request(base, auth, method="PUT", data=body, headers={"Content-Type": ctype})
        att = resp["results"][0] if "results" in resp else resp
        ext = att.get("extensions", {})
        media[fname] = {"id": ext.get("fileId"), "collection": ext.get("collectionName") or f"contentId-{args.page_id}",
                        "attachment_id": att.get("id")}
        print(f"{fname}: attachment {att.get('id')} media {ext.get('fileId')}")
    json.dump(media, open(args.out, "w"), indent=1)
    missing = [k for k, v in media.items() if not v["id"]]
    print(f"wrote {args.out} ({len(media)} entries" + (f", {len(missing)} without a media id: {missing}" if missing else "") + ")")


if __name__ == "__main__":
    main()
